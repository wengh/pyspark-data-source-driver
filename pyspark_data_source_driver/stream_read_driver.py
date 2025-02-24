import logging
import sys
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import pyarrow as pa
from pyspark.sql.datasource import DataSource, DataSourceStreamReader, InputPartition
from pyspark.sql.datasource_internal import _SimpleStreamReaderWrapper
from pyspark.sql.datasource_internal import _streamReader
from pyspark.sql.pandas.types import to_arrow_schema
from pyspark.sql.types import StructType
from pyspark.sql.worker.plan_data_source_read import records_to_arrow_batches
from pyspark.worker_util import pickleSer
from typeguard import typechecked

from pyspark_data_source_driver.read_driver import ReadDriverBase

if TYPE_CHECKING:
    from hypothesis.stateful import RuleBasedStateMachine


logger = logging.getLogger(__name__)


@typechecked  # Enable runtime type checking
class StreamReadDriver(ReadDriverBase):
    """
    A lightweight driver to read data from a streaming data source.

    The read result is an infinite iterator of `pyarrow.Table`s.

    It tries its best to enforce the Python Data Source API requirements, by:
    - Pickling and unpickling objects to simulate the behavior of Spark's serialization.
    - Enforce type hints using `typeguard`.

    Examples
    --------

    Define a simple streaming data source and its reader:
    >>> from pyspark.sql.datasource import DataSource, DataSourceStreamReader, InputPartition
    >>> from pyspark.sql.types import IntegerType, StructField, StructType
    >>> class CounterDataSource(DataSource):
    ...     def schema(self):
    ...         return StructType([StructField("id", IntegerType())])
    ...
    ...     def streamReader(self, schema: StructType):
    ...         return CounterDataSourceStreamReader(self.options)
    ...
    >>> class CounterDataSourceStreamReader(DataSourceStreamReader):
    ...     def __init__(self, options):
    ...         self.batch_size = int(options.get("batch_size", 3))
    ...         self.current = 0
    ...
    ...     def initialOffset(self):
    ...         return {"offset": 0}
    ...
    ...     def latestOffset(self):
    ...         self.current += self.batch_size
    ...         return {"offset": self.current}
    ...
    ...     def partitions(self, start: dict, end: dict):
    ...         return [InputPartition(i) for i in range(start["offset"], end["offset"])]
    ...
    ...     def read(self, partition):
    ...         yield (partition.value,)
    ...

    Read data from the data source.
    >>> from pyspark_data_source_driver import StreamReadDriver
    >>> iterator = iter(StreamReadDriver(CounterDataSource).load())
    >>> for i in range(2):
    ...     print(next(iterator).to_pydict())
    {'id': [0, 1, 2]}
    {'id': [3, 4, 5]}

    ### Advanced Usage

    Manually control the sequence of operations:
    >>> stream = StreamReadDriver(CounterDataSource).load()
    >>> # Get latest offset 3 times.
    >>> stream.update_offsets()
    {'offset': 3}
    >>> stream.update_offsets()
    {'offset': 6}
    >>> stream.update_offsets()
    {'offset': 9}
    >>> # Read from offset 3 to 9, skipping the initial offset 0.
    >>> stream.read(skip=1).to_pydict()
    {'id': [3, 4, 5, 6, 7, 8]}
    >>> # Commit does nothing because offset 0 to 3 is not read yet.
    >>> stream.commit()
    0
    >>> # Read from offset 0 to 6, limiting the number of offsets to read to 2.
    >>> stream.read(max_count=2).to_pydict()
    {'id': [0, 1, 2, 3, 4, 5]}
    >>> # Commit all 3 offsets read.
    >>> stream.commit()
    3
    """

    def load(self, path: Optional[str] = None) -> "StreamReaderDriver":
        data_source, schema = super().load(path)
        return StreamReaderDriver(data_source, schema)

    def hypothesis_state_machine(self) -> Type["RuleBasedStateMachine"]:
        """
        Get the Hypothesis state machine for model-based testing.

        Requires the `hypothesis` package to be installed.

        Returns
        -------
        Type[RuleBasedStateMachine]
            A Hypothesis state machine class.

        Examples
        --------
        Usage in pytest:
        >>> # tests/test_counter.py
        >>> from pyspark_data_source_driver.examples import CounterDataSource
        >>> TestCounter = (
        ...    StreamReadDriver(CounterDataSource)
        ...    .hypothesis_state_machine()
        ...    .TestCase
        ... )
        """
        import hypothesis.strategies as st
        from hypothesis.stateful import RuleBasedStateMachine, precondition, rule

        driver = self

        class MyReaderStream(RuleBasedStateMachine):
            def __init__(self) -> None:
                super().__init__()
                self.stream = driver.load()

            @rule()
            def update_offsets(self) -> None:
                self.stream.update_offsets()

            @precondition(lambda self: len(self.stream.offsets) > 1)
            @rule(skip=st.integers(min_value=0), max_count=st.integers(min_value=1))
            def read(self, skip, max_count) -> None:
                self.stream.read(skip, max_count)

            @precondition(lambda self: len(self.stream.offsets) > 1)
            @rule(max_count=st.integers(min_value=1))
            def commit(self, max_count) -> None:
                self.stream.commit(max_count)

        return MyReaderStream


@dataclass
class Offset:
    offset: dict
    # Did we read the range between this and the next offset?
    is_read: bool = False


@typechecked  # Enable runtime type checking
class StreamReaderDriver:
    def __init__(self, data_source: DataSource, schema: StructType):
        self.data_source = data_source
        self.schema = schema
        self.reader = _streamReader(data_source, schema)
        self.pickled_reader = pickleSer.dumps(self.reader)
        self.offsets: List[Offset] = []
        self.committed = 0

    def _to_arrow_batches(
        self, result: Union[Iterator[Tuple], Iterator[pa.RecordBatch]]
    ) -> Iterable[pa.RecordBatch]:
        return records_to_arrow_batches(
            output_iter=result,
            max_arrow_batch_size=1000,
            return_type=self.schema,
            data_source=self.data_source,
        )

    def _read(self, partition: InputPartition) -> Iterable[pa.RecordBatch]:
        reader = cast(DataSourceStreamReader, pickleSer.loads(self.pickled_reader))
        return self._to_arrow_batches(reader.read(partition))

    def _make_table(self, batches: Iterable[pa.RecordBatch]) -> pa.Table:
        return pa.Table.from_batches(batches, schema=to_arrow_schema(self.schema))

    def update_offsets(self) -> Optional[dict]:
        """
        Fetch the latest offset and append it to the offsets list if it changed.

        Returns
        -------
        Optional[dict]
            The latest offset if it changed, otherwise `None`.
        """
        latest = self.reader.latestOffset()
        if not self.offsets:
            self.offsets.append(Offset(self.reader.initialOffset()))
        if latest != self.offsets[-1].offset:
            self.offsets.append(Offset(latest))
            return latest
        return None

    def read(self, skip=0, max_count=sys.maxsize) -> pa.Table:
        """
        Read batches of data from the stream.

        Parameters
        ----------
        skip : int
            The number of offsets to skip ahead of the last committed offset.
        max_count : int
            The maximum number of offsets to read.
        """
        if skip < 0:
            raise ValueError("cannot read committed offsets")
        begin = self.committed + skip
        if max_count <= 0 or len(self.offsets) - 1 <= self.committed + skip:
            return self._make_table([])
        end = min(begin + max_count, len(self.offsets) - 1)
        assert begin < end
        begin_offset = self.offsets[begin].offset
        end_offset = self.offsets[end].offset
        if isinstance(self.reader, _SimpleStreamReaderWrapper):
            cached = self.reader.getCache(begin_offset, end_offset)
            if cached is not None:
                return self._make_table(self._to_arrow_batches(cached))
        partitions = self.reader.partitions(begin_offset, end_offset)
        result = self._make_table(
            batch for part in partitions for batch in self._read(part)
        )
        for i in range(begin, end):
            self.offsets[i].is_read = True
        return result

    def commit(self, max_count=sys.maxsize) -> int:
        """
        Commit the read offsets.

        Parameters
        ----------
        max_count : int
            The maximum number of offsets to commit.
        """
        committed = self.committed
        for i in range(max_count):
            if not self.offsets[committed].is_read:
                break
            committed += 1
        num_committed = committed - self.committed
        if num_committed > 0:
            self.reader.commit(self.offsets[committed].offset)
        self.committed = committed
        return num_committed

    def __iter__(self) -> "StreamReaderIterator":
        return StreamReaderIterator(self)


@typechecked  # Enable runtime type checking
class StreamReaderIterator:
    def __init__(self, driver: StreamReaderDriver):
        self.driver = driver

    def next_nonblocking(self) -> Optional[pa.Table]:
        """
        Get the next batch if the latest offset changed.
        """
        if self.driver.update_offsets():
            self.driver.commit()
            return self.driver.read()
        return None

    def next(self) -> pa.Table:
        """
        Wait for the next batch to be available, then return it.
        """
        result = self.next_nonblocking()
        while result is None:
            result = self.next_nonblocking()
        return result

    def __next__(self) -> pa.Table:
        return self.next()

    def __iter__(self) -> "StreamReaderIterator":
        return self
