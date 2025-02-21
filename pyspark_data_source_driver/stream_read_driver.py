import logging
from typing import Dict, Iterable, List, Optional, Sequence, Type, TypeVar, Union, cast

import pyarrow as pa
from pyspark.sql.datasource import (
    CaseInsensitiveDict,
    DataSource,
    DataSourceStreamReader,
    InputPartition,
)
from pyspark.sql.datasource_internal import _streamReader
from pyspark.sql.types import StructType
from pyspark.sql.worker.plan_data_source_read import records_to_arrow_batches
from pyspark.worker_util import pickleSer
from typeguard import typechecked

from pyspark_data_source_driver.read_driver import ReadDriverBase

logger = logging.getLogger(__name__)

_sentinel = object()


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
    ...     def partitions(self, start: dict, end: dict) -> Sequence[InputPartition]:
    ...         return [InputPartition(i) for i in range(start["offset"], end["offset"])]
    ...
    ...     def read(self, partition):
    ...         yield (partition.value,)
    ...

    Read data from the data source.

    >>> from pyspark_data_source_driver import StreamReadDriver
    >>> stream = StreamReadDriver(CounterDataSource).load()
    >>> for i in range(2):
    ...     print(next(stream).to_pydict())
    {'id': [0, 1, 2]}
    {'id': [3, 4, 5]}
    """

    def load(self, path: Optional[str] = None) -> "ReaderStream":
        data_source, schema = super().load(path)
        return ReaderStream(data_source, schema)


class ReaderStream:
    def __init__(self, data_source: DataSource, schema: StructType):
        self.data_source = data_source
        self.schema = schema
        self.reader = _streamReader(data_source, schema)
        self.pickled_reader = pickleSer.dumps(self.reader)

    def _read(self, partition: InputPartition) -> Iterable[pa.RecordBatch]:
        reader = cast(DataSourceStreamReader, pickleSer.loads(self.pickled_reader))
        result = reader.read(partition)
        return records_to_arrow_batches(
            output_iter=result,
            max_arrow_batch_size=1000,
            return_type=self.schema,
            data_source=self.data_source,
        )

    def next_nonblocking(self) -> Optional[pa.Table]:
        """
        Get the next batch if the latest offset changed.
        """
        latest = self.reader.latestOffset()
        if not hasattr(self, "start"):
            # initialOffset is called after the first call to latestOffset
            self.start = self.reader.initialOffset()
        elif self.start != latest:
            self.reader.commit(self.start)

        if self.start == latest:
            return None
        partitions = self.reader.partitions(self.start, latest)
        result = pa.Table.from_batches(
            batch for part in partitions for batch in self._read(part)
        )
        self.start = latest
        return result

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

    def __iter__(self) -> "ReaderStream":
        return self
