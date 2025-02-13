import logging
from typing import Dict, List, Optional, Sequence, Type, TypeVar, Union, cast

import pyarrow as pa
from pyspark.sql.datasource import (
    CaseInsensitiveDict,
    DataSource,
    DataSourceReader,
    InputPartition,
)
from pyspark.sql.types import StructType
from pyspark.sql.worker.plan_data_source_read import records_to_arrow_batches
from pyspark.worker_util import pickleSer
from typeguard import typechecked


T = TypeVar("T")

logger = logging.getLogger(__name__)


@typechecked  # Enable runtime type checking
class ReadDriver:
    """
    A lightweight driver to read data from a data source, for testing and prototyping.

    It tries its best to enforce the Python Data Source API requirements, by:
    - Pickling and unpickling objects to simulate the behavior of Spark's serialization.
    - Enforce type hints using `typeguard`.

    Examples
    --------

    Define a simple data source and its reader:
    >>> from pyspark.sql.types import IntegerType, StructField, StructType
    >>> class CounterDataSource(DataSource):
    ...     def schema(self):
    ...         return StructType(
    ...             [StructField("id", IntegerType())]
    ...         )
    ...
    ...     def reader(self, schema: StructType):
    ...         return CounterDataSourceReader(self.options)
    ...
    >>> class CounterDataSourceReader(DataSourceReader):
    ...     def __init__(self, options):
    ...         self.num_rows = int(options.get("num_rows", 3))
    ...
    ...     def partitions(self):
    ...         return [InputPartition(i) for i in range(self.num_rows)]
    ...
    ...     def read(self, partition):
    ...         yield (partition.value,)
    ...

    Read data from the data source.

    >>> from pyspark_data_source_driver.read_driver import ReadDriver
    >>> table = ReadDriver(CounterDataSource).options(num_rows="1").load()
    >>> table.to_pydict()
    {'id': [0]}

    Specify the schema.

    >>> from pyspark.sql import SparkSession
    >>> spark = SparkSession.builder.getOrCreate()
    >>> table = ReadDriver(CounterDataSource).schema("index string").load()
    >>> table.to_pydict()
    {'index': ['0', '1', '2']}
    """

    def __init__(self, data_source: Type[DataSource]):
        self.data_source_class = data_source
        self._options: Dict[str, str] = CaseInsensitiveDict()  # type: ignore
        self._schema: Optional[StructType] = None

    def option(self, key: str, value: str):
        """
        Adds an input option for the underlying data source.
        """
        self._options[key] = value
        return self

    def options(self, **options: str):
        """
        Adds input options for the underlying data source.
        """
        self._options.update(options)
        return self

    def schema(self, schema: Union[StructType, str]):
        """
        Specifies the input schema.
        If not provided, the driver will call DataSource.schema() to infer the schema.

        Parameters
        ----------
        schema : :class:`pyspark.sql.types.StructType` or str
            a :class:`pyspark.sql.types.StructType` object or a DDL-formatted string
            (For example ``col0 INT, col1 DOUBLE``).

        Note:
        Requires a SparkSession to convert DDL-formatted string to StructType.
        """
        self._schema = self._cast_schema(schema)
        return self

    def load(self, path: Optional[str] = None):
        """
        Loads data from a data source and returns it as a :class:`pyarrow.Table`.
        """
        if path is not None:
            self._options["path"] = path
        return self._read()

    @staticmethod
    def _cast_schema(schema: Union[StructType, str]) -> StructType:
        if isinstance(schema, str):
            return cast(StructType, StructType.fromDDL(schema))
        return schema

    def _read(self) -> pa.Table:
        data_source = self.data_source_class(self._options)
        data_source = self._pickle_unpickle(data_source)
        schema = self._schema or self._cast_schema(data_source.schema())
        reader = data_source.reader(schema)
        pickled_reader = pickleSer.dumps(reader)
        # Mutations made by partitions() are not saved
        partitions = self._get_partitions(reader)

        batches: List[pa.RecordBatch] = []
        for partition in partitions:
            reader = cast(DataSourceReader, pickleSer.loads(pickled_reader))
            result = reader.read(partition)  # type: ignore
            batches.extend(
                records_to_arrow_batches(
                    output_iter=result,
                    max_arrow_batch_size=1000,
                    return_type=schema,
                    data_source=data_source,
                )
            )

        return pa.Table.from_batches(batches)

    @classmethod
    def _get_partitions(
        cls, reader: DataSourceReader
    ) -> Sequence[Optional[InputPartition]]:
        try:
            partitions = cast(Sequence[InputPartition], reader.partitions())
            return cls._pickle_unpickle(partitions)
        except NotImplementedError:
            logger.info(
                f"reader.partitions() is not implemented. "
                "Partitioning is required to benefit from distributed scan.",
                exc_info=True,
            )
            return [None]

    @staticmethod
    def _pickle_unpickle(obj: T) -> T:
        return pickleSer.loads(pickleSer.dumps(obj))
