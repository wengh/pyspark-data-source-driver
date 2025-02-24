from pyspark.sql.datasource import DataSource, DataSourceStreamReader, InputPartition
from pyspark.sql.types import IntegerType, StructField, StructType


class CounterDataSource(DataSource):
    def schema(self):
        return StructType([StructField("id", IntegerType())])

    def streamReader(self, schema: StructType):
        return CounterDataSourceStreamReader(self.options)


class CounterDataSourceStreamReader(DataSourceStreamReader):
    def __init__(self, options):
        self.batch_size = int(options.get("batch_size", 3))
        self.current = 0

    def initialOffset(self):
        return {"offset": 0}

    def latestOffset(self):
        self.current += self.batch_size
        return {"offset": self.current}

    def partitions(self, start: dict, end: dict):
        return [InputPartition(i) for i in range(start["offset"], end["offset"])]

    def read(self, partition):
        yield (partition.value,)
