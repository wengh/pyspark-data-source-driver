from typing import Sequence

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql.datasource import DataSource, DataSourceStreamReader, InputPartition
from pyspark.sql.types import IntegerType, StructField, StructType

from pyspark_data_source_driver import StreamReadDriver
from tests.helper import spark


class CounterFinished(Exception):
    def __init__(self, sequence):
        self.sequence = sequence


class CounterDataSource(DataSource):
    def schema(self):
        return StructType([StructField("id", IntegerType())])

    def streamReader(self, schema: StructType):
        return CounterDataSourceStreamReader(self.options)


class CounterDataSourceStreamReader(DataSourceStreamReader):
    def __init__(self, options):
        self.options = options
        self.current = 0
        self.sequence = []

    def log(self, info):
        self.sequence.append(info)

    def initialOffset(self):
        return {"offset": 0}

    def latestOffset(self):
        if not hasattr(self, "sizes"):
            self.sizes = iter(eval(self.options["sizes"]))
        try:
            self.current += next(self.sizes)
        except StopIteration:
            raise CounterFinished(self.sequence)
        return {"offset": self.current}

    def partitions(self, start: dict, end: dict) -> Sequence[InputPartition]:
        return [InputPartition(i) for i in range(start["offset"], end["offset"])]

    def read(self, partition):
        yield (partition.value,)


def test_read():
    stream = StreamReadDriver(CounterDataSource).options(sizes="[3, 3]").load()
    assert next(stream).to_pydict() == {"id": [0, 1, 2]}
    assert next(stream).to_pydict() == {"id": [3, 4, 5]}


def test_commit():
    class MyDataSource(CounterDataSource):
        def streamReader(self, schema: StructType):
            return MyDataSourceStreamReader(self.options)

    class MyDataSourceStreamReader(CounterDataSourceStreamReader):
        def commit(self, end: dict):
            self.log(end["offset"])
            self.current += 1

    stream = StreamReadDriver(MyDataSource).options(sizes="[2, 2, 2]").load()
    assert next(stream).to_pydict() == {"id": [0, 1]}
    assert next(stream).to_pydict() == {"id": [2, 3]}
    assert next(stream).to_pydict() == {"id": [4, 5, 6]}
    with pytest.raises(CounterFinished) as excinfo:
        next(stream)
    assert excinfo.value.sequence == [2, 4]


class TestCallSequence:
    class MyDataSource(CounterDataSource):
        def streamReader(self, schema: StructType):
            return TestCallSequence.MyDataSourceStreamReader(self.options)

    class MyDataSourceStreamReader(CounterDataSourceStreamReader):
        def initialOffset(self):
            self.log("initialOffset")
            return super().initialOffset()

        def latestOffset(self):
            self.log("latestOffset")
            return super().latestOffset()

        def partitions(self, start, end):
            self.log("partitions")
            return super().partitions(start, end)

        def commit(self, end):
            self.log("commit")

        def read(self, partition):
            assert self.sequence == []
            assert not hasattr(self, "gaps"), self.gaps
            self.gaps = "dummy"  # should not be visible to future reads
            return super().read(partition)

    options = {
        "sizes": "[3, 0, 3]",
    }

    expected = [
        "latestOffset",
        "initialOffset",
        "partitions",
        "latestOffset",
        "latestOffset",
        "commit",
        "partitions",
        "latestOffset",
    ]

    def test_call_sequence_driver(self):
        stream = StreamReadDriver(self.MyDataSource).options(**self.options).load()
        assert next(stream).to_pydict() == {"id": [0, 1, 2]}
        assert next(stream).to_pydict() == {"id": [3, 4, 5]}
        with pytest.raises(CounterFinished) as excinfo:
            next(stream)
        assert excinfo.value.sequence == self.expected

    def test_call_sequence_spark(self, spark: SparkSession):
        import re

        spark.dataSource.register(self.MyDataSource)
        pattern = re.escape(repr(self.expected))
        with pytest.raises(Exception, match=pattern):
            (
                spark.readStream.format("MyDataSource")
                .options(**self.options)
                .load()
                .writeStream.format("memory")
                .queryName("test")
                .start()
                .awaitTermination()
            )
        result = spark.sql("select * from test")
        assert result.collect() == [Row(id=i) for i in range(6)]


def test_init_not_pickleable():
    class MyDataSource(CounterDataSource):
        def streamReader(self, schema: StructType):
            return MyDataSourceStreamReader(self.options)

    class MyDataSourceStreamReader(CounterDataSourceStreamReader):
        def __init__(self, options):
            super().__init__(options)
            self.generator = (i for i in range(10))

    with pytest.raises(Exception) as excinfo:
        StreamReadDriver(MyDataSource).load()
    assert "pickle" in str(excinfo.value)


def test_offset_no_change():
    class MyDataSource(CounterDataSource):
        def streamReader(self, schema: StructType):
            return MyDataSourceStreamReader(self.options)

    class MyDataSourceStreamReader(CounterDataSourceStreamReader):
        def latestOffset(self):
            self.current += 1
            return {"offset": self.current // 2}

    driver = StreamReadDriver(MyDataSource).options(sizes="[0, 1, 0, 1]")
    stream = driver.load()
    assert next(stream).to_pydict() == {"id": [0]}
    assert next(stream).to_pydict() == {"id": [1]}

    stream = driver.load()
    assert stream.next_nonblocking() is None
    assert stream.next_nonblocking().to_pydict() == {"id": [0]}
    assert stream.next_nonblocking() is None
    assert stream.next_nonblocking().to_pydict() == {"id": [1]}
