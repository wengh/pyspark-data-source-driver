from dataclasses import dataclass
from typing import Sequence

import pytest
from pyspark.sql.datasource import DataSource, DataSourceReader, InputPartition
from pyspark.sql.types import IntegerType, StringType, StructField, StructType
from typeguard import TypeCheckError

from pyspark_data_source_driver import ReadDriver
from tests.helper import spark


class SimpleDataSource(DataSource):
    def schema(self):
        return StructType(
            [StructField("name", StringType()), StructField("age", IntegerType())]
        )

    def reader(self, schema: StructType):
        return SimpleDataSourceReader()


class SimpleDataSourceReader(DataSourceReader):
    def read(self, partition):
        assert partition is None
        yield ("Alice", 20)
        yield ("Bob", 30)


def test_read():
    table = ReadDriver(SimpleDataSource).load()
    assert table.to_pydict() == {"name": ["Alice", "Bob"], "age": [20, 30]}


def test_supply_schema(spark):
    class MyDataSource(SimpleDataSource):
        def reader(self, schema: StructType):
            assert schema == StructType(
                [StructField("a", StringType()), StructField("b", StringType())]
            )
            return SimpleDataSourceReader()

    table = ReadDriver(MyDataSource).schema("a string, b string").load()
    assert table.to_pydict() == {"a": ["Alice", "Bob"], "b": ["20", "30"]}


def test_options():
    class MyDataSource(SimpleDataSource):
        def reader(self, schema: StructType):
            assert self.options["OPT"] == "foo"
            assert self.options["path"] == "bar"
            return SimpleDataSourceReader()

    ReadDriver(MyDataSource).options(opt="foo").load("bar")


def test_bad_schema(spark):
    with pytest.raises(TypeCheckError):
        ReadDriver(SimpleDataSource).schema("int").load()


def test_schema_mismatch():
    with pytest.raises(RuntimeError):
        ReadDriver(SimpleDataSource).schema(
            StructType([StructField("x", IntegerType())])
        ).load()


def test_not_pickleable():
    class MyDataSource(SimpleDataSource):
        def __init__(self, options):
            super().__init__(options)
            self.generator = (i for i in range(10))

    with pytest.raises(Exception) as excinfo:
        ReadDriver(MyDataSource).load()
    assert "pickle" in str(excinfo.value)


class PartitionDataSource(DataSource):
    def schema(self):
        return StructType([StructField("x", IntegerType())])

    def reader(self, schema: StructType):
        return PartitionDataSourceReader()


@dataclass
class Partition(InputPartition):
    num: int


class PartitionDataSourceReader(DataSourceReader):
    def __init__(self):
        self.used = False

    def partitions(self) -> Sequence[InputPartition]:
        assert not self.used
        self.used = True
        return [Partition(1), Partition(2)]

    def read(self, partition):
        assert not self.used
        self.used = True
        yield (partition.num,)


def test_partitions():
    table = ReadDriver(PartitionDataSource).load()
    assert table.to_pydict() == {"x": [1, 2]}
