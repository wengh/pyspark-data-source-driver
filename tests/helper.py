import pytest


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()
