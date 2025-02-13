# PySpark Data Source Driver

This project provides a lightweight driver to test Python Data Sources for Apache Spark, without the need to run a full Spark cluster.

## Installation

## Usage

`ReadDriver`'s API is similar to `spark.read()`, but `load()` returns a `pyarrow.Table` instead of a Spark DataFrame.

```python
from pyspark_data_source_driver import ReadDriver
from pyspark_huggingface.huggingface import HuggingFaceDatasets

table = ReadDriver(HuggingFaceDatasets).load("rotten_tomatoes")
assert table.num_rows == 8530
```

## Development

```bash
poetry install
poetry run pytest
```
