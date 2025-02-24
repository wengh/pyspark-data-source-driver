# PySpark Data Source Driver

This project provides a lightweight driver to test Python Data Sources for Apache Spark, without the need to run a full Spark cluster.

## Installation

## Usage

### Batch Read

`ReadDriver`'s API is similar to `spark.read`, but `load()` returns a `pyarrow.Table` instead of a Spark DataFrame.

```python
from pyspark_data_source_driver import ReadDriver
from pyspark_huggingface.huggingface import HuggingFaceDatasets

table = ReadDriver(HuggingFaceDatasets).load("rotten_tomatoes")
assert table.num_rows == 8530
```

### Streaming Read

`StreamReadDriver.load()` returns a `StreamReaderDriver` iterable which reads one microbatch each iteration, and also allows manually testing different streaming read sequences.

```python
import time
from pyspark_data_source_driver import StreamReadDriver
from pyspark_datasources.weather import WeatherDataSource

options = {
   "locations": "[(37.7749, -122.4194), (40.7128, -74.0060)]",
   "apikey": "your_api_key_here",
}

stream = StreamReadDriver(WeatherDataSource).options(**options).load()

# Each iteration takes 1 minute
for table in stream:
    print(table)
    time.sleep(60)  # sleep to avoid hitting the API rate limit
```

## Development

```bash
poetry install
poetry run pytest
```
