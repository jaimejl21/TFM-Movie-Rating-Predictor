# tests/conftest.py
import shutil
import tempfile
import pytest
from pyspark.sql import SparkSession

# Fixture to provide a SparkSession for tests (session scope)
@pytest.fixture(scope="session")
def spark():
    try:
        spark = (
            SparkSession.builder
            .master("local[1]")
            .appName("tfm-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
        yield spark
    except Exception as e:
        print("Error creating SparkSession:", e)
        raise
    finally:
        try:
            spark.stop()
        except Exception:
            pass

# Fixture to provide a temporary directory for file I/O during tests
@pytest.fixture
def tmpdir_path():
    p = tempfile.mkdtemp(prefix="tfm_tests_")
    try:
        yield p
    finally:
        shutil.rmtree(p, ignore_errors=True)
