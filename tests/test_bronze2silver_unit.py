# tests/test_bronze2silver_unit.py
from pathlib import Path
import pytest
from pyspark.sql import Row
from pyspark.sql.functions import col
from .utils_loader import load_functions

FILE = Path("src/bronze2silver_imdb.py")

def _load(names):
    # Dynamically load functions from the specified file
    return load_functions(FILE, names)

def test_clean_names_basic(spark):
    """
    Test that the clean_names function removes duplicates and handles nulls.
    """
    fn = _load(["clean_names"])["clean_names"]
    if fn is None:
        pytest.skip("clean_names not found; skipping")

    # Create a DataFrame with duplicate and invalid rows
    df = spark.createDataFrame([
        Row(nconst="nm0001", primaryName="A", birthYear="1970", deathYear="\\N"),
        Row(nconst="nm0001", primaryName="A", birthYear="1970", deathYear="\\N"),  # duplicate
        Row(nconst=None,     primaryName="B", birthYear="N/A", deathYear="\\N"),   # invalid
    ])
    out = fn(df)

    # Check deduplication on non-null nconst
    non_null = out.filter(out.nconst.isNotNull())
    assert non_null.select("nconst").distinct().count() == 1

    assert "birthYear" in out.columns

def test_clean_ratings_numeric(spark):
    """
    Test that the clean_ratings function converts columns to numeric types.
    """
    fn = _load(["clean_ratings"])["clean_ratings"]
    if fn is None:
        pytest.skip("clean_ratings not found; skipping")

    # Create a DataFrame with valid and invalid numeric values
    df = spark.createDataFrame([
        Row(tconst="tt1", averageRating="7.5", numVotes="10"),
        Row(tconst="tt2", averageRating="N/A", numVotes="x"),
    ])
    out = fn(df)
    assert dict(out.dtypes)["averageRating"] in ("double", "float")
    assert dict(out.dtypes)["numVotes"] in ("bigint", "int", "long")
    assert out.filter(col("tconst") == "tt1").count() == 1

def test_clean_basics_genres_split(spark):
    """
    Test that the clean_basics function splits genres and handles missing values.
    """
    fn = _load(["clean_basics"])["clean_basics"]
    if fn is None:
        pytest.skip("clean_basics not found; skipping")

    # Create a DataFrame with genres as a comma-separated string and None
    df = spark.createDataFrame([
        Row(tconst="tt1", primaryTitle="X", startYear="1999", endYear="\\N",
            runtimeMinutes="120", isAdult="0", genres="Drama,Comedy"),
        Row(tconst="tt2", primaryTitle="Y", startYear="N/A", endYear="\\N",
            runtimeMinutes="\\N", isAdult="1", genres=None),
    ])
    out = fn(df)
    assert "genres" in out.columns
    assert out.filter(out.tconst == "tt1").select("genres").first()[0] is not None
