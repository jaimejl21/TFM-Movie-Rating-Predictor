# Databricks notebook source
"""
CLASS: Bronze2Silver
DESCRIPTION: This class is used process bronze tables and write them to silver tables
AUTHORS: Jaime Jimenez, Marc Cañellas, Diego Bermejo & Jaume Adrover
DATE: 2025-07-28
"""
from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

# Layer schemas
bronze_schema = "hive_metastore.bronze"
silver_schema = "hive_metastore.silver"

# Create silver schema if needed
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {silver_schema}")

# COMMAND ----------

# Cleaning per table

# title_basics
def clean_basics(df):
    df = df.withColumn("startYear", F.when(F.col("startYear") == "\\N", None).otherwise(F.col("startYear")).cast("int"))
    df = df.withColumn("endYear", F.when(F.col("endYear") == "\\N", None).otherwise(F.col("endYear")).cast("int"))
    df = df.withColumn("runtimeMinutes", F.when(F.col("runtimeMinutes") == "\\N", None).otherwise(F.col("runtimeMinutes")).cast("int"))
    df = df.withColumn("isAdult", F.col("isAdult").cast("boolean"))
    return df.filter(F.col("tconst").isNotNull())

# title_akas
def clean_akas(df):
    df = df.withColumn("isOriginalTitle", F.col("isOriginalTitle").cast("boolean"))
    return df.filter(F.col("title").isNotNull())

# title_crew
def clean_crew(df):
    return df.filter(F.col("tconst").isNotNull())

# title_episode
def clean_episode(df):
    df = df.withColumn("seasonNumber", F.when(F.col("seasonNumber") == "\\N", None).otherwise(F.col("seasonNumber")).cast("int"))
    df = df.withColumn("episodeNumber", F.when(F.col("episodeNumber") == "\\N", None).otherwise(F.col("episodeNumber")).cast("int"))
    return df.filter(F.col("parentTconst").isNotNull())

# title_principals
def clean_principals(df):
    df = df.withColumn("ordering", F.col("ordering").cast("int"))
    return df.filter(F.col("tconst").isNotNull() & F.col("nconst").isNotNull())

# title_ratings
def clean_ratings(df):
    df = df.withColumn("averageRating", F.col("averageRating").cast("double"))
    df = df.withColumn("numVotes", F.col("numVotes").cast("int"))
    return df.filter(F.col("averageRating").isNotNull() & F.col("numVotes").isNotNull())

# name_basics
def clean_names(df):
    df = df.withColumn("birthYear", F.when(F.col("birthYear") == "\\N", None).otherwise(F.col("birthYear")).cast("int"))
    df = df.withColumn("deathYear", F.when(F.col("deathYear") == "\\N", None).otherwise(F.col("deathYear")).cast("int"))
    return df.filter(F.col("primaryName").isNotNull())

# COMMAND ----------

# Parameterizable function to clean and write tables
def clean_and_write(df, table_name, cleaning_fn=None):
    if cleaning_fn:
        df = cleaning_fn(df)
    df.write.format("delta").mode("overwrite").saveAsTable(f"{silver_schema}.{table_name}")
    print(f"✅ {table_name} procesado.")

# COMMAND ----------

# All table processing
tables = [
    ("title_basics", clean_basics),
    ("title_akas", clean_akas),
    ("title_crew", clean_crew),
    ("title_episode", clean_episode),
    ("title_principals", clean_principals),
    ("title_ratings", clean_ratings),
    ("name_basics", clean_names),
]

# COMMAND ----------

for table, cleaning_fn in tables:
    try:
        bronze_table = f"{bronze_schema}.{table}"
        print(f"\n🚀 Processing {bronze_table}...")
        df = spark.read.table(bronze_table)
        clean_and_write(df, table, cleaning_fn)
    except Exception as e:
        print(f"❌ Error while processing {bronze_table}: {str(e)}")