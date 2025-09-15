# Databricks notebook source
"""
CLASS: Silver2GoldPredictionInput
DESCRIPTION: This file is used as a way to transfer silver data to the gold layer.
AUTHORS: Jaime Jimenez, Marc Cañellas, Diego Bermejo & Jaume Adrover
DATE: 2025-07-26
"""
from pyspark.sql import functions as F
from azure.storage.blob import BlobServiceClient, BlobClient
import json
import numpy as np

# COMMAND ----------

# Load tables
plots = spark.table("silver.title_plots")
basics = spark.table("silver.title_basics")
ratings = spark.table("silver.title_ratings")
crew = spark.table("silver.title_crew")
principals = spark.table("silver.title_principals")
names = spark.table("silver.name_basics")

# --- Obtain actors (actor, actress) ---
actors = principals.filter(F.col("category").isin("actor", "actress")) \
    .join(names.select("nconst", "primaryName"), "nconst", "left") \
    .groupBy("tconst") \
    .agg(F.collect_list("primaryName").alias("actors"))

# --- Obtain directors ---
directors = crew.select("tconst", F.explode(F.split("directors", ",")).alias("nconst")) \
    .join(names.select("nconst", "primaryName"), "nconst", "left") \
    .groupBy("tconst") \
    .agg(F.collect_list("primaryName").alias("directors"))

# --- Obtain scriptwriters ---
writers = crew.select("tconst", F.explode(F.split("writers", ",")).alias("nconst")) \
    .filter(F.col("nconst") != "\\N") \
    .join(names.select("nconst", "primaryName"), "nconst", "left") \
    .groupBy("tconst") \
    .agg(F.collect_list("primaryName").alias("writers"))

# --- Join all ---
movies = plots.alias("p") \
    .join(basics.alias("b"), "tconst") \
    .join(ratings.alias("r"), "tconst", "left") \
    .join(actors, "tconst", "left") \
    .join(directors, "tconst", "left") \
    .join(writers, "tconst", "left")

# --- Relevant column selection ---
gold_df = movies.select(
    "tconst",
    F.col("primaryTitle").alias("title"),
    F.col("p.plot"),
    F.col("startYear").alias("year"),
    "genres",
    "averageRating",
    "directors",
    "writers",
    "actors"
)


# COMMAND ----------

# Gold schema
gold_schema = "hive_metastore.gold"

# --- Save as Delta table in gold scheme ---
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {gold_schema}")

gold_df.write.format("delta").mode("overwrite").saveAsTable("gold.prediction_input")


# Verification displayment
display(gold_df.limit(10))

# COMMAND ----------

# Storage account configuration    
dbutils.widgets.text("storage_account_name", "")
dbutils.widgets.text("storage_account_key", "")
dbutils.widgets.text("container_name", "lakehouse")
dbutils.widgets.text("gold_dir", "gold")  # por si quieres parametrizar la ruta

storage_account_name = dbutils.widgets.get("storage_account_name")
storage_account_key  = dbutils.widgets.get("storage_account_key")
container_name       = dbutils.widgets.get("container_name")
gold_dir             = dbutils.widgets.get("gold_dir")

# Blob Service Client Creation
blob_service_client = BlobServiceClient(
    f"https://{storage_account_name}.blob.core.windows.net",
    credential=storage_account_key
)

# COMMAND ----------

# Convert data to pandas.DataFrame
data = gold_df.toPandas().to_dict(orient="records")

# Convert data to JSON
for record in data:
    for key, value in record.items():
        if isinstance(value, np.ndarray):
            record[key] = value.tolist()

# Transform objects to this format
# transformed_data = [
#     {
#         "input": f"Predict rating of the following movie:\n\n{json.dumps(record, ensure_ascii=False)}\n",
#         "output": {
#             "rating": record["averageRating"]
#         }
#     }
#     for record in data
# ]

# Transform objects to plain format
transformed_data = [
    {
        "tconst": record.get("tconst"),
        "title": record.get("title"),
        "plot": record.get("plot"),
        "year": record.get("year"),
        "genres": record.get("genres"),
        "averageRating": record.get("averageRating"),
        "directors": record.get("directors"),
        "writers": record.get("writers"),
        "actors": record.get("actors"),
    }
    for record in data
]

# Convert transformed list to JSON
json_str = json.dumps(transformed_data, ensure_ascii=False, indent=2)

# Print final JSON
print(json_str)

# Blob Storage Client Configuration
blob_client = BlobClient(
    f"https://{storage_account_name}.blob.core.windows.net",
    container_name,
    f"{gold_dir}/prediction_input/prediction_input.json",
    credential=storage_account_key
)

# Upload as text json file
blob_client.upload_blob(json_str, overwrite=True)