# Databricks notebook source
"""
CLASS: Landing2Bronze
DESCRIPTION: This file is used to transfer landing data to bronze in IMDB
AUTHORS: Jaime Jimenez, Marc Cañellas, Diego Bermejo & Jaume Adrover
DATE: 2025-07-28
"""
from azure.storage.blob import BlobServiceClient
from databricks.sdk.runtime import *

import pyspark.sql.functions as F
from delta.tables import DeltaTable

# COMMAND ----------

# Storage account configuration
dbutils.widgets.text("storage_account_name", "")
dbutils.widgets.text("storage_account_key", "")
dbutils.widgets.text("container_name", "lakehouse")

storage_account_name = dbutils.widgets.get("storage_account_name")
storage_account_key  = dbutils.widgets.get("storage_account_key")
container_name       = dbutils.widgets.get("container_name")

# Spark configuration to mount the storage account
spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    storage_account_key
)

# Blob Service Client creation
blob_service_client = BlobServiceClient(
    f"https://{storage_account_name}.blob.core.windows.net",
    credential=storage_account_key
)
# Get the container client
container_client = blob_service_client.get_container_client(container_name)

# COMMAND ----------

# Obtain all blobs
blob_list = list(container_client.list_blobs())

# Detect files in the landing date folder
dates = sorted(
    {blob.name.split('/')[1] for blob in blob_list if blob.name.startswith('landing/')},
    reverse=True
)

if dates:
    latest_date = dates[0]

    print(f"Processing most recent blobs in landing: {latest_date}")
    filtered_blobs = [blob for blob in blob_list if blob.name.startswith(f'landing/{latest_date}/') and blob.name.endswith('.tsv')]

    if not filtered_blobs:
        print("⚠️ Could not find TSV files in the most recent folder.")
    
    for blob in filtered_blobs:
        file_path = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/{blob.name}"
        table_name = blob.name.split('/')[-1].replace('.tsv', '').replace('.', '_')
        catalog = "hive_metastore"
        schema = f"{catalog}.bronze"
        table = f"{schema}.{table_name}"

        print(f"\n📁 Processing file: {blob.name}")
        print(f" -> Table name: {table_name}")

        # Leer archivo TSV
        df = spark.read.option("header", True).option("delimiter", "\t").csv(file_path)

        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")

        if df.head(1):
            # Añadir columna _processing_time
            df = df.withColumn("_processing_time", F.current_timestamp())

            # Obtener el nombre de la primera columna
            primary_key = df.columns[0]

            # Crear o actualizar la tabla Delta
            if not spark._jsparkSession.catalog().tableExists(table):
                print(f"🔧 Creating Delta table: {table}")
                df.write.format("delta").mode("overwrite").saveAsTable(table)
            else:
                print(f"🔁 Merging in table: {table}")
                delta_table = DeltaTable.forName(spark, table)

                delta_table.alias("target").merge(
                    df.alias("source"),
                    f"target.{primary_key} = source.{primary_key}"
                ).whenMatchedUpdateAll() \
                .whenNotMatchedInsertAll() \
                .execute()

            print(f" ✅ Saved as Delta table in bronze schema.")
            display(df)
        else:
            print(f" ⚠️ Empty file, can not be saved.")
else:
    print("❌ No blobs found in 'landing' folder.")