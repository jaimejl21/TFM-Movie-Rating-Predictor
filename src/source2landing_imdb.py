# Databricks notebook source
"""
CLASS: Source2Landing
DESCRIPTION: This class is used to download IMDb datasets and replace them in the landing folder.
AUTHORS: Jaime Jimenez, Marc Cañellas, Diego Bermejo & Jaume Adrover
DATE: 2025-07-15
"""
import requests
from datetime import date
import gzip
import shutil
from azure.storage.blob import BlobServiceClient
from io import BytesIO

# COMMAND ----------

# Storage Account Parameters
dbutils.widgets.text("storage_account_name", "")
dbutils.widgets.text("storage_account_key", "")
dbutils.widgets.text("container_name", "lakehouse")

storage_account_name = dbutils.widgets.get("storage_account_name")
storage_account_key  = dbutils.widgets.get("storage_account_key")
container_name       = dbutils.widgets.get("container_name")

today = date.today().isoformat()
landing_dir = f"landing/{today}/"

# BlobServiceClient creation
blob_service_client = BlobServiceClient(
    f"https://{storage_account_name}.blob.core.windows.net",
    credential=storage_account_key
)

# COMMAND ----------

# IMDb Datasets Path
datasets = {
    "title.akas.tsv.gz": "https://datasets.imdbws.com/title.akas.tsv.gz",
    "title.basics.tsv.gz": "https://datasets.imdbws.com/title.basics.tsv.gz",
    "title.crew.tsv.gz": "https://datasets.imdbws.com/title.crew.tsv.gz",
    "title.episode.tsv.gz": "https://datasets.imdbws.com/title.episode.tsv.gz",
    "title.principals.tsv.gz": "https://datasets.imdbws.com/title.principals.tsv.gz",
    "title.ratings.tsv.gz": "https://datasets.imdbws.com/title.ratings.tsv.gz",
    "name.basics.tsv.gz": "https://datasets.imdbws.com/name.basics.tsv.gz"
}

# COMMAND ----------

# Download, unzip & replace in landing folder
for filename, url in datasets.items():
    print(f"Downloading, unzipping and replacing {filename} in {landing_dir}...")
    # Download file via HTTP
    response = requests.get(url)
    if response.status_code == 200:
        # Create a BytesIO object to hold the compressed data
        compressed_data = BytesIO(response.content)
        decompressed_data = BytesIO()
        
        # Unzip file
        with gzip.open(compressed_data, 'rb') as f_in:
            shutil.copyfileobj(f_in, decompressed_data)
        
        # Locate pointer to the beginning of the file
        decompressed_data.seek(0)

        # Get blob path
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=f"{landing_dir}{filename.replace('.gz', '')}")
        # Upload file
        blob_client.upload_blob(decompressed_data, overwrite=True)
        
        print(f"Download process, unzipped and {filename} replacement completed in {landing_dir}")
    else:
        print(f"Error while downloading {filename}: Status {response.status_code}")

print("✅ Process complete.")