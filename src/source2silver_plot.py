# Databricks notebook source
"""
CLASS: Source2SilverPlot
DESCRIPTION: This class is used to download film plots from OMDB API.
AUTHORS: Jaime Jimenez, Marc Cañellas, Diego Bermejo & Jaume Adrover
DATE: 2025-07-24
"""
from pyspark.sql import functions as F
from pyspark.sql import Row
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# COMMAND ----------

# Group films by average rating and get count
ratings_df = spark.table("silver.title_ratings")
ratings_df.groupBy(F.floor("averageRating").alias("rating_floor")).count().orderBy("rating_floor").show()

# COMMAND ----------


# 1. General config
total_movies = 300000

# Group proportions (should be a total of 1.0)
group_proportions = {
    "low": 0.2,
    "medium_low": 0.3,
    "medium_high": 0.3,
    "high": 0.2
}

# 2. Read and filter silver table with films with more than 1000 votes
silver_ratings = spark.table("silver.title_ratings")
filtered = silver_ratings.filter(F.col("numVotes") >= 1000)

# Rating group classification
filtered = filtered.withColumn(
    "rating_group",
    F.when(F.col("averageRating") <= 4, "low")
     .when((F.col("averageRating") > 4) & (F.col("averageRating") <= 6), "medium_low")
     .when((F.col("averageRating") > 6) & (F.col("averageRating") <= 8), "medium_high")
     .otherwise("high")
)

# 3. Calculate every rating group sizes
group_sizes = {
    group: int(total_movies * proportion)
    for group, proportion in group_proportions.items()
}

# 4. Exact stratified sampling
samples = []
for group, size in group_sizes.items():
    group_df = (
        filtered.filter(F.col("rating_group") == group)
                .orderBy(F.rand(seed=42))
                .limit(size)
    )
    samples.append(group_df)

# 5. Join results
final_sample = samples[0]
for df in samples[1:]:
    final_sample = final_sample.union(df)

# 6. Extract film IDs
movie_ids = [row['tconst'] for row in final_sample.select("tconst").collect()]

print(f"🎬 Selected films: {len(movie_ids)}")

# COMMAND ----------

# Get first 5 movie IDs
print(movie_ids[:5])

# COMMAND ----------

# OMDB API key
dbutils.widgets.text("OMDB_API_KEY", "")
OMDB_API_KEY = dbutils.widgets.get("OMDB_API_KEY")

"""Fetches the plot for a given IMDb ID using the OMDB API."""
def fetch_plot(imdb_id):
    # Build OMDB API URL given a IMDb ID
    url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}&plot=short"
    try:
        # Response from OMDB API
        response = requests.get(url, timeout=10)
        data = response.json()
        # Check if the response is successful
        if data.get("Response") == "True":
            return {"tconst": imdb_id, "title": data.get("Title"), "year": data.get("Year"), "plot": data.get("Plot")}
        else:
            # Handle errors from OMDB API
            return {"tconst": imdb_id, "title": None, "year": None, "plot": None}
    except Exception as e:
        print(f"Error WITH {imdb_id}: {e}")
        return {"tconst": imdb_id, "title": None, "year": None, "plot": None}

# Initialising counters
total = len(movie_ids)
completadas = 0
con_plot = 0
results = []

print(f"Requesting plots from OMDb for a total of {total} movies...\n")

# Parallel plot fetching
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch_plot, imdb_id) for imdb_id in movie_ids]
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        completadas += 1
        if result["plot"] not in (None, "N/A"):
            con_plot += 1
        
        # Show progress every 100 movies
        if completadas % 100 == 0 or completadas == total:
            print(f"Completed: {completadas}/{total} - With plot: {con_plot} ({(con_plot / completadas)*100:.2f}%)")

print(f"✅ Plot valid films: {con_plot}/{total} ({(con_plot/total)*100:.2f}%)")

# Convert to Spark DataFrame
rows = [Row(**res) for res in results if res["plot"] not in (None, "N/A")]
plots_df = spark.createDataFrame(rows)

# COMMAND ----------

# Cast year to int
plots_df = plots_df.withColumn("year", F.col("year").cast("int"))

# Overwrite Delta table
plots_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("silver.title_plots")

# Display sample
display(plots_df.limit(10))