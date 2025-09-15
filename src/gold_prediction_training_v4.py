# Databricks notebook source
# ===========================================
# 0 — Imports, parameters and utilities
# ===========================================

# ——— Imports
import os, time, itertools
import numpy as np
import pandas as pd
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient

# (optional) DBSQL connector 
try:
    import databricks.sql  # pip: databricks-sql-connector
except Exception:
    databricks = None

# Scikit-learn for featuring and model
from typing import Iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ——— Global parameters ———

# Names in Registry
MODEL_SKLEARN_NAME = "movie_rating_rf_sklearn"   # sklearn pipeline (process + RF)
API_MODEL_NAME     = "movie_rating_api"          # pyfunc wrapper (pandas-only for Serving)

# Gold table (prediction inputs) in the metastore
GOLD_CATALOG_DB_TABLE = "gold.prediction_input"  # adjust if your table is named differently

# Columns to be created/used for training the sklearn model
NUMERIC_COLS = [
    "actor_score_mean","actor_score_max",
    "director_score_mean","director_score_max",
    "writer_score_mean","writer_score_max",
    "numVotes_log1p","startYear_z",
]
TEXT_COL   = "plot"          # free text
GENRES_COL = "genres"        # list of genres (array<string> or CSV → normalize it)
LABEL_COL  = "averageRating" # target

# Featurization parameters (text and genres)
TFIDF_MAX_FEATURES = 2000
TFIDF_MIN_DF       = 3
SVD_COMPONENTS     = 64
RANDOM_STATE       = 42

# Artifacts needed for inference (used by the wrapper in Serving)
ART_DIR_LOCAL   = "/dbfs/tmp/movie_serving_assets"
PERSON_STATS_PQ = f"{ART_DIR_LOCAL}/person_stats.parquet"   # cols: nconst, role, score_role
NAME_INDEX_PQ   = f"{ART_DIR_LOCAL}/name_index.parquet"     # cols: name, nconst
GLOBAL_AVG_TXT  = f"{ART_DIR_LOCAL}/global_avg.txt"

# (optional) DBSQL credentials if fallback without Spark in notebook is needed
DBSQL_SERVER_HOSTNAME = os.getenv("DBSQL_SERVER_HOSTNAME", "")
DBSQL_HTTP_PATH       = os.getenv("DBSQL_HTTP_PATH", "")
DBSQL_TOKEN           = os.getenv("DBSQL_TOKEN", "")


# ——— Utilities ———

def ensure_artifacts_exist():
    """Check that the artifacts used by the wrapper exist before proceeding."""
    missing = [p for p in [PERSON_STATS_PQ, NAME_INDEX_PQ, GLOBAL_AVG_TXT] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing artifacts for inference: {missing}")
    print("Artifacts present ✅")

def to_list_of_str(x):
    """Normalize genres: accept array<string> or CSV and return list[str]."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple)):
        return [str(i).strip() for i in x if str(i).strip()]
    return [s.strip() for s in str(x).split(",") if s.strip()]

# Configure the MLflow experiment robustly
def _current_user_fallback():
    try:
        u = spark.sql("select current_user()").first()[0]
        if u: return u
    except Exception:
        pass
    for k in ("USER","USERNAME","LOGNAME"):
        if os.getenv(k): return os.getenv(k)
    return "shared"

def set_experiment_safe():
    user = _current_user_fallback()
    base = "/Users" if user != "shared" else "/Shared"
    path = f"{base}/{user}/movie_rating_experiment" if user != "shared" else f"{base}/movie_rating_experiment"
    mlflow.set_experiment(path)
    exp = mlflow.get_experiment_by_name(path)
    print(f"MLflow experiment → {path} (id={exp.experiment_id})")
    return path

EXPERIMENT_PATH = set_experiment_safe()
ensure_artifacts_exist()

# COMMAND ----------

# ===========================================
# 1 — Load Gold and create FEATURES with Spark
# ===========================================
from pyspark.sql import functions as F

def _dbfs_uri(p: str) -> str:
    # Convert /dbfs/... to dbfs:/... for Spark to understand
    if p.startswith("/dbfs/"):
        return "dbfs:" + p[len("/dbfs"):]
    if p.startswith("dbfs:"):
        return p
    raise ValueError(f"Non-DBFS path: {p}")

def build_df_gold_with_spark():
    """
    1) Read gold.prediction_input with Spark (actors, directors, writers, genres, plot, year...).
    2) Normalize 'genres' to array<string>.
    3) Load name_index/person_stats (Parquet) and calculate mean/max score by role.
    4) Derive startYear_z from year; set numVotes_log1p=0.0 (not available in Gold).
    5) Return pandas DataFrame with NUMERIC_COLS + [GENRES_COL, TEXT_COL, LABEL_COL].
    """
    # 1) Base Gold
    base = spark.table(GOLD_CATALOG_DB_TABLE).select(
        "tconst", "title", "plot", "year", "genres", "averageRating",
        "actors", "directors", "writers"
    )

    # 2) Normalize genres to array<string>
    genres_dtype = dict(base.dtypes)["genres"]
    if genres_dtype.startswith("array"):
        df = base.withColumn("genres_arr", F.col("genres"))
    else:
        df = base.withColumn(
            "genres_arr",
            F.when(F.col("genres").isNull(), F.array())
             .otherwise(F.split(F.col("genres"), r"\s*,\s*"))
        )

    # 3) Load name_index and person_stats (Spark over Parquet)
    name_index_sdf   = spark.read.parquet(_dbfs_uri(NAME_INDEX_PQ))   # ["name","nconst"]
    person_stats_sdf = spark.read.parquet(_dbfs_uri(PERSON_STATS_PQ))  # ["nconst","role","score_role"]

    # Read global_avg (Python) for imputing where missing
    with open(GLOBAL_AVG_TXT, "r") as f:
        global_avg = float(f.read().strip())

    # Helper: calculate scores by role
    def add_role_scores(base_df, list_col, role_name, out_prefix):
        ex = (base_df
              .select("tconst", F.explode_outer(F.col(list_col)).alias("name"))
              .join(name_index_sdf, on="name", how="left"))
        ps = person_stats_sdf.filter(F.col("role") == role_name).select("nconst","score_role")
        agg = (ex.join(ps, on="nconst", how="left")
                 .groupBy("tconst")
                 .agg(F.avg("score_role").alias(f"{out_prefix}_score_mean"),
                      F.max("score_role").alias(f"{out_prefix}_score_max")))
        out = (base_df.join(agg, on="tconst", how="left")
                      .fillna({f"{out_prefix}_score_mean": global_avg,
                               f"{out_prefix}_score_max":  global_avg}))
        return out

    # 4) Scores by actor / director / writer
    df = add_role_scores(df, "actors",    "actor",    "actor")
    df = add_role_scores(df, "directors", "director", "director")
    df = add_role_scores(df, "writers",   "writer",   "writer")

    # 5) startYear_z (z-score over year) and numVotes_log1p = 0.0
    stats = df.agg(F.mean("year").alias("mu"), F.stddev_samp("year").alias("sigma")).first()
    mu, sigma = (stats.mu or 0.0), (stats.sigma if stats.sigma not in (None, 0.0) else 1.0)
    df = df.withColumn("startYear_z", (F.col("year").cast("double") - F.lit(mu)) / F.lit(sigma))
    df = df.withColumn("numVotes_log1p", F.lit(0.0))

    # 6) Final selection → pandas
    df = df.select(
        "actor_score_mean","actor_score_max",
        "director_score_mean","director_score_max",
        "writer_score_mean","writer_score_max",
        "numVotes_log1p","startYear_z",
        F.col("genres_arr").alias(GENRES_COL),
        F.col("plot").alias(TEXT_COL),
        F.col("averageRating").alias(LABEL_COL),
    )

    pdf = df.toPandas()
    pdf[GENRES_COL] = pdf[GENRES_COL].apply(lambda x: x if isinstance(x, list) else [])
    return pdf

df_gold = build_df_gold_with_spark()
print("df_gold:", df_gold.shape)
df_gold.head(5)

# COMMAND ----------

# ==================================================
# 2 — Sklearn transformers (genres and plot)
# ==================================================
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import numpy as np

class GenresBinarizer(BaseEstimator, TransformerMixin):
    """One-hot for genres: from list[str] to binary matrix."""
    def __init__(self):
        pass  # no hyperparameters

    def fit(self, X, y=None):
        self.mlb_ = MultiLabelBinarizer().fit(X)
        return self

    def transform(self, X):
        return self.mlb_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"genre__{g}" for g in self.mlb_.classes_])

# COMMAND ----------

# ==========================================================
# 3 — Build Sklearn Pipeline and prepare the data
# ==========================================================
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Native pipeline for text: TF-IDF -> SVD
text_pipe = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF)),
    ("svd",   TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)),
])

# ColumnTransformer: numeric columns pass through; genres via binarizer; plot via text_pipe
preprocess = ColumnTransformer(
    transformers=[
        ("num",    "passthrough", NUMERIC_COLS),
        ("genres", GenresBinarizer(), GENRES_COL),
        ("plot",   text_pipe, TEXT_COL),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

# Assemble the complete Pipeline (preprocessing + model)
model_pipeline = Pipeline(steps=[
    ("pre", preprocess),
    ("rf",  RandomForestRegressor(
        n_estimators=500, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
    )),
])

# Splits
X_df = df_gold[NUMERIC_COLS + [GENRES_COL, TEXT_COL]].copy()
y    = df_gold[LABEL_COL].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=RANDOM_STATE
)

X_train.shape, X_test.shape

# COMMAND ----------

# ===================================================
# 4 — Train, evaluate and log the Pipeline
# ===================================================
with mlflow.start_run() as run:
    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae",  mae)
    mlflow.log_metric("r2",   r2)

    mlflow.sklearn.log_model(
        model_pipeline,
        artifact_path="model",
        pip_requirements=["mlflow","scikit-learn","pandas","numpy","scipy","pyarrow"],
    )
    RUN_ID_SK = run.info.run_id

print("Logged sklearn run:", RUN_ID_SK, "| RMSE:", rmse, "| MAE:", mae, "| R2:", r2)

# COMMAND ----------

# ============================================================
# 5 — Register in the Registry and move to Production
# ============================================================
client = MlflowClient()

reg = mlflow.register_model(
    model_uri=f"runs:/{RUN_ID_SK}/model",
    name=MODEL_SKLEARN_NAME
)
print("Registered:", MODEL_SKLEARN_NAME, "v", reg.version)

# Wait for READY and promote
for _ in range(120):
    mv = client.get_model_version(MODEL_SKLEARN_NAME, reg.version)
    if mv.status == "READY":
        break
    time.sleep(1)

client.transition_model_version_stage(
    name=MODEL_SKLEARN_NAME, version=reg.version,
    stage="Production", archive_existing_versions=True
)
try:
    client.set_registered_model_alias(MODEL_SKLEARN_NAME, "prod", reg.version)
except Exception:
    pass

print(f"✅ {MODEL_SKLEARN_NAME} v{reg.version} → Production")

# COMMAND ----------

# =========================================================
# 6 — Wrapper pyfunc WITHOUT Spark (pandas-only) — API
#         (sklearn embedded as local artifact)
# =========================================================
class MovieAPI(PythonModel):
    """
    Inference wrapper without Spark:
      - Maps names→nconst and calculates scores with pandas.
      - Builds features NUMERIC_COLS + [genres, plot].
      - Calls the sklearn Pipeline **embedded** in 'sk_model/'.
    """

    def load_context(self, context):
        # Supporting artifacts
        self.person_stats = pd.read_parquet(context.artifacts["person_stats"])[["nconst","role","score_role"]]
        self.name_index   = pd.read_parquet(context.artifacts["name_index"])   # ["name","nconst"]
        with open(context.artifacts["global_avg"], "r") as f:
            self.global_avg = float(f.read().strip())

        # ⚠️ Load sklearn from the packaged local artifact (not from models:/…)
        sk_local_path = context.artifacts["sk_model"]     # directory of the already downloaded sklearn model
        self.pipe = mlflow.pyfunc.load_model(sk_local_path)

        self.numeric_cols = list(NUMERIC_COLS)

    # —— Helpers in pandas ——
    def _names_to_ids(self, df, col_name, out_col):
        ex = df[["tconst", col_name]].explode(col_name)
        ex = ex.merge(self.name_index, left_on=col_name, right_on="name", how="left")
        ids = ex.groupby("tconst")["nconst"].apply(lambda s: [x for x in s.dropna()]).rename(out_col)
        return df.drop(columns=[out_col], errors="ignore").merge(ids, on="tconst", how="left")

    def _add_role_scores(self, df, col_ids, role_name, out_prefix):
        ex = df[["tconst", col_ids]].explode(col_ids).rename(columns={col_ids:"nconst"})
        ps = self.person_stats[self.person_stats["role"]==role_name][["nconst","score_role"]]
        j  = ex.merge(ps, on="nconst", how="left")

        agg = j.groupby("tconst")["score_role"].agg(["mean","max"]).rename(
            columns={"mean":f"{out_prefix}_score_mean","max":f"{out_prefix}_score_max"}
        ).reset_index()

        out = df.merge(agg, on="tconst", how="left")
        out[f"{out_prefix}_score_mean"] = out[f"{out_prefix}_score_mean"].fillna(self.global_avg)
        out[f"{out_prefix}_score_max"]  = out[f"{out_prefix}_score_max"].fillna(self.global_avg)
        return out

    def _build_rows(self, genres, plot, actors, directors, writers):
        rows = []
        for a, d, w in itertools.product(actors or [], directors or [], writers or []):
            rows.append({
                "tconst": f"combo_{hash((a,d,w)) & 0xffffffff}",
                "title":  f"{a} + {d} + {w}",
                GENRES_COL: list(genres or []),
                TEXT_COL:   plot or "",
                "numVotes_log1p": 0.0,
                "startYear_z": 0.0,
                "actors":    [a],
                "directors": [d],
                "writers":   [w],
            })
        return pd.DataFrame(rows)

    def _to_pipeline_input(self, df):
        df = self._names_to_ids(df, "actors",    "actors_ids")
        df = self._names_to_ids(df, "directors", "directors_ids")
        df = self._names_to_ids(df, "writers",   "writers_ids")
        for c in ["actors_ids","directors_ids","writers_ids"]:
            df[c] = df[c].apply(lambda x: x or [])

        df = self._add_role_scores(df, "actors_ids",    "actor",    "actor")
        df = self._add_role_scores(df, "directors_ids", "director", "director")
        df = self._add_role_scores(df, "writers_ids",   "writer",   "writer")

        for c in self.numeric_cols:
            if c not in df.columns:
                df[c] = 0.0
        X = df[self.numeric_cols + [GENRES_COL, TEXT_COL]].copy()
        meta = df[["title"]].copy()
        return X, meta

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        out_all = []
        pdf = model_input.fillna({TEXT_COL: ""})

        for _, row in pdf.iterrows():
            genres    = row.get("genres", []) or []
            plot      = row.get("plot", "") or ""
            actors    = row.get("actors", []) or []
            directors = row.get("directors", []) or []
            writers   = row.get("writers", []) or []
            top_k     = int(row.get("top_k", 10))

            df_rows = self._build_rows(genres, plot, actors, directors, writers)
            if df_rows.empty:
                out_all.append({"predictions": []})
                continue

            X, meta = self._to_pipeline_input(df_rows)
            preds = self.pipe.predict(X)
            meta["rating_estimado"] = preds.astype(float)
            meta = meta.sort_values("rating_estimado", ascending=False).head(top_k)

            out_all.append({
                "predictions": [
                    {"combination": t, "rating_estimado": float(s)}
                    for t, s in meta[["title","rating_estimado"]].itertuples(index=False)
                ]
            })
        return pd.DataFrame(out_all)

# COMMAND ----------

# ===========================================================
# 7 — Log and register the API wrapper (→ Production)
#           packaging sklearn as a local artifact
# ===========================================================
import tempfile
from mlflow.artifacts import download_artifacts

# 1) Download the sklearn Production model to a temporary local directory
sk_local_dir = download_artifacts(artifact_uri=f"models:/{MODEL_SKLEARN_NAME}/Production")
print("Sklearn local dir:", sk_local_dir)

# 2) Log the wrapper including ALL artifacts (parquets + sk_model)
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="serving_model",
        python_model=MovieAPI(),
        artifacts={
            "person_stats": PERSON_STATS_PQ,
            "name_index":   NAME_INDEX_PQ,
            "global_avg":   GLOBAL_AVG_TXT,
            "sk_model":     sk_local_dir,     # <<— the embedded sklearn (directory)
        },
        pip_requirements=[
            "mlflow","pandas","numpy","scipy","pyarrow","scikit-learn"
        ],
    )
    RUN_ID_API = run.info.run_id
    print("Logged API run:", RUN_ID_API)

# 3) Register the wrapper and promote to Production
client = MlflowClient()
reg_api = mlflow.register_model(
    model_uri=f"runs:/{RUN_ID_API}/serving_model",
    name=API_MODEL_NAME
)
print("Registered:", API_MODEL_NAME, "v", reg_api.version)

for _ in range(120):
    mv = client.get_model_version(API_MODEL_NAME, reg_api.version)
    if mv.status == "READY":
        break
    time.sleep(1)

client.transition_model_version_stage(
    name=API_MODEL_NAME, version=reg_api.version,
    stage="Production", archive_existing versions=True
)
try:
    client.set_registered_model_alias(API_MODEL_NAME, "prod", reg_api.version)
except Exception:
    pass

print(f"✅ {API_MODEL_NAME} v{reg_api.version} → Production")

# COMMAND ----------

# ===========================================
# 8 — Smoke test local (Registry)
# ===========================================
from mlflow.pyfunc import PyFuncModel

loaded_api = mlflow.pyfunc.load_model("models:/movie_rating_api/Production")  # or @prod
assert isinstance(loaded_api, PyFuncModel)

payload = pd.DataFrame([{
    "genres": ["Comedy","Romance"],
    "plot": "Office romance spirals into hilarious misunderstandings.",
    "actors": ["Jack Lemmon","Walter Matthau"],
    "directors": ["Billy Wilder"],
    "writers": ["Nora Ephron","Richard Curtis"],
    "top_k": 5
}])

resp = loaded_api.predict(payload)
preds = pd.DataFrame(resp.iloc[0]["predictions"]).rename(
    columns={"combination":"Combination","rating_estimado":"Estimated Rating"}
)
if not preds.empty:
    preds["Estimated Rating"] = preds["Estimated Rating"].round(2)
    preds.insert(0, "#", range(1, len(preds)+1))
preds

# COMMAND ----------

# ======================================================
# 9 — Robust HTTP call to the endpoint (with retry)
# ======================================================
import os, time, requests, pandas as pd

dbutils.widgets.text("serving_url", "")
dbutils.widgets.text("databricks_token", "")

SERVING_URL = dbutils.widgets.get("serving_url")
DATABRICKS_TOKEN = dbutils.widgets.get("databricks_token")

headers = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

# --------- Payload ---------
# To speed up the FIRST call (cold start), try the minimal mode (uncomment one of the two options):

# (A) MINIMAL MODE (recommended for first ping)
row = {
    "genres": ["Comedy"],
    "plot": "A light-hearted office romance.",
    "actors": ["Jack Lemmon"],       # 1 actor
    "directors": ["Billy Wilder"],   # 1 director
    "writers": ["Nora Ephron"],      # 1 writer
    "top_k": 1
}

body = {"dataframe_records": [row]}

# --------- Call helper with retry ---------
def call_with_retry(url, headers, json, max_retries=4, connect_timeout=15, read_timeout=300):
    """
    Makes POST with exponential retries.
    - short connect_timeout (15s)
    - long read_timeout (300s) in case the container is starting
    """
    backoff = 2
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            # requests accepts tuple (connect, read)
            resp = requests.post(url, headers=headers, json=json, timeout=(connect_timeout, read_timeout))
            # If the endpoint is 'Ready' but still starting internally it may give 503/502
            if resp.status_code >= 500:
                print(f"[attempt {attempt}] HTTP {resp.status_code}. Retrying in {backoff}s…")
                time.sleep(backoff)
                backoff *= 2
                continue
            return resp
        except (requests.ReadTimeout, requests.ConnectionError) as e:
            last_exc = e
            print(f"[attempt {attempt}] {type(e).__name__}: {e}. Retrying in {backoff}s…")
            time.sleep(backoff)
            backoff *= 2
    if last_exc:
        raise last_exc
    raise RuntimeError("Request failed without exception")

# --------- Call ---------
resp = call_with_retry(SERVING_URL, headers, body)

print("HTTP:", resp.status_code)
if resp.status_code != 200:
    print(resp.text[:1200])
resp.raise_for_status()

# --------- Parse response ---------
js = resp.json()
pred_list = []
if "predictions" in js and js["predictions"]:
    # Structure: {"predictions": [ {"predictions": [ {...}, ... ]} ]}
    pred_list = js["predictions"][0].get("predictions", [])

df_pred = pd.DataFrame(pred_list)
if df_pred.empty:
    print("⚠️ No predictions received.")
else:
    df_pred = df_pred.rename(columns={"combination":"Combination", "rating_estimado":"Estimated Rating"})
    df_pred["Estimated Rating"] = df_pred["Estimated Rating"].astype(float).round(2)
    df_pred.insert(0, "#", range(1, len(df_pred)+1))
    display(df_pred)