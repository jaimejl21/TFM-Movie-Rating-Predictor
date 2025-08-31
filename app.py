import os
import json
import time
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Config & Secrets
# -----------------------------
SERVING_URL = st.secrets.get("SERVING_URL", os.getenv("SERVING_URL", ""))
DATABRICKS_TOKEN = st.secrets.get("DATABRICKS_TOKEN", os.getenv("DATABRICKS_TOKEN", ""))

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

st.set_page_config(page_title="Movie Rating Predictor (TFM)", page_icon="🎬", layout="centered")

st.title("🎬 TFM — Movie Rating Predictor")
st.write(
    """
    This tool helps users to find the best main actor, main director and main writer for a movie, 
    given the selected genres and a brief plot. 

    Fill in the details below. 
    The app will call the Databricks endpoint and display the estimated rating predictions for each combination.
    """
)

# -----------------------------
# Inputs
# -----------------------------
def _split_csv(s: str):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

genres = st.multiselect(
    "🎭 Genres",
    [
        "Action", "Adventure", "Comedy", "Drama", "Romance", "Sci-Fi",
        "Thriller", "Horror", "Fantasy", "Mystery", "Crime", "Animation",
        "Family", "Biography", "History", "Music", "War", "Western",
        "Sport", "Documentary"
    ],
    default=["Comedy", "Romance"]
)
plot = st.text_area("📝 Plot", "A light-hearted office romance.", height=100)

actors_csv = st.text_input("🧑‍🎤 Actors (comma separated)", "Jack Lemmon, Will Smith, Walter Matthau")
directors_csv = st.text_input("🎬 Directors (comma separated)", "Billy Wilder, Roger Michell")
writers_csv = st.text_input("✍️ Writers (comma separated)", "Nora Ephron, Richard Curtis")
top_k = st.number_input("🔢 Top K combinations", min_value=1, max_value=50, value=12)

actors = _split_csv(actors_csv)
directors = _split_csv(directors_csv)
writers = _split_csv(writers_csv)

# -----------------------------
# Helper para llamada robusta
# -----------------------------
def call_with_retry(url, headers, payload, max_retries=4, connect_timeout=15, read_timeout=300):
    backoff = 2
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=(connect_timeout, read_timeout))
            if resp.status_code >= 500:
                st.info(f"[Try {attempt}] HTTP {resp.status_code}. Retrying in {backoff}s…")
                time.sleep(backoff)
                backoff *= 2
                continue
            return resp
        except (requests.ReadTimeout, requests.ConnectionError) as e:
            last_exc = e
            st.info(f"[Try {attempt}] {type(e).__name__}: {e}. Retrying in {backoff}s…")
            time.sleep(backoff)
            backoff *= 2
    if last_exc:
        raise last_exc
    raise RuntimeError("Request failed without exception")

# -----------------------------
# Predict
# -----------------------------
if st.button("🚀 Predict rating"):
    if not SERVING_URL or not DATABRICKS_TOKEN:
        st.error("Configure SERVING_URL and DATABRICKS_TOKEN in Secrets.")
    else:
        row = {
            "genres": genres,
            "plot": plot or "",
            "actors": actors,
            "directors": directors,
            "writers": writers,
            "top_k": int(top_k),
        }
        payload = {"dataframe_records": [row]}

        with st.spinner("Calling Databricks Model Serving… (first try take a while as it has cold start)"):
            try:
                resp = call_with_retry(SERVING_URL, HEADERS, payload)
                if resp.status_code != 200:
                    st.error(f"Error HTTP {resp.status_code}")
                    st.code(resp.text[:1000])
                else:
                    js = resp.json()
                    pred_list = []
                    if "predictions" in js and js["predictions"]:
                        pred_list = js["predictions"][0].get("predictions", [])

                    df = pd.DataFrame(pred_list)
                    if df.empty:
                        st.warning("⚠️ No predictions received.")
                    else:
                        df = df.rename(columns={
                            "combination": "Combination",
                            "rating_estimado": "Estimated Rating",
                            "rating": "Estimated Rating",
                        })
                        if "Estimated Rating" in df.columns:
                            df["Estimated Rating"] = pd.to_numeric(df["Estimated Rating"], errors="coerce").round(2)
                        df.insert(0, "#", range(1, len(df)+1))
                        st.dataframe(df, use_container_width=True)
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button("⬇️ Download CSV", csv, "predictions.csv", "text/csv")

                        with st.expander("🔎 View raw JSON"):
                            st.code(json.dumps(js, indent=2)[:10000])

            except Exception as e:
                st.exception(e)

