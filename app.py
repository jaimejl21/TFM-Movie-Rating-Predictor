import os
import re
import cloudpickle
import pandas as pd
import requests
import streamlit as st

# ====================================================
# 1️⃣ Config
# ====================================================
HF_MODEL_URL = "https://huggingface.co/jaimejl21/movie-rating-predictor/resolve/main/movie_rating_rf_sklearn.pkl"
CACHE_PATH = "movie_rating_rf_sklearn.pkl"

st.set_page_config(page_title="Movie Rating Predictor (TFM)", page_icon="🎬", layout="centered")

st.title("🎬 TFM — Movie Rating Predictor")
st.write("""
This tool helps users find the best main actor, main director and main writer for a movie,
given selected genres and a brief plot. Fill in the details below.
""")

# ====================================================
# 2️⃣ Download & Load Model (cached)
# ====================================================
@st.cache_data(show_spinner=False)
def download_model(url=HF_MODEL_URL, path=CACHE_PATH):
    if not os.path.exists(path):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return path

@st.cache_data(show_spinner=True)
def load_model(path):
    with open(path, "rb") as f:
        return cloudpickle.load(f)

CACHE_PATH = download_model()
model = load_model(CACHE_PATH)

# ====================================================
# 3️⃣ Helper Functions (Validation)
# ====================================================
def _split_csv(s: str):
    if not s:
        return []
    seen, out = set(), []
    for x in [t.strip() for t in s.split(",") if t.strip()]:
        if x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out

def _has_letter(s: str) -> bool:
    return any(ch.isalpha() for ch in s)

def _too_many_symbols(s: str, threshold: float = 0.5) -> bool:
    allowed = set(" .,'()-")
    if not s:
        return True
    nonsafe = sum(1 for ch in s if not (ch.isalnum() or ch in allowed))
    return (nonsafe / len(s)) > threshold

def validate_name_list(items, label):
    errors, clean = [], []
    for it in items:
        s = re.sub(r"\s+", " ", it).strip()
        if not s or not _has_letter(s) or _too_many_symbols(s):
            errors.append(f"• {label}: “{it}” looks invalid.")
            continue
        clean.append(s)
    if not clean:
        errors.append(f"• {label}: must contain at least one valid value.")
    return clean, errors

# ====================================================
# 4️⃣ User Inputs
# ====================================================
COMMON_GENRES = [
    "Action","Adventure","Comedy","Drama","Romance","Sci-Fi","Thriller","Horror","Fantasy","Mystery",
    "Crime","Animation","Family","Biography","History","Music","War","Western","Sport","Documentary"
]

genres = st.multiselect("🎭 Genres", COMMON_GENRES, default=["Comedy", "Romance"])
plot = st.text_area("📝 Plot", "A light-hearted office romance.", height=100, max_chars=100)

actors_csv = st.text_input("🧑‍🎤 Actors (comma separated)", "Jack Lemmon, Will Smith, Walter Matthau")
directors_csv = st.text_input("🎬 Directors (comma separated)", "Billy Wilder, Roger Michell")
writers_csv = st.text_input("✍️ Writers (comma separated)", "Nora Ephron, Richard Curtis")
top_k = st.number_input("🔢 Top K combinations", min_value=1, max_value=50, value=12, step=1)

actors, directors, writers = map(_split_csv, [actors_csv, directors_csv, writers_csv])

# ====================================================
# 5️⃣ Predict
# ====================================================
if st.button("🚀 Predict rating"):
    errors = []
    if not genres:
        errors.append("• Genres: select at least one genre.")
    if not plot.strip():
        errors.append("• Plot: cannot be empty.")
    if len(plot) > 100:
        errors.append("• Plot must be ≤ 100 characters.")

    actors_clean, err_a = validate_name_list(actors, "Actors")
    directors_clean, err_d = validate_name_list(directors, "Directors")
    writers_clean, err_w = validate_name_list(writers, "Writers")
    errors.extend(err_a + err_d + err_w)

    if errors:
        st.error("Please fix the following issues:\n" + "\n".join(errors))
        st.stop()

    # Generate all combinations
    rows = []
    for a in actors_clean:
        for d in directors_clean:
            for w in writers_clean:
                rows.append({
                    "genres": ", ".join(genres),
                    "plot": plot,
                    "actor": a,
                    "director": d,
                    "writer": w,
                })

    df = pd.DataFrame(rows)
    with st.spinner("🎬 Predicting ratings…"):
        try:
            # Add placeholder numeric columns expected by model
            for col in [
                "actor_score_mean", "actor_score_max",
                "director_score_mean", "director_score_max",
                "writer_score_mean", "writer_score_max",
                "numVotes_log1p", "startYear_z",
            ]:
                df[col] = 0.0

            # Normalize genres as list if model expects
            df["genres"] = df["genres"].apply(lambda g: [g] if isinstance(g, str) else g)
            df["plot"] = df["plot"].fillna("")

            preds = model.predict(df)
            df["Estimated Rating"] = preds
            df.insert(0, "#", range(1, len(df) + 1))

            st.dataframe(df.head(top_k), use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error("Error during prediction:")
            st.exception(e)
