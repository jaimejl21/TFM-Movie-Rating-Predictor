import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import cloudpickle
import gzip

# ====================================================
# 1️⃣ Import Custom Transformers
# ====================================================
# These classes must exist in a local file called custom_transformers.py
# to properly unpickle the model that references them.
from custom_transformers import GenresBinarizer, SynopsisVectorizer

# ====================================================
# 2️⃣ Streamlit Page Configuration
# ====================================================
st.set_page_config(
    page_title="🎬 Movie Rating Predictor",
    page_icon="🎥",
    layout="centered",
)

st.title("🎬 Movie Rating Predictor")
st.write("Predict the average IMDb rating of a movie based on its synopsis, genres, cast, and director.")

# ====================================================
# 3️⃣ Model Download & Load
# ====================================================
# Update this URL with your actual Hugging Face model file link
MODEL_URL = "https://huggingface.co/jaimejl21/movie-rating-predictor/resolve/main/movie_rating_rf_sklearn.pkl.gz"
MODEL_PATH = "movie_rating_rf_sklearn.pkl"


@st.cache_data(show_spinner=True)
def download_model() -> str:
    """
    Download the serialized model file from Hugging Face Hub.
    This only happens once per session and is cached afterward.
    """
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Hugging Face...")
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("✅ Model successfully downloaded.")
    return MODEL_PATH


@st.cache_resource(show_spinner=True)
def load_model(path: str):
    """
    Load the model from disk using cloudpickle.
    It supports custom transformers defined in custom_transformers.py.
    """
     with gzip.open(path, "rb") as f:
        return cloudpickle.load(f)


st.write("⏳ Loading model...")
model = load_model(download_model())
st.success("✅ Model loaded successfully.")

# ====================================================
# 4️⃣ User Input Form
# ====================================================
st.subheader("🎞️ Enter movie details")

title = st.text_input("Movie title", "")
director = st.text_input("Director", "")
actors = st.text_area("Main actors (comma-separated)", "")
genres = st.multiselect(
    "Genres",
    ["Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"],
)
synopsis = st.text_area("Synopsis", "")

# ====================================================
# 5️⃣ Make Prediction
# ====================================================
if st.button("🎯 Predict Rating"):
    if not synopsis or not genres or not director:
        st.warning("Please fill in at least the synopsis, director, and genres.")
    else:
        # Prepare input as a pandas DataFrame
        input_data = pd.DataFrame(
            {
                "title": [title],
                "director": [director],
                "actors": [actors],
                "genres": [genres],
                "synopsis": [synopsis],
            }
        )

        try:
            prediction = model.predict(input_data)[0]
            st.success(f"🌟 Predicted rating: **{prediction:.2f}/10**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# ====================================================
# 6️⃣ Footer
# ====================================================
st.markdown("---")
st.caption(
    "Developed for TFM 🎓 | Model trained on Databricks and deployed with Streamlit 🌐"
)


