# 🎬 IMDb Movie Rating Predictor

[![Streamlit App](https://img.shields.io/badge/Live_App-Streamlit-blue?logo=streamlit)](https://tfm-movie-rating-predictor.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green?logo=python)](https://www.python.org/)
[![Databricks](https://img.shields.io/badge/Databricks-Big%20Data-orange?logo=databricks)](https://www.databricks.com/)

This repository contains the source code and documentation of my **Master’s Thesis (TFM)**:  
**“Design and Implementation of a Scalable Architecture for Predicting Movie Ratings with IMDb Data”**.  

The project builds an **end-to-end data pipeline** using IMDb datasets, enriches them with external sources, and trains a **RandomForestRegressor model** to predict the rating of hypothetical movies.  
The results are exposed through a **Streamlit web app** that allows users to test scenarios with different genres, plots, actors, directors, and writers.

🌍 Live demo: [Streamlit App](https://tfm-movie-rating-predictor.streamlit.app/)

---

## 🚀 Features

- **Data Pipeline (Medallion Architecture)**  
  - Landing → Bronze → Silver → Gold layers  
  - Ingestion of IMDb datasets (`.tsv.gz`) into Delta Lake  
  - Data cleaning, normalization, and enrichment with OMDb API (plots)  
- **Machine Learning**  
  - RandomForestRegressor trained on enriched Gold dataset  
  - MLflow for experiment tracking and model versioning  
- **Model Serving**  
  - Deployed on Databricks Model Serving as a REST API  
- **Web Application**  
  - Built with Streamlit  
  - Interactive inputs: genres, plot, actors, directors, writers  
  - Returns **Top-K creative scenarios** with predicted IMDb ratings  

---

## 🏗️ Architecture
IMDb Datasets → Landing (Raw) → Bronze (Delta) → Silver (Cleaned/Enriched) → Gold (Features) → ML Training → Model Serving → Streamlit App


Technologies used:
- **Databricks** (Workflows, Delta Lake, MLflow)  
- **PySpark** for distributed data processing  
- **scikit-learn / Spark MLlib** for ML modeling  
- **Streamlit** for web interface  
- **Azure Blob Storage** for data storage  

---

## 🧪 Tests

You can run them with:  pytest -q

---

## 📊 Results

- **RMSE:** 0.96  
- **MAE:** 0.72  
- **R²:** 0.50 (~50% variance explained)  

The model performs best on **mid-range movies (ratings 5–7)** and tends to:  
- Slightly **underestimate** very high-rated films.  
- **Overestimate** very low-rated films.  

---

## 📌 Limitations & Future Work

- Add more features (budget, box office, critic reviews, popularity).  
- Explore more advanced models (XGBoost, LightGBM, Transformers for NLP).  
- Improve predictions on rating extremes.  
- Optimize Databricks costs for production scenarios.  

---

## 👨‍💻 Authors

- **Jaime Jiménez López**
- **Jaume Adrover**
- **Marc Cañellas**
- **Diego Bermejo**

  Master’s in Big Data & Data Engineering, UCM (2025)  

  Supervisors: **Jorge Centeno & Alberto González**  


