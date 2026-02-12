import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)


# -----------------------------
# Load saved objects
# -----------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/pkl/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/pkl/decision_tree.pkl"),
        "KNN": joblib.load("model/pkl/knn.pkl"),
        "Naive Bayes": joblib.load("model/pkl/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/pkl/random_forest.pkl"),
        "XGBoost": joblib.load("model/pkl/xgboost.pkl"),
    }

    scaler = joblib.load("model/pkl/scaler.pkl")
    metrics = joblib.load("model/pkl/metrics.pkl")

    return models, scaler, metrics


models, scaler, metrics = load_models()

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Model Comparison", "Predict on New Data", "Dataset Information"]
)

# -----------------------------
# PAGE 1 — Model Comparison
# -----------------------------
if page == "Model Comparison":

    st.title("📊 Model Performance Comparison")

    # Load stored metrics
    metrics_df = pd.DataFrame(metrics).T

    st.subheader("✅ Evaluation Metrics Table")
    st.dataframe(metrics_df)

    st.info("Metrics computed using training/test split during model training.")

# -----------------------------
# PAGE 2 — Predict on New Data
# -----------------------------
elif page == "Predict on New Data":

    st.title("🔮 Predict on New Data")

    uploaded_file = st.file_uploader("Upload New Data (CSV)", key="predict")

    model_name = st.selectbox("Select Model", list(models.keys()))

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("📄 Uploaded Data Preview")
        st.dataframe(df.head())

        model = models[model_name]

        X = df.copy()

        if "target" in X.columns:
            X = X.drop("target", axis=1)

        if model_name in ["Logistic Regression", "KNN"]:
            X = scaler.transform(X)

        predictions = model.predict(X)

        df["Prediction"] = predictions

        st.subheader("✅ Predictions")
        st.dataframe(df)

# -----------------------------
# PAGE 3 — Dataset Information
# -----------------------------
elif page == "Dataset Information":

    st.title("📚 Dataset Information")

    st.markdown("""
    ### 🫀 Heart Disease Dataset

    **Problem Type:** Binary Classification  

    **Goal:** Predict whether a patient has heart disease.

    **Features Include:**
    - Age
    - Sex
    - Chest Pain Type
    - Resting Blood Pressure
    - Cholesterol
    - Fasting Blood Sugar
    - ECG Results
    - Max Heart Rate
    - Exercise Induced Angina
    - ST Depression
    - Slope
    - Number of Major Vessels
    - Thalassemia

    **Target Variable:**  
    - `0` → No Heart Disease  
    - `1` → Heart Disease

    ---
    ### 🎯 Why This Dataset?

    ✔ Meets assignment criteria (≥12 features, ≥500 rows)  
    ✔ Suitable for classification models  
    ✔ Works well with AUC metric  
    ✔ Popular benchmark dataset
    """)

    st.success("Dataset description ready for README & PDF submission ✅")
