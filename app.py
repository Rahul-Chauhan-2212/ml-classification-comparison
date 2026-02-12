import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="ML Model Dashboard",
    page_icon="📊",
    layout="wide"
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
# Model Name Mapping
# -----------------------------
name_mapping = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

reverse_mapping = {v: k for k, v in name_mapping.items()}

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

    st.title("Adult Income Classification - Model Performance Comparison")

    # Convert metrics dict → DataFrame
    metrics_df = pd.DataFrame(metrics).T

    # Remove confusion matrix column
    if "Confusion Matrix" in metrics_df.columns:
        metrics_df = metrics_df.drop(columns=["Confusion Matrix"])

    # Beautify model names
    metrics_df.index = metrics_df.index.map(
        lambda x: name_mapping.get(x, x)
    )

    st.subheader("Evaluation Metrics Table")
    st.dataframe(metrics_df, width="stretch")

    st.info("Metrics computed using training/test split during model training.")

    st.divider()

    # -----------------------------
    # Confusion Matrix Viewer
    # -----------------------------
    st.subheader("Confusion Matrix")

    selected_model = st.selectbox(
        "Select model to view confusion matrix",
        metrics_df.index
    )

    # Convert pretty name → original key
    selected_model_key = reverse_mapping[selected_model]

    cm_data = metrics[selected_model_key]["Confusion Matrix"]
    cm_df = pd.DataFrame(cm_data)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        linewidths=1,
        linecolor="black",
        ax=ax
    )

    ax.set_title(f"Confusion Matrix — {selected_model}")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    st.pyplot(fig)


# -----------------------------
# PAGE 2 — Predict on New Data
# -----------------------------
elif page == "Predict on New Data":

    st.title("Predict on New Data")

    uploaded_file = st.file_uploader("Upload New Data (CSV)", key="predict")
    model_name = st.selectbox("Select Model", list(models.keys()))

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("📄 Uploaded Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        model = models[model_name]

        X = df.copy()

        # Drop target column if present
        if "target" in X.columns:
            X = X.drop("target", axis=1)

        # Apply scaler if required
        if model_name in ["Logistic Regression", "KNN"]:
            X = scaler.transform(X)

        predictions = model.predict(X)

        df["Prediction"] = predictions

        st.subheader("✅ Predictions")
        st.dataframe(df, use_container_width=True)

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
