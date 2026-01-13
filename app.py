import streamlit as st
from train_models import train_all_models
from evaluate_models import evaluate_models
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="ML Classification Models Comparison", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Breast Cancer Wisconsin(Diagnostic) Classification</h1>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Model Comparison", "Predict on New Data", "Dataset Information"])


# Load models
@st.cache_resource
def load_models():
    """Load Trained Models"""
    models, (X_test, y_test) = train_all_models()
    return models, X_test, y_test


# Load Metrics
@st.cache_data
def load_metrics(models, X_test, y_test):
    """Load saved metrics"""
    metrics, confusion_matrices = evaluate_models(models, X_test, y_test)
    return metrics, confusion_matrices


# PAGE 1: Model Comparison
if page == "Model Comparison":
    st.markdown('<h2 class="main-header">Model Performance Comparison Dashboard</h2>', unsafe_allow_html=True)
    st.subheader("Evaluation Metrics Table")
    models, X_test, y_test = load_models()
    metrics, confusion_matrices = evaluate_models(models, X_test, y_test)
    metrics_df = pd.DataFrame(metrics).T  # transpose view
    st.dataframe(metrics_df)

    model_names = list(models.keys())
    selected_model = st.selectbox("Select a Model for detailed analysis:", model_names)
    st.subheader(f"{selected_model}")

    col1, col2 = st.columns(2)
    model_metrics = metrics.get(selected_model, None)
    with col1:
        if model_metrics:
            st.markdown("Metrics")
            selected_metrics = metrics[selected_model]
            st.metric("Accuracy", f"{selected_metrics['accuracy']:.4f}")
            st.metric("AUC Score", f"{selected_metrics['auc']:.4f}")
            st.metric("Precision", f"{selected_metrics['precision']:.4f}")
            st.metric("Recall", f"{selected_metrics['recall']:.4f}")
            st.metric("F1 Score", f"{selected_metrics['f1']:.4f}")
            st.metric("MCC Score", f"{selected_metrics['mcc']:.4f}")

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrices[selected_model]
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)


# PAGE 2: Predict Data on new Data
elif page == "Predict on New Data":
    st.markdown('<h2 class="main-header">Predict Breast Cancer on Custom Input</h2>', unsafe_allow_html=True)
    st.header("Upload Test Data and Make Predictions")

    # Load models once
    models, X_test, y_test = load_models()

    # Dropdown to select trained model
    model_names = list(models.keys())
    selected_model = st.selectbox("Select a Trained Model", model_names)

    uploaded_file = st.file_uploader("Upload CSV File for Prediction", type=["csv"])

    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Test Data")
        st.dataframe(test_df)

        # Predict
        if st.button("Run Prediction"):
            model = models[selected_model]
            preds = model.predict(test_df)

            pred_df = pd.DataFrame({
                "Prediction": preds
            })

            st.subheader("Prediction Results")
            st.dataframe(pred_df)

            # Optional: convert 0/1 to labels (Benign/Malignant)
            st.subheader("Prediction Labels")
            pred_df["Label"] = pred_df["Prediction"].map({0: "Benign", 1: "Malignant"})
            st.dataframe(pred_df)



# PAGE 3: Dataset Information
elif page == "Dataset Information":
    st.markdown('<h2 class="main-header">Dataset Information</h2>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>BITS ML Assignment 2 - Classification Models Comparison</div>",
    unsafe_allow_html=True)
