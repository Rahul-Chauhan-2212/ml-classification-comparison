import streamlit as st

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
@st.cache_data
def load_models():
    """Load all trained models"""
    models = {}
    scaler = None
    return models, scaler


# Load Metrics
@st.cache_data
def load_metrics():
    """Load saved metrics"""
    return None


# PAGE 1: Model Comparison
if page == "Model Comparison":
    st.markdown('<h2 class="main-header">Model Performance Comparison Dashboard</h2>', unsafe_allow_html=True)

# PAGE 2: Predict Data on new Data
elif page == "Predict on New Data":
    st.markdown('<h2 class="main-header">Predict Breast Cancer on Custom Input</h2>', unsafe_allow_html=True)
    st.header("Upload Test Data and Make Predictions")
    st.info(
        "Upload a CSV file with test data. The file should have the same features as the training data (excluding the target column).")


# PAGE 3: Dataset Information
elif page == "Dataset Information":
    st.markdown('<h2 class="main-header">Dataset Information</h2>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>BITS ML Assignment 2 - Classification Models Comparison</div>",
    unsafe_allow_html=True)
