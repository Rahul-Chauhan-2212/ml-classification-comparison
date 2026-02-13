import joblib
import matplotlib.pyplot as plt
import numpy as np
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
        "logistic_regression": joblib.load("model/pkl/logistic_regression.pkl"),
        "decision_tree": joblib.load("model/pkl/decision_tree.pkl"),
        "knn": joblib.load("model/pkl/knn.pkl"),
        "naive_bayes": joblib.load("model/pkl/naive_bayes.pkl"),
        "random_forest": joblib.load("model/pkl/random_forest.pkl"),
        "xgboost": joblib.load("model/pkl/xgboost.pkl"),
    }

    scaler = joblib.load("model/pkl/scaler.pkl")
    metrics = joblib.load("model/pkl/metrics.pkl")
    encoder = joblib.load("model/pkl/encoder.pkl")

    return models, scaler, encoder, metrics


# -----------------------------
# Load data once
# -----------------------------
@st.cache_data
def load_data():
    print("Loading complete dataset....")
    return pd.read_csv("adult.csv")


models, scaler, encoder, metrics = load_models()

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

    # -----------------------------
    # Load Default test.csv
    # -----------------------------
    try:
        sample_df = pd.read_csv("test.csv")

        csv = sample_df.to_csv(index=False).encode("utf-8")

        st.markdown("### 📥 Download Sample Test File")

        st.download_button(
            label="⬇️ Download test.csv",
            data=csv,
            file_name="test.csv",
            mime="text/csv"
        )

    except:
        st.warning("⚠️ test.csv not found in project root")

    st.markdown("---")

    # -----------------------------
    # Upload Section
    # -----------------------------
    uploaded_file = st.file_uploader("Upload New Data (CSV)", key="predict")

    model_name = st.selectbox("Select Model", list(name_mapping.values()))

    selected_model_key = reverse_mapping[model_name]

    # -----------------------------
    # Select Data Source
    # -----------------------------
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Using Uploaded File")
    else:
        df = None

    # -----------------------------
    # Prediction Logic
    # -----------------------------
    if df is not None:

        st.write("📄 Data Preview")
        st.write("Test Data Size: ", df.shape[0])
        st.dataframe(df.head(), width="stretch")
        model = models[selected_model_key]

        print("Shape:", df.shape)
        print("Duplicate rows:", df.duplicated().sum())
        print("Dropping duplicate rows...")
        df = df.drop_duplicates()
        print("Shape after dropping duplicate: ", df.shape)
        print("Missing Entries: \n", df.isnull().sum())
        df.replace('?', np.nan, inplace=True)
        df = df.dropna()
        print("Shape after dropping wrong entries:", df.shape)

        X = df.copy()

        # Drop target column if present
        if "income" in X.columns:
            X = X.drop("income", axis=1)

        # -----------------------------
        # Apply scaler (if needed)
        # -----------------------------
        # ------------------------------
        # Column Types
        # ------------------------------
        categorical_cols = X.select_dtypes(include=["object"]).columns
        numerical_cols = X.select_dtypes(exclude=["object"]).columns

        # ------------------------------
        # Encoding
        # ------------------------------
        X_cat = encoder.transform(X[categorical_cols])

        # ------------------------------
        # Scaling
        # ------------------------------
        X_num = scaler.transform(X[numerical_cols])

        # ------------------------------
        # Combine Features
        # ------------------------------
        X_processed = np.hstack([X_num, X_cat])

        y_pred = model.predict(X_processed)

        df["predicted_income"] = y_pred
        df["predicted_income"] = df["predicted_income"].map({
            0: "<=50K",
            1: ">50K"
        })

        priority_cols = ["income", "predicted_income"]
        priority_cols = [col for col in priority_cols if col in df.columns]

        df = df[priority_cols + [col for col in df.columns if col not in priority_cols]]

        st.subheader("✅ Predictions")
        st.write("📄 The below table has `predicted_income` column which is calculated based on selected model.")
        st.dataframe(df, width="stretch")

        # -----------------------------
        # Download Predictions
        # -----------------------------
        csv_pred = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Predictions Dataset",
            data=csv_pred,
            file_name=f"predictions_{selected_model_key}.csv",
            mime="text/csv"
        )

    else:
        st.error("❌ No data available. Please upload a CSV.")

# -----------------------------
# PAGE 3 — Dataset Information
# -----------------------------
elif page == "Dataset Information":

    st.title("Adult Income Dataset Information")

    st.markdown("""
    **Source:** [Kaggle - Adult Income Dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)

    **Description:**
    An individual’s annual income results from various factors. Intuitively, it is influenced by the individual’s education level, age, gender, occupation, and etc.
    This dataset contains 14 attributes extracted from the 1994 Census database.
    
    **Features:**
    1. **age** - continuous.
    2. **workclass** - Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    3. **fnlwgt** - continuous.
    4. **education** - Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    5. **education-num** - continuous.
    6. **marital-status** - Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    7. **occupation** - Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    8. **relationship** - Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    9. **race** - White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    10. **sex** - Female, Male.
    11. **capital-gain** - continuous.
    12. **capital-loss** - continuous.
    13. **hours-per-week** - continuous.
    14. **native-country** - United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    
    **Target:**
    - **income**: <=50K, >50K
    """)

    st.divider()

    st.subheader("Dataset Exploration")

    try:
        df = load_data()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        col4.metric("Missing Values", df.isnull().sum().sum())

        st.divider()

        tabs = st.tabs([
            "First 10 Rows",
            "Last 10 Rows",
            "Dataset Statistics",
            "Income Distribution",
            "Correlation Heatmap"
        ])

        with tabs[0]:
            st.write("### First 10 Rows")
            st.dataframe(df.head(10), width="stretch")

        with tabs[1]:
            st.write("### Last 10 Rows")
            st.dataframe(df.tail(10), width="stretch")

        with tabs[2]:
            st.write("### Dataset Statistics")
            st.write(df.describe())

        with tabs[3]:
            st.write("#### Income Distribution")
            fig, ax = plt.subplots()
            sns.countplot(
                x="income",
                hue="income",
                data=df,
                ax=ax,
                palette={"<=50K": "#4C72B0", ">50K": "#55A868"},
                legend=False
            )
            st.pyplot(fig)

        with tabs[4]:
            st.write("#### Correlation Heatmap")

            # Encode categorical columns
            df_encoded = df.copy()
            for col in df_encoded.select_dtypes(include=['object']).columns:
                df_encoded[col] = df_encoded[col].astype('category').cat.codes

            corr = df_encoded.corr()

            # Create larger figure
            fig, ax = plt.subplots(figsize=(14, 10))

            sns.heatmap(
                corr,
                cmap="coolwarm",
                annot=True,
                linewidths=0.5,
                ax=ax
            )

            ax.set_title("Feature Correlation Matrix", fontsize=16)
            ax.tick_params(axis='x', labelrotation=45)
            ax.tick_params(axis='y', labelrotation=0)

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
