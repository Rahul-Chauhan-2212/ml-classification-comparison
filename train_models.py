import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("train.csv")

print(df.head())
print("Shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

# ------------------------------
# Features / Target
# ------------------------------
X = df.drop("income", axis=1)
y = df["income"].map({
    "<=50K": 0,
    ">50K": 1
})


# ------------------------------
# Column Types
# ------------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# ------------------------------
# Encoding
# ------------------------------
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat = encoder.fit_transform(X[categorical_cols])

# ------------------------------
# Scaling
# ------------------------------
scaler = StandardScaler()
X_num = scaler.fit_transform(X[numerical_cols])

# ------------------------------
# Combine Features
# ------------------------------
X_processed = np.hstack([X_num, X_cat])

# ------------------------------
# Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.1, random_state=42, stratify=y
)

# ------------------------------
# Models
# ------------------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(eval_metric='logloss')
}

metrics = {}

# ------------------------------
# Output Directory
# ------------------------------
os.makedirs("model/pkl", exist_ok=True)

# ------------------------------
# Train + Evaluate
# ------------------------------
for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    joblib.dump(model, f"model/pkl/{name}.pkl")

# ------------------------------
# Save Preprocessing Objects
# ------------------------------
joblib.dump(scaler, "model/pkl/scaler.pkl")
joblib.dump(encoder, "model/pkl/encoder.pkl")
joblib.dump(metrics, "model/pkl/metrics.pkl")

print("✅ Models, scaler, encoder & metrics saved!")
