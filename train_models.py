import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model.py import (
    logistic_regression,
    decision_tree,
    knn,
    naive_bayes,
    random_forest,
    xgboost
)

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv("train.csv")

print("Shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Dropping duplicate rows...")
df = df.drop_duplicates()
print("Shape after dropping duplicate: ", df.shape)
print("Missing Entries: \n", df.isnull().sum())
df.replace('?', np.nan, inplace=True)
df = df.dropna()
print("Shape after dropping wrong entries:", df.shape)

# ------------------------------
# Features / Target
# ------------------------------
X = df.drop("income", axis=1)
label_map = {0: "<=50K", 1: ">50K"}
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
# All these models are initialized in their respective python file for clarity. Please check model/py directory
models = {
    "logistic_regression": logistic_regression(),
    "decision_tree": decision_tree(),
    "knn": knn(),
    "naive_bayes": naive_bayes(),
    "random_forest": random_forest(),
    "xgboost": xgboost()
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
    cm = confusion_matrix(y_test, y_pred)

    metrics[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Confusion Matrix": {
            "Actual <=50K": {
                "Predicted <=50K": int(cm[0, 0]),  # TN
                "Predicted >50K": int(cm[0, 1])  # FP
            },
            "Actual >50K": {
                "Predicted <=50K": int(cm[1, 0]),  # FN
                "Predicted >50K": int(cm[1, 1])  # TP
            }
        }
    }

    joblib.dump(model, f"model/pkl/{name}.pkl", compress=("gzip", 9))

# ------------------------------
# Save Preprocessing Objects
# ------------------------------
joblib.dump(scaler, "model/pkl/scaler.pkl", compress=("gzip", 9))
joblib.dump(encoder, "model/pkl/encoder.pkl", compress=("gzip", 9))
joblib.dump(metrics, "model/pkl/metrics.pkl", compress=("gzip", 9))

print("✅ Models, scaler, encoder & metrics saved!")
