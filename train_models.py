from sklearn.model_selection import train_test_split
import pandas as pd

from model import (
    logistic_regression,
    decision_tree,
    knn,
    naive_bayes,
    random_forest,
    xgboost,
)


def train_all_models():
    # Load dataset
    print("Loading Breast Cancer Dataset....")
    df = pd.read_csv("breast_cancer.csv")
    print("Breast Cancer Dataset Loaded....")
    df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})

    X = df.drop(['Diagnosis'], axis=1)
    y = df['Diagnosis']

    # Split Train and Test Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Models....")

    models = {
        "Logistic Regression": logistic_regression(X_train, y_train),
        "Decision Tree": decision_tree(X_train, y_train),
        "KNN": knn(X_train, y_train),
        "Naive Bayes": naive_bayes(X_train, y_train),
        "Random Forest": random_forest(X_train, y_train),
        "XGBoost": xgboost(X_train, y_train),
    }

    print("Training Completed....")
    return models, (X_test, y_test)
