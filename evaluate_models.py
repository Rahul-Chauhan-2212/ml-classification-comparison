from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)


def evaluate_models(models, X_test, y_test):
    metrics = {}
    confusion_matrices = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        # AUC -> use predict_proba if available
        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        except:
            auc = 0.0

        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": auc,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_test, y_pred),
        }
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm

    return metrics, confusion_matrices
