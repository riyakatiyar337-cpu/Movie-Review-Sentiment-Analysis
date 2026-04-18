import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

def evaluate_model(model, X_test, y_test):
    y_pred_raw = model.predict(X_test)

    if hasattr(y_pred_raw, "ndim") and y_pred_raw.ndim > 1:
        y_pred_raw = np.ravel(y_pred_raw)

    y_pred = y_pred_raw
    if not np.array_equal(np.unique(y_pred), [0, 1]):
        y_pred = np.where(y_pred >= 0.5, 1, 0)

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    # ROC-AUC only if probability exists
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    elif np.issubdtype(np.array(y_pred_raw).dtype, np.floating):
        y_proba = np.ravel(y_pred_raw)
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    else:
        metrics["roc_auc"] = None

    metrics["confusion_matrix"] = cm

    return metrics