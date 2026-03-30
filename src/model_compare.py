import time
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from config import CV_FOLDS, RANDOM_STATE


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE)
    }


def compare_models(X_train, y_train, X_test, y_test, evaluate_model):
    models = get_models()
    results = []

    for name, model in models.items():
        print(f"Training {name}...")

        # Training Time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Inference Time
        start_time = time.time()
        _ = model.predict(X_test)
        inference_time = time.time() - start_time

        # Cross Validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=CV_FOLDS,
            scoring="f1"
        )

        # Evaluation
        metrics = evaluate_model(model, X_test, y_test)

        results.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1"],
            "ROC-AUC": metrics["roc_auc"],
            "CV F1 Mean": cv_scores.mean(),
            "Training Time (s)": training_time,
            "Inference Time (s)": inference_time
        })

    return pd.DataFrame(results)