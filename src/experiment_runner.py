import time
import pandas as pd
import os
import joblib

from .feature_engineering import build_vectorizer, transform_text
from .model_factory import get_model, AVAILABLE_MODELS, AVAILABLE_FEATURES
from .hyperparameter import tune_model
from .evaluation import evaluate_model


def compute_feature_stats(X_vec, vectorizer):
    vocab_size = len(vectorizer.vocabulary_)
    total_elements = X_vec.shape[0] * X_vec.shape[1]
    sparsity = 1 - (X_vec.count_nonzero() / total_elements)
    return vocab_size, sparsity


def run_full_experiment(X_train, X_test, y_train, y_test):

    results = []

    for feature in AVAILABLE_FEATURES:
        print(f"\n==== Feature Type: {feature} ====")

        vectorizer = build_vectorizer(feature)
        X_train_vec, X_test_vec = transform_text(vectorizer, X_train, X_test)

        vocab_size, sparsity = compute_feature_stats(X_train_vec, vectorizer)

        for model_name in AVAILABLE_MODELS:
            print(f"\nTraining Model: {model_name}")

            model = get_model(model_name)

            # Hyperparameter tuning
            model, cv_score = tune_model(model, model_name, X_train_vec, y_train)

            # Training time
            start = time.time()
            model.fit(X_train_vec, y_train)
            training_time = time.time() - start

            # Training accuracy
            train_accuracy = model.score(X_train_vec, y_train)

            # Inference time
            start = time.time()
            _ = model.predict(X_test_vec)
            inference_time = time.time() - start

            metrics = evaluate_model(model, X_test_vec, y_test)
            
            # FEATURE IMPORTANCE + ERROR ANALYSIS
            # Only for best model (tfidf_uni_bi + svm)
            
            if feature == "tfidf_uni_bi" and model_name == "svm":

                print("\n===== FEATURE IMPORTANCE ANALYSIS =====")

                # Works only for linear models
                if model_name in ["logreg", "svm"]:

                    feature_names = vectorizer.get_feature_names_out()
                    coefficients = model.coef_[0]

                    # Top 20 Positive
                    top_pos_indices = coefficients.argsort()[-20:]
                    top_pos_words = [feature_names[i] for i in top_pos_indices]

                    # Top 20 Negative
                    top_neg_indices = coefficients.argsort()[:20]
                    top_neg_words = [feature_names[i] for i in top_neg_indices]

                    print("\nTop 20 Positive Tokens:")
                    print(top_pos_words)

                    print("\nTop 20 Negative Tokens:")
                    print(top_neg_words)

                print("\n===== MISCLASSIFIED EXAMPLES =====")

                y_pred = model.predict(X_test_vec)

                misclassified = []

                for text, true_label, pred_label in zip(X_test, y_test, y_pred):
                    if true_label != pred_label:
                        misclassified.append((text, true_label, pred_label))

                print("\nTotal Misclassified:", len(misclassified))
                print("\nSample Misclassified Reviews:\n")

                for i in range(min(5, len(misclassified))):
                    print("Review:", misclassified[i][0][:300])
                    print("Actual:", misclassified[i][1])
                    print("Predicted:", misclassified[i][2])
                    print("-" * 80)

            # Save model
            os.makedirs("models", exist_ok=True)
            model_path = f"models/{feature}_{model_name}.pkl"
            joblib.dump(model, model_path)

          

            # Save vectorizer
            vectorizer_path = f"models/{feature}_{model_name}_vectorizer.pkl"
            joblib.dump(vectorizer, vectorizer_path)

            model_size = os.path.getsize(model_path) / (1024 * 1024)

            results.append({
            "Feature": feature,
            "Model": model_name,
            "Train_Accuracy": train_accuracy,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "ROC_AUC": metrics["roc_auc"],
            "CV_F1": cv_score,
            "Vocab_Size": vocab_size,
            "Sparsity": sparsity,
            "Training_Time": training_time,
            "Inference_Time": inference_time,
            "Model_Size_MB": model_size
})
    return pd.DataFrame(results)