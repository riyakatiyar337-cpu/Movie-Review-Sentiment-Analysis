import time
import joblib
import os

from .feature_engineering import build_vectorizer, transform_text
from .model_factory import get_model
from .evaluation import evaluate_model

def train_pipeline(X_train, X_test, y_train, y_test, model_name):

    vectorizer = build_vectorizer()
    X_train_vec, X_test_vec = transform_text(vectorizer, X_train, X_test)

    model = get_model(model_name)

    print("Training model...")
    start_time = time.time()
    model.fit(X_train_vec, y_train)
    training_time = time.time() - start_time

    print("Evaluating...")
    metrics = evaluate_model(model, X_test_vec, y_test)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{model_name}_model.pkl")
    joblib.dump(vectorizer, f"models/{model_name}_vectorizer.pkl")

    metrics["training_time"] = training_time

    return metrics