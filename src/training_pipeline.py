import argparse
import json
import os
import time
import joblib

from .data_loader import load_data, split_data
from .feature_engineering import build_vectorizer, transform_text
from .model_factory import get_model
from .evaluation import evaluate_model
from .preprocessing import preprocess_series
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

LEADERBOARD_FILE = "models/leaderboard.json"


def train_lstm(x_train_text, y_train, x_test_text=None, y_test=None):
    vocab_size = 20000
    max_len = 200

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train_text)

    x_train_seq = tokenizer.texts_to_sequences(x_train_text)
    x_train_pad = pad_sequences(x_train_seq, maxlen=max_len)

    model = Sequential()
    model.add(Embedding(vocab_size, 128))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_pad, y_train, epochs=3, batch_size=64)

    os.makedirs("models", exist_ok=True)
    model.save("models/lstm.h5")

    with open("models/lstm_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    metrics = {}
    if x_test_text is not None and y_test is not None:
        x_test_seq = tokenizer.texts_to_sequences(x_test_text)
        x_test_pad = pad_sequences(x_test_seq, maxlen=max_len)
        metrics = evaluate_model(model, x_test_pad, y_test)
        save_accuracy("lstm", metrics.get("accuracy", 0.0))

    print("✅ LSTM saved")
    return metrics


def train_bilstm(x_train_text, y_train, x_test_text=None, y_test=None):
    vocab_size = 20000
    max_len = 200

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x_train_text)

    x_train_seq = tokenizer.texts_to_sequences(x_train_text)
    x_train_pad = pad_sequences(x_train_seq, maxlen=max_len)

    model = Sequential()
    model.add(Embedding(vocab_size, 128))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_pad, y_train, epochs=3, batch_size=64)

    os.makedirs("models", exist_ok=True)
    model.save("models/bilstm.h5")

    with open("models/bilstm_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    metrics = {}
    if x_test_text is not None and y_test is not None:
        x_test_seq = tokenizer.texts_to_sequences(x_test_text)
        x_test_pad = pad_sequences(x_test_seq, maxlen=max_len)
        metrics = evaluate_model(model, x_test_pad, y_test)
        save_accuracy("bilstm", metrics.get("accuracy", 0.0))

    print("✅ BiLSTM saved")
    return metrics


def train_pipeline(X_train, X_test, y_train, y_test, model_name):
    if model_name == "lstm":
        return train_lstm(X_train, y_train, X_test, y_test)

    if model_name == "bilstm":
        return train_bilstm(X_train, y_train, X_test, y_test)

    vectorizer = build_vectorizer("tfidf_uni_bi")
    X_train_vec, X_test_vec = transform_text(vectorizer, X_train, X_test)

    model = get_model(model_name)

    print("Training model...")
    start_time = time.time()
    model.fit(X_train_vec, y_train)
    training_time = time.time() - start_time

    print("Evaluating...")
    metrics = evaluate_model(model, X_test_vec, y_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{model_name}_model.pkl")
    joblib.dump(vectorizer, f"models/{model_name}_vectorizer.pkl")

    metrics["training_time"] = training_time
    save_accuracy(model_name, metrics["accuracy"])
    return metrics


def save_accuracy(model_name, accuracy):
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[model_name] = accuracy
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Train a sentiment model from src.training_pipeline")
    parser.add_argument(
        "--model",
        choices=["lstm", "bilstm", "logreg", "svm", "nb", "rf"],
        default="lstm",
        help="Model to train"
    )
    parser.add_argument(
        "--data-path",
        default="IMDB_Dataset.csv",
        help="Path to the IMDB dataset CSV"
    )
    args = parser.parse_args()

    df = load_data(args.data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train = preprocess_series(X_train)
    X_test = preprocess_series(X_test)

    print(f"Training model: {args.model}")
    result = train_pipeline(X_train, X_test, y_train, y_test, args.model)
    print("Finished training. Result:")
    print(result)


if __name__ == "__main__":
    main()