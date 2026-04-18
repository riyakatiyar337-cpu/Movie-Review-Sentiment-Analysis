from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pickle
import os
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocessing import preprocess_series

app = FastAPI()

# ==============================
# 🌐 CORS (Frontend Connection)
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 🔧 CONFIG
# ==============================
MODEL_PATH = "models/"
MAX_LEN = 200

# ==============================
# 📥 REQUEST SCHEMA
# ==============================
class RequestData(BaseModel):
    text: str
    model_name: str = "svm"


# ==============================
# 📦 LOAD MODEL FUNCTION
# ==============================
def load_model_and_processor(model_name):

    # 🔥 Deep Learning Models
    if model_name == "lstm":
        model = load_model(os.path.join(MODEL_PATH, "lstm.h5"))
        tokenizer = pickle.load(open(os.path.join(MODEL_PATH, "lstm_tokenizer.pkl"), "rb"))
        return model, tokenizer

    elif model_name == "bilstm":
        model = load_model(os.path.join(MODEL_PATH, "bilstm.h5"))
        tokenizer = pickle.load(open(os.path.join(MODEL_PATH, "bilstm_tokenizer.pkl"), "rb"))
        return model, tokenizer

    # 🔥 Classical ML Models
    else:
        model_file = os.path.join(MODEL_PATH, f"tfidf_uni_bi_{model_name}.pkl")
        vectorizer_file = os.path.join(MODEL_PATH, f"tfidf_uni_bi_{model_name}_vectorizer.pkl")

        if not os.path.exists(model_file):
            raise Exception(f"Model {model_name} not found")

        model = joblib.load(model_file)
        vectorizer = joblib.load(vectorizer_file)

        return model, vectorizer


# ==============================
# 🔮 PREDICTION FUNCTION
# ==============================
def predict_text(model, processor, text, model_name):

    # 🔥 Preprocess text (common)
    processed = preprocess_series(pd.Series([text]))

    # 🔥 LSTM / BiLSTM
    if model_name in ["lstm", "bilstm"]:
        seq = processor.texts_to_sequences(processed)
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        prob = model.predict(padded)[0][0]
        sentiment = "Positive" if prob > 0.5 else "Negative"

        return sentiment, float(prob)

    # 🔥 ML Models
    else:
        vec = processor.transform(processed)
        pred = model.predict(vec)[0]

        sentiment = "Positive" if pred == 1 else "Negative"

        confidence = None

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(vec)[0]
            confidence = float(max(prob))

        elif hasattr(model, "decision_function"):
            score = model.decision_function(vec)[0]
            confidence = float(1 / (1 + pow(2.718, -score)))

        return sentiment, confidence


# ==============================
# 🌐 ROUTES
# ==============================
@app.get("/")
def home():
    return {"message": "API is running 🚀"}


@app.post("/predict")
def predict(data: RequestData):

    try:
        model, processor = load_model_and_processor(data.model_name)

        sentiment, confidence = predict_text(
            model,
            processor,
            data.text,
            data.model_name
        )

        return {
            "text": data.text,
            "model": data.model_name,
            "sentiment": sentiment,
            "confidence": confidence
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}



import json

@app.get("/leaderboard")
def get_leaderboard():
    try:
        with open("models/leaderboard.json", "r") as f:
            data = json.load(f)

        # Sort by accuracy
        sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))

        return sorted_data

    except:
        return {"error": "Leaderboard not found"}