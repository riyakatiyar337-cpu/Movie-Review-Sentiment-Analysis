from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from fastapi.middleware.cors import CORSMiddleware
from src.preprocessing import preprocess_series

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("models/tfidf_uni_bi_svm.pkl")
vectorizer = joblib.load("models/tfidf_uni_bi_svm_vectorizer.pkl")


class RequestData(BaseModel):
    text: str
    model_name: str = "svm"

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: RequestData):
     try:
        processed = preprocess_series(pd.Series([data.text]))

        vectorized = vectorizer.transform(processed)

        pred = model.predict(vectorized)[0]

        return {
            "sentiment": "Positive" if pred == 1 else "Negative",
            "model": data.model_name
        }

     except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}
    