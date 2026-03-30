import pandas as pd
from sklearn.model_selection import train_test_split
from .config import TEST_SIZE, RANDOM_STATE

def load_data(path):
    df = pd.read_csv(path)
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    
    # USE SMALL SUBSET FOR FASTER EXPERIMENTS
    df = df.sample(n=20000, random_state=42)
    
    return df

def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["sentiment"]
    )
    return X_train, X_test, y_train, y_test