from src.data_loader import load_data, split_data
from src.preprocessing import preprocess_series
from src.training_pipeline import train_lstm

DATA_PATH = "IMDB_Dataset.csv"


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Preprocessing training data...")
    X_train = preprocess_series(X_train)
    X_test = preprocess_series(X_test)

    print("Training LSTM model...")
    train_lstm(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
