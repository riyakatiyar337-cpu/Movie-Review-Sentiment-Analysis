from src.data_loader import load_data, split_data
from src.preprocessing import preprocess_series
from src.experiment_runner import run_full_experiment

DATA_PATH = "IMDB_Dataset.csv"

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Preprocessing...")
    X_train = preprocess_series(X_train)
    X_test = preprocess_series(X_test)

    print("\nRunning Full Experimental Study...")
    results_df = run_full_experiment(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n===== FINAL RESULTS =====")
    print(results_df)

    results_df.to_csv("final_results.csv", index=False)


if __name__ == "__main__":
    main() 

    