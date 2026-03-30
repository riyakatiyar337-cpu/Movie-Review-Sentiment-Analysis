from sklearn.model_selection import GridSearchCV

def tune_model(model, model_name, X_train, y_train):

    param_grids = {
        "logreg": {"C": [0.01, 0.1, 1, 10]},
        "svm": {"C": [ 0.1, 1]},
        "nb": {"alpha": [0.1, 0.5, 1.0]},
        "rf": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20]
        }
    }

    grid = GridSearchCV(
        model,
        param_grids[model_name],
        cv=5,
        scoring="f1",
        n_jobs=1,
        verbose=2
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_score_