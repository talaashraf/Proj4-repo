from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def tune_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)

    parameters = {
        "C": [0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Hyperparameters:")
    print(grid_search.best_params_)

    print("\nBest Cross-Validation F1-score:")
    print(grid_search.best_score_)

    return grid_search.best_estimator_