"""
Baseline models: Logistic Regression and Random Forest.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> LogisticRegression:
    """Fit a logistic regression model and return it."""
    y = np.asarray(y_train).ravel()

    params = {
        "max_iter": 1000,
        "solver": "lbfgs",
        "random_state": 42,
    }
    params.update(kwargs)

    model = LogisticRegression(**params)
    model.fit(X_train, y)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestClassifier:
    """Fit a random forest classifier and return it."""
    y = np.asarray(y_train).ravel()

    params = {
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(kwargs)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y)
    return model
