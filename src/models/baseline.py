"""
Baseline models: Logistic Regression and Random Forest.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> LogisticRegression:
    """Fit a logistic regression model and return it."""
    raise NotImplementedError


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestClassifier:
    """Fit a random forest classifier and return it."""
    raise NotImplementedError
