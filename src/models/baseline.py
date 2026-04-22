"""
Baseline models: Logistic Regression and Random Forest.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent


def _load_compositions() -> pd.DataFrame:
    data_file = PROJECT_ROOT / "data/raw/compositions_50k.csv"
    if not data_file.exists():
        data_file = PROJECT_ROOT / "src/data/compositions.csv"
    return pd.read_csv(data_file)


def build_baseline_dataset():
    """
    Build fixed-length tabular features for classical ML models.
    Feature vector = [blue_champion_multihot, red_champion_multihot].
    """
    df = _load_compositions()
    champions = sorted(df["champion_name"].unique())
    champ_to_idx = {name: idx for idx, name in enumerate(champions)}

    X_list = []
    y_list = []

    for _, group in df.groupby("match_id"):
        blue = group[group["team_id"] == 100]
        red = group[group["team_id"] == 200]
        if len(blue) != 5 or len(red) != 5:
            continue

        x = np.zeros(len(champions) * 2, dtype=np.float32)
        for name in blue["champion_name"].tolist():
            x[champ_to_idx[name]] = 1.0
        for name in red["champion_name"].tolist():
            x[len(champions) + champ_to_idx[name]] = 1.0

        X_list.append(x)
        y_list.append(1 if bool(blue["win"].iloc[0]) else 0)

    return np.array(X_list), np.array(y_list), champions


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> LogisticRegression:
    """Fit a logistic regression model and return it."""
    params = {
        "max_iter": 1000,
        "n_jobs": -1,
        "random_state": 42,
    }
    params.update(kwargs)
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestClassifier:
    """Fit a random forest classifier and return it."""
    params = {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": 42,
    }
    params.update(kwargs)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def benchmark_baselines(test_size: float = 0.2, random_state: int = 42):
    """Train and evaluate Logistic Regression + RandomForest on same split."""
    X, y, _ = build_baseline_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results = {}

    lr_model = train_logistic_regression(X_train, y_train)
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    results["logistic_regression"] = {
        "accuracy": float(accuracy_score(y_test, lr_pred)),
        "f1": float(f1_score(y_test, lr_pred)),
        "auc": float(roc_auc_score(y_test, lr_prob)),
    }

    rf_model = train_random_forest(X_train, y_train)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)
    results["random_forest"] = {
        "accuracy": float(accuracy_score(y_test, rf_pred)),
        "f1": float(f1_score(y_test, rf_pred)),
        "auc": float(roc_auc_score(y_test, rf_prob)),
    }

    return results
