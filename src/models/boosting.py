"""
Gradient-boosted models: XGBoost and LightGBM.
"""

import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> XGBClassifier:
    """Fit an XGBoost classifier and return it."""
    raise NotImplementedError


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> LGBMClassifier:
    """Fit a LightGBM classifier and return it."""
    raise NotImplementedError
