"""
Evaluation metrics: accuracy, precision, recall, F1, AUC-ROC,
confusion matrix, and statistical significance test vs the 50% baseline.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """
    Compute all metrics and return as a dict:
    {accuracy, precision, recall, f1, auc_roc, confusion_matrix}
    """
    raise NotImplementedError


def significance_test(y_true: np.ndarray, y_pred: np.ndarray, baseline: float = 0.5) -> dict:
    """Binomial test: is model accuracy significantly better than random (50%)?"""
    raise NotImplementedError


def print_report(results: dict) -> None:
    """Pretty-print evaluation results."""
    raise NotImplementedError
