"""Reusable evaluation metrics for ECDSS-Heart.

Implements standard classification metrics calculated from actual
model predictions on the test set.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)


def calculate_accuracy(y_true, y_pred) -> float:
    """Calculate accuracy score."""
    return float(accuracy_score(y_true, y_pred))


def calculate_precision(y_true, y_pred, average: str = "binary") -> float:
    """Calculate precision score."""
    return float(precision_score(y_true, y_pred, average=average, zero_division=0))


def calculate_recall(y_true, y_pred, average: str = "binary") -> float:
    """Calculate recall score."""
    return float(recall_score(y_true, y_pred, average=average, zero_division=0))


def calculate_f1(y_true, y_pred, average: str = "binary") -> float:
    """Calculate F1-score."""
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def calculate_roc_auc(y_true, y_prob) -> float:
    """Calculate ROC-AUC from true labels and predicted probabilities for positive class."""
    return float(roc_auc_score(y_true, y_prob))


def calculate_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Calculate confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def calculate_classification_report(
    y_true, y_pred, labels=None, output_dict: bool = False,
) -> Dict[str, Any]:
    """Calculate classification report."""
    return classification_report(
        y_true, y_pred, labels=labels, output_dict=output_dict, zero_division=0
    )


def calculate_roc_curve(y_true, y_prob) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate ROC curve coordinates."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds


def calculate_auc_from_curve(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Calculate AUC from FPR/TPR arrays."""
    return float(auc(fpr, tpr))