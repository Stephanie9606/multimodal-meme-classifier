"""Evaluation metrics: accuracy, F1, per-class precision/recall."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Return a dict with accuracy, macro-F1, report dict, and confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim > 1 and y_true.shape[-1] > 1:
        y_true = y_true.argmax(axis=-1)
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        y_pred = y_pred.argmax(axis=-1)
    else:
        y_pred = (y_pred.squeeze() >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
