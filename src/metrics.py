from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    *,
    y_proba: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Computes multiple binary classification metrics.
    Requires `y_proba` for ROC-AUC; otherwise skips it.
    """
    if y_proba is None:
        raise ValueError("y_proba is required for metrics (for ROC-AUC + thresholding).")

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # ROC-AUC requires both classes to be present.
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    return metrics

