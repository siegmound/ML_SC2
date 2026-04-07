from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score


def classification_summary(y_true, y_pred, y_prob: Optional[np.ndarray] = None) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_rows": int(len(y_true)),
    }
    if y_prob is not None:
        summary["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        summary["log_loss"] = float(log_loss(y_true, y_prob))
    return summary
