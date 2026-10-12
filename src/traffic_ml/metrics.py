"""Accuracy, rare-event, probabilistic and calibration metrics."""
from __future__ import annotations

from itertools import pairwise

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    confidence, prediction = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.
    for lo, hi in pairwise(edges):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs((prediction[mask] == y[mask]).mean() - confidence[mask].mean())
    return float(ece)


def adaptive_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    confidence, correct = p.max(1), p.argmax(1) == y
    chunks = np.array_split(np.argsort(confidence), bins)
    return float(np.mean([abs(correct[i].mean() - confidence[i].mean()) for i in chunks if len(i)]))


def multiclass_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1)))


def evaluate_probabilities(y: np.ndarray, p: np.ndarray, severe: int = 2, bins: int = 15) -> dict[str, float]:
    pred = p.argmax(1)
    severe_y = (y == severe).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(severe_y, pred == severe,
                                                               average="binary", zero_division=0)
    result = {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
              "macro_f1": f1_score(y, pred, average="macro"), "severe_precision": precision,
              "severe_recall": recall, "severe_f1": f1, "nll": log_loss(y, p, labels=range(p.shape[1])),
              "brier": multiclass_brier(y, p), "ece": expected_calibration_error(y, p, bins),
              "ace": adaptive_calibration_error(y, p, bins),
              "severe_auprc": average_precision_score(severe_y, p[:, severe])}
    try:
        result["severe_auroc"] = roc_auc_score(severe_y, p[:, severe])
    except ValueError:
        result["severe_auroc"] = float("nan")
    return {k: float(v) for k, v in result.items()}
