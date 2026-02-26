"""Evaluation utilities.

Computes:
- accuracy
- per-class F1
- macro F1
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _to_int_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        return np.argmax(y, axis=1).astype(np.int32)
    return y.astype(np.int32).reshape(-1)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_i = _to_int_labels(y_true)
    y_pred_i = _to_int_labels(y_pred)
    if y_true_i.shape != y_pred_i.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true_i.size == 0:
        return 0.0
    return float(np.mean(y_true_i == y_pred_i))


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    y_true_i = _to_int_labels(y_true)
    y_pred_i = _to_int_labels(y_pred)
    c = int(num_classes)
    cm = np.zeros((c, c), dtype=np.int64)
    for t, p in zip(y_true_i.tolist(), y_pred_i.tolist()):
        if 0 <= t < c and 0 <= p < c:
            cm[t, p] += 1
    return cm


def f1_scores(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> np.ndarray:
    cm = _confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = (np.sum(cm, axis=0) - tp).astype(np.float64)
    fn = (np.sum(cm, axis=1) - tp).astype(np.float64)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    return f1.astype(np.float32)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_map: Optional[Dict[str, int]] = None,
) -> Dict[str, object]:
    y_true_i = _to_int_labels(y_true)
    y_pred_i = _to_int_labels(y_pred)
    num_classes = int(max(int(np.max(y_true_i)) if y_true_i.size else 0, int(np.max(y_pred_i)) if y_pred_i.size else 0) + 1)

    acc = accuracy_score(y_true_i, y_pred_i)
    per_class = f1_scores(y_true_i, y_pred_i, num_classes=num_classes)
    macro_f1 = float(np.mean(per_class)) if per_class.size else 0.0

    if label_map:
        inv = {v: k for k, v in label_map.items()}
        per_class_f1 = {inv.get(i, str(i)): float(per_class[i]) for i in range(num_classes)}
    else:
        per_class_f1 = {str(i): float(per_class[i]) for i in range(num_classes)}

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
    }
