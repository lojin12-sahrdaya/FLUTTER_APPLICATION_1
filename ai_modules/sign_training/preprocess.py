"""Preprocessing for ASL Alphabet landmark vectors.

Inputs:
- X: (N, 63) raw MediaPipe hand landmark vectors (x,y,z)
- y: (N,) integer labels

Outputs:
- X_train, X_test, y_train, y_test
    where y_* are one-hot (categorical) label arrays.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np



def normalize_landmarks_vector(x: np.ndarray) -> np.ndarray:
    """Normalize a single (63,) landmark vector.

    - Re-centers on wrist (landmark 0)
    - Scales by max 2D distance in x/y
    """

    x = np.asarray(x, dtype=np.float32)
    if x.shape != (63,):
        raise ValueError(f"Expected shape (63,), got {x.shape}")

    pts = x.reshape(21, 3)
    pts = pts - pts[0:1]
    scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
    if scale > 0:
        pts = pts / scale
    return pts.reshape(63).astype(np.float32, copy=False)


def normalize_X(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 63:
        raise ValueError(f"Expected X shape (N,63), got {X.shape}")
    out = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        out[i] = normalize_landmarks_vector(X[i])
    return out


def to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    n = y.shape[0]
    c = int(num_classes)
    out = np.zeros((n, c), dtype=np.float32)
    out[np.arange(n), y] = 1.0
    return out


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)

    train_idx: List[int] = []
    test_idx: List[int] = []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        idx = idx.copy()
        rng.shuffle(idx)
        n = idx.shape[0]
        if n < 2:
            train_idx.extend(idx.tolist())
            continue
        n_test = int(round(n * test_ratio))
        n_test = max(1, min(n - 1, n_test))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    train_idx = np.array(train_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def preprocess(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = normalize_X(X)
    X_train, X_test, y_train_int, y_test_int = stratified_train_test_split(X, y, test_ratio=test_ratio, seed=seed)
    num_classes = int(np.max(y) + 1)
    y_train = to_categorical(y_train_int, num_classes)
    y_test = to_categorical(y_test_int, num_classes)
    return X_train.astype(np.float32), X_test.astype(np.float32), y_train.astype(np.float32), y_test.astype(np.float32)
