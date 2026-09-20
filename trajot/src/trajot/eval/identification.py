from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IdentificationResult:
    accuracy: float
    correct_mask: np.ndarray
    predictions: np.ndarray
    score_matrix: np.ndarray
    metric: str = "pearson"


def flatten_features(x: np.ndarray) -> np.ndarray:
    """Flatten subject features for identification.

    - ``(S, R, R)`` square connectomes: upper triangle ``k=1`` -> ``(S, R*(R-1)//2)``
    - ``(S, R, d)`` non-square features: reshape -> ``(S, R*d)``
    - ``(S, F)`` already flat: cast to float64
    """
    arr = np.asarray(x)
    if arr.ndim == 2:
        return np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"features must be 2D or 3D, found shape {arr.shape}")
    s, a, b = arr.shape
    if a == b:
        idx = np.triu_indices(a, k=1)
        return np.ascontiguousarray(arr[:, idx[0], idx[1]], dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(s, a * b), dtype=np.float64)


def pearson_scores(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation score matrix.

    ``scores[i, j] = corr(query[i], gallery[j])``
    """
    q = flatten_features(query)
    g = flatten_features(gallery)
    if q.shape[1] != g.shape[1]:
        raise ValueError(f"feature dim mismatch: {q.shape} vs {g.shape}")
    q_c = q - q.mean(axis=1, keepdims=True)
    g_c = g - g.mean(axis=1, keepdims=True)
    q_n = np.linalg.norm(q_c, axis=1, keepdims=True)
    g_n = np.linalg.norm(g_c, axis=1, keepdims=True)
    denom = np.maximum(q_n * g_n.T, 1e-12)
    return (q_c @ g_c.T) / denom


def _cosine_scores(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    q = flatten_features(query)
    g = flatten_features(gallery)
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    gn = np.linalg.norm(g, axis=1, keepdims=True)
    denom = np.maximum(qn * gn.T, 1e-12)
    return (q @ g.T) / denom


def _score_matrix(query: np.ndarray, gallery: np.ndarray, metric: str) -> np.ndarray:
    key = metric.lower()
    if key in {"pearson", "correlation", "corr"}:
        return pearson_scores(query, gallery)
    if key in {"cosine", "cos"}:
        return _cosine_scores(query, gallery)
    raise ValueError(f"unknown identification metric: {metric!r}")


def identification_accuracy(
    run1: np.ndarray,
    run2: np.ndarray,
    *,
    metric: str = "pearson",
) -> IdentificationResult:
    """Top-1 identification of run1 subjects in run2.

    Spec metric is Pearson correlation on flattened upper-triangle features.
    Cosine remains available as a secondary metric.
    """
    q = flatten_features(run1)
    g = flatten_features(run2)
    if q.shape != g.shape:
        raise ValueError(
            f"run1 and run2 must match shape after vectorization: {q.shape} vs {g.shape}"
        )
    scores = _score_matrix(run1, run2, metric)
    pred = np.argmax(scores, axis=1)
    truth = np.arange(scores.shape[0], dtype=np.int64)
    correct = pred == truth
    acc = float(np.mean(correct))
    return IdentificationResult(
        accuracy=acc,
        correct_mask=correct,
        predictions=pred,
        score_matrix=scores,
        metric=metric.lower(),
    )


def identify_subjects(run1: np.ndarray, run2: np.ndarray) -> IdentificationResult:
    """Backward-compatible entry point; Pearson identification."""
    return identification_accuracy(run1, run2, metric="pearson")


def accuracy_ci(
    correct_mask: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for top-1 identification accuracy."""
    correct = np.asarray(correct_mask, dtype=np.float64).ravel()
    n = correct.size
    if n == 0:
        return (0.0, 0.0)
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    rng = np.random.default_rng(seed)
    # Chunked bootstrap keeps memory bounded for large n_boot
    chunk = max(1, min(n_boot, 2000))
    means = np.empty(n_boot, dtype=np.float64)
    filled = 0
    while filled < n_boot:
        m = min(chunk, n_boot - filled)
        idx = rng.integers(0, n, size=(m, n))
        means[filled : filled + m] = correct[idx].mean(axis=1)
        filled += m
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(max(0.0, lo)), float(min(1.0, hi)))
