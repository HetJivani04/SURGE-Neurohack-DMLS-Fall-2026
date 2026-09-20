from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IdentificationResult:
    accuracy: float
    correct_mask: np.ndarray
    predictions: np.ndarray
    score_matrix: np.ndarray


def _vectorize_connectomes(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"connectomes must be (S,R,R), found {arr.shape}")
    idx = np.triu_indices(arr.shape[1], k=1)
    return arr[:, idx[0], idx[1]]


def _cosine_scores(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    qn = np.linalg.norm(query, axis=1, keepdims=True)
    gn = np.linalg.norm(gallery, axis=1, keepdims=True)
    denom = np.maximum(qn * gn.T, 1e-12)
    return (query @ gallery.T) / denom


def identify_subjects(run1: np.ndarray, run2: np.ndarray) -> IdentificationResult:
    """Top-1 identification of run1 subjects in run2 using cosine similarity."""
    q = _vectorize_connectomes(run1)
    g = _vectorize_connectomes(run2)
    if q.shape != g.shape:
        raise ValueError(f"run1 and run2 must match shape after vectorization: {q.shape} vs {g.shape}")

    scores = _cosine_scores(q, g)
    pred = np.argmax(scores, axis=1)
    truth = np.arange(scores.shape[0], dtype=np.int64)
    correct = pred == truth
    acc = float(np.mean(correct))

    return IdentificationResult(
        accuracy=acc,
        correct_mask=correct,
        predictions=pred,
        score_matrix=scores,
    )
