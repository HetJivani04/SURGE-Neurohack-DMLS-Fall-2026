from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .identification import flatten_features


def _row_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = float(np.linalg.norm(a_c) * np.linalg.norm(b_c))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_c, b_c) / denom)


def _as_index_pairs(pairs_idx: Sequence | np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(pairs_idx)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return [(int(i), int(j)) for i, j in arr]
    if isinstance(pairs_idx, (list, tuple)):
        out: list[tuple[int, int]] = []
        for item in pairs_idx:
            if isinstance(item, (list, tuple, np.ndarray)) and len(item) == 2:
                out.append((int(item[0]), int(item[1])))
            else:
                raise TypeError(f"pairs_idx entries must be (i, j) index pairs, got {item!r}")
        return out
    raise TypeError(f"pairs_idx must be (P,2) or list of pairs, got type {type(pairs_idx)!r}")


def pair_gain(
    aligned1: np.ndarray,
    aligned2: np.ndarray,
    raw1: np.ndarray,
    raw2: np.ndarray,
    pairs_idx: Sequence | np.ndarray,
) -> np.ndarray:
    """Per-pair held-out correlation gain.

    For each declared pair ``(a, b)``::

        gain = corr(aligned_a_r1, aligned_b_r2) - corr(raw_a_r1, raw_b_r2)

    Self-pairs ``(s, s)`` implement the two-run within-subject form.
    """
    a1 = flatten_features(aligned1)
    a2 = flatten_features(aligned2)
    r1 = flatten_features(raw1)
    r2 = flatten_features(raw2)
    if not (a1.shape == a2.shape == r1.shape == r2.shape):
        raise ValueError(
            "aligned/raw feature shapes must match after flattening: "
            f"{a1.shape}, {a2.shape}, {r1.shape}, {r2.shape}"
        )
    pairs = _as_index_pairs(pairs_idx)
    n = a1.shape[0]
    gains = np.empty(len(pairs), dtype=np.float64)
    for k, (i, j) in enumerate(pairs):
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError(f"pair ({i}, {j}) out of range for S={n}")
        gains[k] = _row_corr(a1[i], a2[j]) - _row_corr(r1[i], r2[j])
    return gains


def alignment_gain(
    aligned1: np.ndarray,
    aligned2: np.ndarray,
    raw1: np.ndarray,
    raw2: np.ndarray,
    pairs_idx: Sequence | np.ndarray,
) -> float:
    """Mean held-out correlation gain over the declared pair subsample."""
    gains = pair_gain(aligned1, aligned2, raw1, raw2, pairs_idx)
    if gains.size == 0:
        raise ValueError("pairs_idx must contain at least one pair")
    return float(np.mean(gains))
