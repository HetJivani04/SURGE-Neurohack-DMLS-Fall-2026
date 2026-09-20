from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .identification import identify_subjects


@dataclass(frozen=True)
class PermutationResult:
    p_value: float
    null_distribution: np.ndarray
    observed: float
    null_max: float


def _p_from_null(observed: float, null: np.ndarray) -> float:
    ge = int(np.sum(null >= observed))
    return float((ge + 1) / (null.size + 1))


def null_max(result: PermutationResult | np.ndarray | Sequence[float]) -> float:
    """Maximum of a permutation null distribution (Finn-style comparison)."""
    if isinstance(result, PermutationResult):
        if result.null_distribution.size == 0:
            return result.null_max
        return float(np.max(result.null_distribution))
    arr = np.asarray(result, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("empty null distribution")
    return float(np.max(arr))


def permutation_null(
    stat_fn: Callable[[Any], float],
    items: Any,
    *,
    B: int,
    seed: int,
) -> PermutationResult:
    """Generic permutation null.

    ``stat_fn`` maps a permutation of ``items`` to a scalar statistic.
    ``items`` is an ndarray (permuted along axis 0) or a list/tuple
    (permuted by index). The null uses label/pairing shuffles; ``p`` is the
    finite-sample one-sided estimate ``(1 + #{null >= obs}) / (1 + B)``.
    """
    if B < 1:
        raise ValueError("B must be >= 1")
    rng = np.random.default_rng(seed)
    observed = float(stat_fn(items))
    null = np.empty(B, dtype=np.float64)

    if isinstance(items, np.ndarray):
        n = items.shape[0]
        if n < 2:
            raise ValueError("items must have length >= 2 to permute")
        for b in range(B):
            perm = rng.permutation(n)
            null[b] = float(stat_fn(items[perm]))
    elif isinstance(items, (list, tuple)):
        n = len(items)
        if n < 2:
            raise ValueError("items must have length >= 2 to permute")
        for b in range(B):
            perm = rng.permutation(n)
            shuffled = [items[int(i)] for i in perm]
            null[b] = float(stat_fn(shuffled))
    else:
        raise TypeError(
            f"items must be ndarray or list/tuple for permutation, got {type(items)!r}"
        )

    return PermutationResult(
        p_value=_p_from_null(observed, null),
        null_distribution=null,
        observed=observed,
        null_max=float(np.max(null)),
    )


def sign_flip_permutation(
    values: np.ndarray,
    *,
    B: int,
    seed: int,
) -> np.ndarray:
    """Paired sign-flip null of the mean (Winkler et al. 2014).

    Returns shape ``(B,)`` — the null distribution of mean(sign * values).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        raise ValueError("values must be non-empty")
    if B < 1:
        raise ValueError("B must be >= 1")
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(B, v.size))
    return (signs * v).mean(axis=1)


def permutation_p(
    run1: np.ndarray,
    run2: np.ndarray,
    *,
    B: int,
    seed: int,
) -> PermutationResult:
    """One-sided permutation p-value for identification accuracy.

    Uses the exact finite-sample estimate ``(count + 1) / (B + 1)``.
    ``null_max`` is reported for Finn-style comparison.
    """
    if B < 1:
        raise ValueError("B must be >= 1")

    obs = identify_subjects(run1, run2).accuracy
    rng = np.random.default_rng(seed)
    null = np.empty(B, dtype=np.float64)

    for b in range(B):
        perm = rng.permutation(run2.shape[0])
        null[b] = identify_subjects(run1, run2[perm]).accuracy

    return PermutationResult(
        p_value=_p_from_null(obs, null),
        null_distribution=null,
        observed=obs,
        null_max=float(np.max(null)),
    )
