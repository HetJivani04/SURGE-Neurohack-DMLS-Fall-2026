from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .identification import identify_subjects


@dataclass(frozen=True)
class PermutationResult:
    p_value: float
    null_distribution: np.ndarray
    observed: float


def permutation_p(
    run1: np.ndarray,
    run2: np.ndarray,
    *,
    B: int,
    seed: int,
) -> PermutationResult:
    """One-sided permutation p-value for identification accuracy.

    Uses the exact finite-sample estimate (count + 1) / (B + 1).
    """
    if B < 1:
        raise ValueError("B must be >= 1")

    obs = identify_subjects(run1, run2).accuracy
    rng = np.random.default_rng(seed)
    null = np.empty(B, dtype=np.float64)

    for b in range(B):
        perm = rng.permutation(run2.shape[0])
        null[b] = identify_subjects(run1, run2[perm]).accuracy

    ge = int(np.sum(null >= obs))
    p = float((ge + 1) / (B + 1))
    return PermutationResult(p_value=p, null_distribution=null, observed=obs)
