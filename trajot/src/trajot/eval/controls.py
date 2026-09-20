from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlResult:
    mean_accuracy: float
    per_pair_uncertainty: float
    flags: np.ndarray


def random_permutation_control(run1: np.ndarray, run2: np.ndarray, *, seed: int) -> ControlResult:
    """A weak negative control that pairs run1 with a random shuffle of run2."""
    if run1.shape != run2.shape:
        raise ValueError(f"run1 and run2 shapes must match, found {run1.shape} vs {run2.shape}")

    n = run1.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    is_correct = perm == np.arange(n, dtype=np.int64)
    accuracy = float(np.mean(is_correct))
    uncertainty = float(1.0 - np.mean(np.abs(is_correct.astype(np.float64) - 0.5) * 2.0))

    return ControlResult(
        mean_accuracy=accuracy,
        per_pair_uncertainty=uncertainty,
        flags=is_correct,
    )
