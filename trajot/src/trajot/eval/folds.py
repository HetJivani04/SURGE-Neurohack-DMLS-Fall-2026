from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def sample_pairs(subjects: Sequence[str], n_pairs: int, seed: int) -> list[tuple[str, str]]:
    """Sample ordered distinct subject pairs with replacement, reproducibly."""
    subj = [str(s) for s in subjects]
    if len(subj) < 2:
        raise ValueError("need at least two subjects to sample pairs")
    if n_pairs < 1:
        raise ValueError("n_pairs must be >= 1")

    rng = np.random.default_rng(seed)
    out: list[tuple[str, str]] = []
    while len(out) < n_pairs:
        i = int(rng.integers(0, len(subj)))
        j = int(rng.integers(0, len(subj)))
        if i == j:
            continue
        out.append((subj[i], subj[j]))
    return out


def make_two_run_splits(subjects: Sequence[str]) -> tuple[list[str], list[str]]:
    """For two-run identifiability, both folds contain the same subject list.

    Run-1 serves as query against run-2 gallery and vice versa.
    """
    ordered = sorted(str(s) for s in subjects)
    if len(ordered) < 2:
        raise ValueError("need at least two subjects")
    return ordered, ordered.copy()
