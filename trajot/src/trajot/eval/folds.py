from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

FOLD_PROCEDURE = "default_rng.choice without replacement over lexicographic pair index"


@dataclass(frozen=True)
class FoldAssignment:
    pairs: list[tuple[str, str]]
    seed: int
    n_pairs: int
    procedure: str = FOLD_PROCEDURE


def _resolve_subjects(subjects_or_manifest: Any) -> list[str]:
    if subjects_or_manifest is None:
        raise ValueError("subjects_or_manifest must not be None")

    if isinstance(subjects_or_manifest, (list, tuple, set, np.ndarray)):
        return sorted(str(s) for s in subjects_or_manifest)

    # pandas DataFrame manifest
    if hasattr(subjects_or_manifest, "columns") and hasattr(subjects_or_manifest, "groupby"):
        try:
            from trajot.io.contract import two_run_subjects

            subs = two_run_subjects(subjects_or_manifest, strict=False)
            if subs:
                return subs
        except Exception:
            pass
        if "subject_id" in subjects_or_manifest.columns:
            return sorted({str(s) for s in subjects_or_manifest["subject_id"].tolist()})
        raise ValueError("manifest has no subject_id column and no two-run subjects")

    raise TypeError(f"unsupported subjects_or_manifest type: {type(subjects_or_manifest)!r}")


def all_ordered_pairs(subjects: Sequence[str]) -> list[tuple[str, str]]:
    """All ordered distinct pairs (i, j), i != j, lexicographic by subject id."""
    subj = sorted(str(s) for s in subjects)
    return [(a, b) for a in subj for b in subj if a != b]


def make_folds(
    subjects_or_manifest: Any,
    scheme: str = "pairs_without_replacement",
    n_pairs: int = 500,
    seed: int = 0,
) -> FoldAssignment:
    """Declare a pair subsample without replacement.

    Pool is all ordered distinct pairs over lexicographic subject ids.
    Draws use ``numpy.random.default_rng(seed).choice`` without replacement.
    """
    if n_pairs < 1:
        raise ValueError("n_pairs must be >= 1")
    subjects = _resolve_subjects(subjects_or_manifest)
    if len(subjects) < 2:
        raise ValueError("need at least two subjects to form pairs")

    pool = all_ordered_pairs(subjects)
    if n_pairs > len(pool):
        raise ValueError(
            f"n_pairs={n_pairs} exceeds available ordered pairs {len(pool)}; "
            "sampling is without replacement"
        )

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n_pairs, replace=False)
    pairs = [pool[int(i)] for i in idx]
    return FoldAssignment(
        pairs=pairs,
        seed=int(seed),
        n_pairs=int(n_pairs),
        procedure=FOLD_PROCEDURE,
    )


def identification_subjects(manifest: Any) -> list[str]:
    """Fixed two-run identification subject list via ``io.two_run_subjects``."""
    if isinstance(manifest, (list, tuple, set, np.ndarray)):
        return sorted(str(s) for s in manifest)
    from trajot.io.contract import two_run_subjects

    return two_run_subjects(manifest, strict=False)


def sample_pairs(subjects: Sequence[str], n_pairs: int, seed: int) -> list[tuple[str, str]]:
    """Sample ordered distinct subject pairs without replacement when possible."""
    return make_folds(
        list(subjects),
        scheme="pairs_without_replacement",
        n_pairs=n_pairs,
        seed=seed,
    ).pairs


def make_two_run_splits(subjects: Sequence[str]) -> tuple[list[str], list[str]]:
    """For two-run identifiability, both folds contain the same subject list."""
    ordered = sorted(str(s) for s in subjects)
    if len(ordered) < 2:
        raise ValueError("need at least two subjects")
    return ordered, ordered.copy()
