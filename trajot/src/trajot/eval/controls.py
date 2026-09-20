from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .identification import accuracy_ci, flatten_features, identification_accuracy, pearson_scores


@dataclass(frozen=True)
class ControlResult:
    mean_accuracy: float
    per_pair_uncertainty: float | None
    flags: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


def _connectomes_from_timeseries(ts: np.ndarray, *, seed: int | None = None, shuffle: bool = False) -> np.ndarray:
    """``(S, V, T)`` time series -> ``(S, V, V)`` connectomes.

    ``shuffle=True`` permutes timepoints **independently per vertex**, which
    destroys cross-regional temporal coupling. A joint time permutation would
    leave Pearson FC unchanged, so it is not a valid N2.
    """
    arr = np.asarray(ts, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"timeseries must be (S, V, T), found {arr.shape}")
    S, V, T = arr.shape
    if T < 2:
        raise ValueError("need T >= 2 to form a connectome")
    rng = np.random.default_rng(seed) if seed is not None else None
    conns = np.empty((S, V, V), dtype=np.float64)
    for s in range(S):
        x = arr[s]
        if shuffle:
            if rng is None:
                raise ValueError("seed required when shuffle=True")
            x = np.stack([x[v, rng.permutation(T)] for v in range(V)], axis=0)
        c = np.corrcoef(x)
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        c = 0.5 * (c + c.T)
        np.fill_diagonal(c, 0.0)
        conns[s] = c
    return conns


def _looks_like_connectomes(x: np.ndarray) -> bool:
    return x.ndim == 3 and x.shape[1] == x.shape[2]


def _id_control_result(run1: np.ndarray, run2: np.ndarray, *, meta: dict[str, Any]) -> ControlResult:
    res = identification_accuracy(run1, run2, metric="pearson")
    ci_lo, ci_hi = accuracy_ci(res.correct_mask, n_boot=500, seed=0)
    uncertainty = float(0.5 * (ci_hi - ci_lo))
    return ControlResult(
        mean_accuracy=float(res.accuracy),
        per_pair_uncertainty=uncertainty,
        flags=np.asarray(res.correct_mask, dtype=bool),
        meta=meta,
    )


def random_permutation_control(run1: np.ndarray, run2: np.ndarray, *, seed: int) -> ControlResult:
    """Negative control: pair run1 with a random shuffle of run2 labels."""
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
        meta={"control": "random_permutation", "seed": int(seed)},
    )


def banded_coupling(
    run1: np.ndarray,
    run2: np.ndarray,
    *,
    band: int,
    seed: int,
    coords: np.ndarray | None = None,
) -> ControlResult:
    """N1: data-independent banded coupling control.

    Restricts correspondence to a spatial band ``|i - j| < band`` in region
    index space (or to spatial-neighbour ranks when ``coords`` is given).
    Residual identification gain after N1 is a smoothing artifact, not
    alignment.
    """
    if run1.shape != run2.shape:
        raise ValueError(f"run1 and run2 shapes must match, found {run1.shape} vs {run2.shape}")
    if band < 1:
        raise ValueError("band must be >= 1")

    arr1 = np.asarray(run1, dtype=np.float64)
    arr2 = np.asarray(run2, dtype=np.float64)
    if arr1.ndim != 3 or arr1.shape[1] != arr1.shape[2]:
        raise ValueError(f"connectomes must be (S,R,R), found {arr1.shape}")
    R = arr1.shape[1]

    if coords is not None:
        coords = np.asarray(coords, dtype=np.float64)
        if coords.shape[0] != R:
            raise ValueError(f"coords must have shape (R, D) with R={R}, found {coords.shape}")
        order = np.lexsort(coords.T[::-1])
        rank = np.empty(R, dtype=np.int64)
        rank[order] = np.arange(R)
        dist = np.abs(rank[:, None] - rank[None, :])
    else:
        dist = np.abs(np.subtract.outer(np.arange(R), np.arange(R)))

    mask = dist < band
    if not mask.any():
        raise ValueError("band mask is empty; increase band")

    # Data-independent banded view: only banded entries can match.
    masked1 = arr1 * mask
    masked2 = arr2 * mask
    return _id_control_result(
        masked1,
        masked2,
        meta={
            "control": "N1",
            "name": "banded_coupling",
            "band": int(band),
            "seed": int(seed),
            "coords": coords is not None,
        },
    )


def shuffle_time_control(
    timeseries: np.ndarray,
    run2: np.ndarray | None = None,
    *,
    seed: int,
) -> ControlResult:
    """N2: shuffled-time control — destroy temporal structure.

    When given ``(S, V, T)`` time series, independently permutes timepoints
    per subject, recomputes connectomes, and runs identification. Expected:
    accuracy collapses toward chance ``1/S``.

    When given connectomes only ``(S, R, R)``, falls back to shuffling
    vectorized features (weaker proxy); the limitation is recorded in meta.
    """
    ts1 = np.asarray(timeseries)
    ts2 = None if run2 is None else np.asarray(run2)

    if ts2 is None and ts1.ndim == 4 and ts1.shape[0] == 2:
        ts2 = ts1[1]
        ts1 = ts1[0]

    if ts2 is None:
        raise ValueError("shuffle_time_control requires run2 or stacked (2, S, ...) input")

    if ts1.shape != ts2.shape:
        raise ValueError(f"run shapes must match, found {ts1.shape} vs {ts2.shape}")

    if _looks_like_connectomes(ts1):
        # Weaker proxy: shuffle flattened feature vectors across subjects.
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed + 1)
        f1 = flatten_features(ts1)
        f2 = flatten_features(ts2)
        f1 = f1[rng1.permutation(f1.shape[0])]
        f2 = f2[rng2.permutation(f2.shape[0])]
        scores = pearson_scores(f1, f2)
        pred = np.argmax(scores, axis=1)
        truth = np.arange(scores.shape[0], dtype=np.int64)
        correct = pred == truth
        lo, hi = accuracy_ci(correct, n_boot=200, seed=seed)
        return ControlResult(
            mean_accuracy=float(np.mean(correct)),
            per_pair_uncertainty=float(0.5 * (hi - lo)),
            flags=correct,
            meta={
                "control": "N2",
                "name": "shuffle_time",
                "seed": int(seed),
                "input": "connectomes",
                "limitation": (
                    "timeseries unavailable; shuffled vectorized features "
                    "as a weaker temporal-structure proxy"
                ),
            },
        )

    # Timeseries path: shuffle timepoints, then recompute connectomes.
    shuffled1 = _connectomes_from_timeseries(ts1, seed=seed, shuffle=True)
    shuffled2 = _connectomes_from_timeseries(ts2, seed=int(seed) + 10007, shuffle=True)
    return _id_control_result(
        shuffled1,
        shuffled2,
        meta={
            "control": "N2",
            "name": "shuffle_time",
            "seed": int(seed),
            "input": "timeseries",
        },
    )
