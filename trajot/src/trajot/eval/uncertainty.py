"""Gap-filling uncertainty metrics for hierarchical posterior couplings.

Scientific mandate
------------------
These diagnostics score a *posterior* over alignments — calibrated coverage,
non-identifiability (AUROC of τ), held-out predictive residual, and subject
mean τ. They exist to evaluate TrajOT's hierarchical population-of-couplings
where baselines emit point estimates only.

They are **necessary but not sufficient** for the literature-gap claim. A
posterior that is well-calibrated but recovers alignment poorly (bad held-out
score / recovery vs planted or run-2 structure) does **not** succeed. Models
must win alignment quality *and* fill uncertainty columns baselines lack.
Never treat coverage/AUROC as a substitute for recovery metrics.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "UNCERTAINTY_KEYS",
    "heldout_predictive_score",
    "nonident_auroc",
    "posterior_coverage",
    "shrinkage_transform",
    "subject_mean_tau",
]

UNCERTAINTY_KEYS = (
    "posterior_coverage",
    "nonident_auroc",
    "heldout_score",
    "group_neff",
)


def shrinkage_transform(C: np.ndarray, Q: np.ndarray, lam: float) -> np.ndarray:
    r"""Posterior-gated shrinkage toward an orthogonal gauge.

    C̃ = (1-λ) C + λ Q^T C Q, with λ ∈ [0, 1].

    λ=0 leaves the observed connectome unchanged; λ=1 applies the full
    orthogonal map Q. Intermediate λ shrinks toward the population-template
    gauge when subject-level τ is large (non-identifiable).

    Used inside OursFull only after alignment recovery is measured — the
    shrinkage must not degrade held-out structure.
    """
    C = np.asarray(C, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be square 2D, got shape {C.shape}")
    if Q.shape != C.shape:
        raise ValueError(f"Q shape {Q.shape} must match C shape {C.shape}")
    mapped = Q.T @ C @ Q
    return (1.0 - lam) * C + lam * mapped


def posterior_coverage(
    pi_samples: np.ndarray,
    pi_star: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Empirical (1-α) coverage of planted ground-truth couplings.

    Fraction of coordinates (i, k) where ``pi_star[i, k]`` falls inside the
    empirical quantile interval of the posterior draws at α/2 and 1-α/2.

    ``pi_samples`` is (M, R, R) or (M, V, K); ``pi_star`` shares the trailing
    shape. Coverage near the nominal level is evidence of calibration — it
    does **not** by itself show the posterior recovers alignment quality;
    pair with held-out / recovery scores.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    draws = np.asarray(pi_samples, dtype=np.float64)
    star = np.asarray(pi_star, dtype=np.float64)
    if draws.ndim < 2:
        raise ValueError(f"pi_samples must be at least 2D (M, ...), got {draws.shape}")
    if draws.shape[1:] != star.shape:
        raise ValueError(
            f"pi_star trailing shape {star.shape} must match pi_samples {draws.shape[1:]}"
        )
    if draws.shape[0] == 0:
        raise ValueError("pi_samples must contain at least one draw")
    lo = np.quantile(draws, alpha / 2.0, axis=0)
    hi = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
    inside = (star >= lo) & (star <= hi)
    return float(np.mean(inside))


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    """1-based average ranks (ties get mean rank)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = scores.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg = 0.5 * (i + 1 + j)  # 1-based ranks i+1 .. j
        ranks[order[i:j]] = avg
        i = j
    return ranks


def nonident_auroc(
    tau_phi: Sequence[np.ndarray] | np.ndarray,
    ambiguous: np.ndarray,
) -> float:
    """Rank AUROC of subject-level mean(τ) detecting planted non-identifiability.

    High mean(τ) is treated as a positive score for ``ambiguous[i]==True``.
    Mann–Whitney rank formula::

        AUC = (R_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    with average ranks for ties. Returns 0.5 if either class is empty.
    High AUROC means τ flags ambiguous subjects — it does not replace
    alignment-recovery success criteria.
    """
    scores = subject_mean_tau(tau_phi)
    amb = np.asarray(ambiguous, dtype=bool).ravel()
    if scores.shape[0] != amb.shape[0]:
        raise ValueError(f"ambiguous length {amb.shape[0]} != n_subjects {scores.shape[0]}")
    n_pos = int(np.count_nonzero(amb))
    n_neg = int(amb.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(scores)
    r_pos = float(ranks[amb].sum())
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def heldout_predictive_score(
    C_run2: np.ndarray,
    C_aligned_from_run1: np.ndarray,
) -> float:
    """Common held-out residual: -||C_run2 - aligned||_F^2 (higher better).

    All methods must be scored under this residual so alignment recovery is
    comparable. Ours may report an ELBO internally, but the shared criterion
    is this Frobenius score — uncertainty columns do not compensate for a
    worse held-out residual.
    """
    a = np.asarray(C_run2, dtype=np.float64)
    b = np.asarray(C_aligned_from_run1, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    return float(-np.sum(diff * diff))


def subject_mean_tau(tau_phi: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """(S,) mean τ per subject from per-subject tau arrays or stacked (S, ...)."""
    if isinstance(tau_phi, np.ndarray):
        arr = np.asarray(tau_phi, dtype=np.float64)
        if arr.ndim == 0:
            raise ValueError("tau_phi must be at least 1D")
        if arr.ndim == 1:
            return arr.copy()
        return arr.reshape(arr.shape[0], -1).mean(axis=1)
    means = []
    for s, tau in enumerate(tau_phi):
        t = np.asarray(tau, dtype=np.float64)
        if t.size == 0:
            raise ValueError(f"tau_phi[{s}] is empty")
        means.append(float(np.mean(t)))
    return np.asarray(means, dtype=np.float64)
