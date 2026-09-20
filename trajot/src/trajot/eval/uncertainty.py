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
    "barycentric_map_variance",
    "heldout_predictive_score",
    "mean_row_entropy",
    "nonident_auroc",
    "posterior_coverage",
    "scale_template_frobenius",
    "shrinkage_transform",
    "subject_mean_tau",
    "subject_row_entropies",
    "temperature_calibrated_coverage",
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


def scale_template_frobenius(
    C_bar: np.ndarray,
    C_ref: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Rescale a learned template so ``||C_bar_scaled||_F = ||C_ref||_F``.

    ``C_bar = B B^T`` after ``B_max_row_norm`` is often an order of magnitude
    smaller than the empirical population connectome. Using the raw ``C_bar``
    as an OT / Procrustes target puts subject maps in the wrong metric.
    Returns ``(C_bar * scale, scale)``.
    """
    C_bar = np.asarray(C_bar, dtype=np.float64)
    C_ref = np.asarray(C_ref, dtype=np.float64)
    if C_bar.shape != C_ref.shape:
        raise ValueError(f"shape mismatch: C_bar {C_bar.shape} vs C_ref {C_ref.shape}")
    n_bar = float(np.linalg.norm(C_bar, ord="fro"))
    n_ref = float(np.linalg.norm(C_ref, ord="fro"))
    if n_bar <= 0.0:
        raise ValueError("C_bar Frobenius norm is zero; cannot rescale")
    scale = n_ref / n_bar
    return C_bar * scale, scale


def _row_stochastic_rows(pi: np.ndarray) -> np.ndarray:
    P = np.asarray(pi, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] == 0:
        raise ValueError(f"pi must be non-empty 2D, got {P.shape}")
    row = np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
    return P / row


def mean_row_entropy(pi: np.ndarray) -> float:
    """Mean row entropy (nats) of a coupling after row-normalization.

    Sharp (near-permutation) maps have entropy near 0; 50/50 mixtures of two
    permutations have entropy near ``log 2``. This is the pre-registered
    *assignment* uncertainty score — not Sinkhorn ``tau_phi``.
    """
    P = _row_stochastic_rows(pi)
    ent = -(P * np.log(np.clip(P, 1e-300, None))).sum(axis=1)
    return float(np.mean(ent))


def subject_row_entropies(pi_bars: Sequence[np.ndarray]) -> np.ndarray:
    """``(S,)`` mean row entropy per subject from pooled posterior-mean couplings."""
    return np.asarray([mean_row_entropy(p) for p in pi_bars], dtype=np.float64)


def barycentric_map_variance(pi_draws: np.ndarray) -> float:
    """Mean across vertices of the variance of the barycentric template index.

    For draws ``(M, V, K)``, the barycentric map of draw ``m`` is
    ``E[k | v] = sum_k P_m[v, k] * k`` under row-stochastic ``P_m``. High
    variance means the posterior moves mass across template nodes (alignment
    ambiguity); a collapsed sharp posterior has near-zero variance.
    """
    draws = np.asarray(pi_draws, dtype=np.float64)
    if draws.ndim != 3:
        raise ValueError(f"pi_draws must be (M,V,K), got {draws.shape}")
    M = draws.shape[0]
    if M < 2:
        return 0.0
    P = draws / np.maximum(draws.sum(axis=2, keepdims=True), 1e-12)
    idx = np.arange(draws.shape[2], dtype=np.float64)
    bc = P @ idx  # (M, V)
    return float(np.mean(np.var(bc, axis=0)))


def _coverage_at_temperature(
    draws: np.ndarray,
    star: np.ndarray,
    alpha: float,
    temperature: float,
    half_floor: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    mean = draws.mean(axis=0)
    lo = np.quantile(draws, alpha / 2.0, axis=0)
    hi = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
    half = np.maximum.reduce([
        (hi - lo) / 2.0,
        np.abs(mean - lo),
        np.abs(hi - mean),
        np.full_like(mean, half_floor),
    ])
    lo_t = mean - temperature * half
    hi_t = mean + temperature * half
    cov = float(np.mean((star >= lo_t) & (star <= hi_t)))
    return cov, lo_t, hi_t


def temperature_calibrated_coverage(
    draws_by_subject: Sequence[np.ndarray],
    stars_by_subject: Sequence[np.ndarray],
    *,
    alpha: float = 0.1,
    target: float = 0.90,
    cal_subjects: Sequence[int] | None = None,
    halfwidth_floor_frac: float = 0.25,
    temperature_grid: Sequence[float] | None = None,
) -> dict[str, float | bool]:
    """Held-out temperature calibration of empirical coupling coverage.

    Scientific caveat (calibration debt, not Bayes)
    -----------------------------------------------
    The Sinkhorn entropy Jacobian is **detached** in ``trajot.inference.train``,
    so posterior ``pi`` draws are overconfident: raw quantile intervals miss
    planted ``P*`` almost everywhere. This helper expands quantile half-widths
    by a scalar ``T`` fit on calibration subjects, then reports coverage on
    held-out subjects. Intervals are in the **same space** as the planted
    object (pooled posterior draws vs ``pi_star``), never mixed across spaces.

    Half-width floor: ``halfwidth_floor_frac / R`` with ``R = star.shape[-1]``,
    so collapsed (zero-width) posteriors still have measurable intervals.

    Odd subjects are evaluation by default; even subjects are calibration.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if len(draws_by_subject) != len(stars_by_subject):
        raise ValueError("draws and stars must have the same number of subjects")
    n = len(draws_by_subject)
    if n == 0:
        raise ValueError("empty subject list")
    draws_list = [np.asarray(d, dtype=np.float64) for d in draws_by_subject]
    stars = [np.asarray(s, dtype=np.float64) for s in stars_by_subject]
    for i, (d, s) in enumerate(zip(draws_list, stars)):
        if d.ndim < 2 or d.shape[1:] != s.shape:
            raise ValueError(f"subject {i}: draws {d.shape} vs star {s.shape}")
        if d.shape[0] == 0:
            raise ValueError(f"subject {i}: no draws")
    R = int(stars[0].shape[-1])
    half_floor = float(halfwidth_floor_frac) / float(max(R, 1))
    if cal_subjects is None:
        cal_idx = [i for i in range(n) if i % 2 == 0]
        eval_idx = [i for i in range(n) if i % 2 == 1]
    else:
        cal_idx = [int(i) for i in cal_subjects]
        eval_idx = [i for i in range(n) if i not in set(cal_idx)]
    if not cal_idx or not eval_idx:
        # Degenerate split: calibrate and report on the same subjects (flagged).
        cal_idx = list(range(n))
        eval_idx = list(range(n))

    raw_cal = [
        posterior_coverage(draws_list[i], stars[i], alpha=alpha) for i in cal_idx
    ]
    raw_eval = [
        posterior_coverage(draws_list[i], stars[i], alpha=alpha) for i in eval_idx
    ]

    def mean_cov_at(t: float, idxs: Sequence[int]) -> float:
        vals = [
            _coverage_at_temperature(draws_list[i], stars[i], alpha, t, half_floor)[0]
            for i in idxs
        ]
        return float(np.mean(vals)) if vals else 0.0

    if temperature_grid is None:
        grid = np.concatenate([
            np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]),
            np.geomspace(10.0, 200.0, num=12),
        ])
    else:
        grid = np.asarray(list(temperature_grid), dtype=np.float64)
    best_t = 1.0
    best_err = float("inf")
    for t in grid:
        cov = mean_cov_at(float(t), cal_idx)
        err = abs(cov - float(target))
        if err < best_err - 1e-12 or (abs(err - best_err) <= 1e-12 and t < best_t):
            best_err = err
            best_t = float(t)

    cov_eval = mean_cov_at(best_t, eval_idx)
    cov_cal = mean_cov_at(best_t, cal_idx)
    in_band = bool(0.85 <= cov_eval <= 0.95)
    return {
        "coverage_eval": cov_eval,
        "coverage_cal": cov_cal,
        "coverage_raw_eval": float(np.mean(raw_eval)) if raw_eval else 0.0,
        "coverage_raw_cal": float(np.mean(raw_cal)) if raw_cal else 0.0,
        "temperature": best_t,
        "halfwidth_floor": half_floor,
        "target": float(target),
        "alpha": float(alpha),
        "n_cal": len(cal_idx),
        "n_eval": len(eval_idx),
        "in_pre_registered_band": in_band,
        "calibration_is_bayes": False,
    }


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
