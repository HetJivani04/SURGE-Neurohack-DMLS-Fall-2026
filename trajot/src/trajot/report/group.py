"""Group-level random-effects analysis of the subject maps (PLAN section 5.7), and the declared sign-flip permutation.

A subject map ``z_s`` (V,) is carried into the template by the coupling: ``w_s = diag(nu)^-1 pi_s^T z_s`` (K,). Draws
of the coupling give a per-subject mean ``m_s`` and alignment covariance ``Sigma_s^al``; per template node ``k``

    w_sk = theta_k + b_sk + e_sk,   b_sk ~ N(0, tau_k^2),   e_sk ~ N(0, sigma_sk^2),   sigma_sk^2 = Sigma_s^al[k, k]

is fitted with weights ``u_sk = 1 / (tau_hat_k^2 + sigma_sk^2)``. A subject whose alignment the data do not
determine has a large ``sigma_sk^2`` and is down-weighted instead of silently averaged in. With
``Sigma_s^al -> 0`` every weight is ``1 / tau_hat_k^2`` and the model is exactly the one-sample t-test.

Dimensions are S subjects, K template nodes, V vertices, M posterior draws; everything is float64.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from functools import partial

import numpy as np
from scipy import stats

from trajot.runlog.parallel import parallel_map

_EPS = float(np.finfo(np.float64).eps)
_METHODS = ("REML", "MOM")
SIGN_FLIP_CHUNK = 250  # permutations per independent work item; part of the declared procedure (fixes the streams)


def _as_array(x, ndim: int, name: str) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64)
    if out.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {out.shape}")
    return out


def _check_nu(nu, K: int) -> np.ndarray:
    nu = _as_array(nu, 1, "nu")
    if nu.shape != (K,):
        raise ValueError(f"nu has shape {nu.shape} but the coupling has K={K} template nodes")
    if not (nu > 0).all():
        raise ValueError("nu must be strictly positive")
    return nu


def subject_maps(z_s: np.ndarray, pi_s: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """``diag(nu)^-1 pi_s^T z_s``: the (K,) template-space image of the subject map ``z_s`` (V,) under ``pi_s`` (V, K)."""
    pi = _as_array(pi_s, 2, "pi_s")
    z = _as_array(z_s, 1, "z_s")
    if z.shape != (pi.shape[0],):
        raise ValueError(f"z_s has shape {z.shape} but the coupling has V={pi.shape[0]} vertices")
    return pi.T @ z / _check_nu(nu, pi.shape[1])


def _draw_maps(pi_samples: np.ndarray, z_s: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """The (M, K) maps ``w_s^(m)`` of the M draws of one subject."""
    pis = _as_array(pi_samples, 3, "pi_samples")
    z = _as_array(z_s, 1, "z_s")
    if z.shape != (pis.shape[1],):
        raise ValueError(f"z_s has shape {z.shape} but the draws have V={pis.shape[1]} vertices")
    return np.einsum("mvk,v->mk", pis, z) / _check_nu(nu, pis.shape[2])


def alignment_covariance(pi_samples: np.ndarray, z_s: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """``Cov_m(w_s^(m))`` (K, K) over the M draws ``pi_samples`` (M, V, K), by ``np.cov``."""
    maps = _draw_maps(pi_samples, z_s, nu)
    if maps.shape[0] < 2:
        raise ValueError("need at least two draws to estimate a covariance")
    return np.atleast_2d(np.cov(maps, rowvar=False))


def delta_method_cov(J_s: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """``J_s diag(tau^2) J_s^T`` (K, K) from the Jacobian ``J_s`` (K, V) and the per-vertex noise variances ``tau2`` (V,)."""
    J = _as_array(J_s, 2, "J_s")
    tau2 = _as_array(tau2, 1, "tau2")
    if tau2.shape != (J.shape[1],):
        raise ValueError(f"tau2 has shape {tau2.shape} but J_s has V={J.shape[1]} vertices")
    if (tau2 < 0).any():
        raise ValueError("tau2 must be non-negative")
    return (J * tau2) @ J.T


def _check_maps(m, sigma2) -> tuple[np.ndarray, np.ndarray]:
    m = _as_array(m, 2, "m")
    sigma2 = _as_array(sigma2, 2, "sigma2")
    if sigma2.shape != m.shape:
        raise ValueError(f"sigma2 has shape {sigma2.shape} but m has shape {m.shape}")
    if m.shape[0] < 2:
        raise ValueError("need at least two subjects")
    if not (np.isfinite(m).all() and np.isfinite(sigma2).all()):
        raise ValueError("m and sigma2 must be finite")
    if (sigma2 < 0).any():
        raise ValueError("sigma2 must be non-negative")
    return m, sigma2


def _reml_tau2(m: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """REML ``tau_k^2 >= 0`` per node, by bisection on the restricted score (all nodes at once).

    With ``u = 1 / (tau^2 + sigma^2)`` and ``theta = sum(u m) / sum(u)`` the score is
    ``(sum u^2 (m - theta)^2 - sum u + sum u^2 / sum u) / 2``. It is positive at 0 when the optimum is interior.
    """
    K = m.shape[1]
    tau2 = np.zeros(K)
    scale = np.maximum(m.var(axis=0, ddof=1), sigma2.max(axis=0))
    live = scale > 0  # a node where nothing varies has tau^2 = 0
    if not live.any():
        return tau2
    m, sigma2, scale = m[:, live], sigma2[:, live], scale[live]

    def score(t2: np.ndarray) -> np.ndarray:
        u = 1.0 / (t2[None, :] + sigma2)
        total = u.sum(axis=0)
        theta = (u * m).sum(axis=0) / total
        return 0.5 * ((u**2 * (m - theta) ** 2).sum(axis=0) - total + (u**2).sum(axis=0) / total)

    lo, hi = 1e-15 * scale, scale.copy()
    interior = score(lo) > 0
    for _ in range(200):  # the score is negative for a large enough tau^2
        above = score(hi) > 0
        if not above.any():
            break
        hi = np.where(above, 2.0 * hi, hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        above = score(mid) > 0
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
        if (hi - lo <= 4.0 * _EPS * hi).all():
            break
    tau2[live] = np.where(interior, 0.5 * (lo + hi), 0.0)
    return tau2


def _mom_tau2(m: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """Method of moments: ``E[var_s(m)] = tau^2 + mean_s(sigma^2)``, so ``tau^2 = max(0, var(m) - mean(sigma^2))``."""
    return np.maximum(0.0, m.var(axis=0, ddof=1) - sigma2.mean(axis=0))


def meta_analysis(m: np.ndarray, sigma2: np.ndarray, method: str = "REML") -> dict:
    """Random-effects estimate per template node from ``m`` (S, K) and known ``sigma2`` (S, K).

    ``method`` picks ``tau_hat_k^2``: ``"REML"`` or ``"MoM"`` (method of moments). Returns ``tau_hat_k2`` (K,), the
    weights ``u_sk`` (S, K), ``theta_hat_k`` (K,), ``se_theta`` (K,), ``t_k`` (K,), the Satterthwaite ``df`` (K,) and
    the two-sided ``p_k`` (K,). The df is ``2 (SE^2)^2 / Var(SE^2)`` with ``Var(tau_hat^2)`` from the REML
    information ``tr(P^2) / 2``; it equals S - 1 when the weights are equal, and is at least 1. The test keeps its
    size, and is conservative (measured: 1% to 4% at a 5% level) when ``tau_hat^2`` sits at 0 and the weights are
    very unequal, because the df then treats ``tau_hat^2`` as if it could not be truncated at 0.

    A zero variance never divides by zero: ``u`` is infinite for a subject with ``tau^2 + sigma^2 = 0``, which then
    fixes ``theta`` (their mean) with ``se = 0`` and ``df = S - 1``, as in a t-test of constant data.
    """
    m, sigma2 = _check_maps(m, sigma2)
    if not isinstance(method, str) or method.upper() not in _METHODS:
        raise ValueError(f"method must be 'REML' or 'MoM', got {method!r}")
    tau2 = _reml_tau2(m, sigma2) if method.upper() == "REML" else _mom_tau2(m, sigma2)

    S, K = m.shape
    v = tau2[None, :] + sigma2
    zero = v <= 0
    degenerate = zero.any(axis=0)
    with np.errstate(divide="ignore"):
        u = np.where(zero, np.inf, 1.0 / np.where(zero, 1.0, v))

    theta, se, df = np.empty(K), np.empty(K), np.full(K, S - 1.0)
    fine = ~degenerate
    uf, mf = u[:, fine], m[:, fine]
    total, sq, cube = uf.sum(axis=0), (uf**2).sum(axis=0), (uf**3).sum(axis=0)
    theta[fine], se[fine] = (uf * mf).sum(axis=0) / total, total**-0.5
    df[fine] = np.maximum(total**2 * (sq - 2.0 * cube / total + sq**2 / total**2) / sq**2, 1.0)
    if degenerate.any():
        z = zero[:, degenerate]
        theta[degenerate], se[degenerate] = (z * m[:, degenerate]).sum(axis=0) / z.sum(axis=0), 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = theta / se
    return {"tau_hat_k2": tau2, "u_sk": u, "theta_hat_k": theta, "se_theta": se, "t_k": t, "df": df,
            "p_k": 2.0 * stats.t.sf(np.abs(t), df)}


def one_sample_ttest(m: np.ndarray) -> dict:
    """The one-sample t-test of ``m`` (S, K) against 0 per node: what ``meta_analysis`` collapses to as sigma2 -> 0.

    Same keys as ``meta_analysis``: ``tau_hat_k2`` is the sample variance, ``u_sk`` its inverse in every row, and
    ``df`` is S - 1.
    """
    m = _as_array(m, 2, "m")
    S, K = m.shape
    if S < 2:
        raise ValueError("need at least two subjects")
    mean, var = m.mean(axis=0), m.var(axis=0, ddof=1)
    se, df = np.sqrt(var / S), np.full(K, S - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t, u = mean / se, np.broadcast_to(1.0 / var, m.shape).copy()
    return {"tau_hat_k2": var, "u_sk": u, "theta_hat_k": mean, "se_theta": se, "t_k": t, "df": df,
            "p_k": 2.0 * stats.t.sf(np.abs(t), df)}


def meta_analysis_map(pi_samples: Iterable[np.ndarray], tau_phi: Iterable[np.ndarray], z_s: Iterable[np.ndarray],
                      nu: np.ndarray, method: str = "REML") -> dict:
    """The group analysis over all subjects.

    ``pi_samples`` yields one (M, V_s, K) block of coupling draws per subject, ``tau_phi`` the subject's (V_s,)
    posterior widths and ``z_s`` its (V_s,) subject map; ``nu`` is the (K,) template mass. The three are consumed
    together one subject at a time, so a generator of blocks is never held whole in memory. ``Sigma_s^al`` is the
    covariance over the draws (only its diagonal enters). Returns the ``meta_analysis`` keys, ``m`` and ``sigma2``
    (S, K), and the down-weighting diagnostics: ``mean_sigma2`` (S,) and ``mean_tau_phi`` (S,) per subject,
    ``mean_weight`` (S,), the subject's mean normalised weight (1/S each with no alignment uncertainty), and
    ``n_eff`` (K,), the effective number of subjects ``(sum u)^2 / sum u^2`` per node (nodes where some
    ``u`` is infinite take the equal-weight limit ``#{u = inf}``), and ``n_degenerate_nodes``, the number of
    nodes where ``tau^2 + sigma^2 = 0`` exactly for at least one subject.
    """
    means, variances, widths = [], [], []
    missing = object()
    blocks, taus, maps_of = iter(pi_samples), iter(tau_phi), iter(z_s)
    index = 0
    while True:
        pi, tau, z = next(blocks, missing), next(taus, missing), next(maps_of, missing)
        ended = [item is missing for item in (pi, tau, z)]
        if all(ended):
            break
        if any(ended):
            raise ValueError("pi_samples, tau_phi and z_s must have the same number of subjects")
        maps = _draw_maps(pi, z, nu)
        tau = _as_array(tau, 1, "tau_phi")
        if maps.shape[0] < 2:
            raise ValueError("need at least two draws to estimate a covariance")
        if len(tau) != len(z):
            raise ValueError(f"subject {index}: the draws have {len(z)} vertices but tau_phi has {len(tau)}")
        means.append(maps.mean(axis=0))
        variances.append(maps.var(axis=0, ddof=1))
        widths.append(float(tau.mean()))
        del pi, maps  # freed before the next block is produced
        index += 1
    m, sigma2 = np.stack(means) if means else np.empty((0, 0)), np.stack(variances) if variances else np.empty((0, 0))
    out = meta_analysis(m, sigma2, method)
    u = np.asarray(out["u_sk"], dtype=np.float64)
    # n_eff = (sum u)^2 / sum u^2 on per-node rescaled weights, so weights at the fp noise floor
    # (sigma^2 ~ 1e-30) cannot overflow. A subject-node with tau^2 + sigma^2 = 0 exactly has
    # u = inf and fixes theta at its own mean with equal weight (see ``meta_analysis``), so those
    # nodes take the limit n_eff = #{u = inf} instead of nan.
    n_inf = np.isinf(u).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_max = np.where(np.isfinite(u), u, 0.0).max(axis=0)
        scaled = u / np.where(u_max > 0, u_max, 1.0)[None, :]
        total, sq = scaled.sum(axis=0), (scaled**2).sum(axis=0)
        n_eff = total**2 / sq
        weights = scaled / total[None, :]
    n_eff = np.where(n_inf > 0, n_inf.astype(np.float64), n_eff)
    return {**out, "m": m, "sigma2": sigma2, "mean_sigma2": sigma2.mean(axis=1), "mean_tau_phi": np.array(widths),
            "mean_weight": np.nanmean(weights, axis=1), "n_eff": n_eff,
            "n_degenerate_nodes": int((n_inf > 0).sum())}


def _flip_chunk(contrast: np.ndarray, item: tuple[int, int, int]) -> np.ndarray:
    index, size, seed = item
    rng = np.random.default_rng([seed, index])
    signs = 2.0 * rng.integers(0, 2, size=(size, contrast.size)) - 1.0
    return signs @ contrast / contrast.size


def _check_contrast(contrast) -> np.ndarray:
    c = np.asarray(contrast, dtype=np.float64)
    if c.ndim != 1:
        raise ValueError(f"contrast must be 1-D, got shape {c.shape}")
    if c.size == 0:
        raise ValueError("contrast is empty")
    if not np.isfinite(c).all():
        raise ValueError("contrast must be finite")
    return c


def sign_flip_null(contrast: np.ndarray, B: int, seed: int, n_jobs: int = 1) -> np.ndarray:
    """The (B,) null of the mean of ``contrast`` (P,) under random sign flips (Winkler et al. 2014).

    Each null value is ``mean(s * contrast)`` for a uniformly random sign vector ``s`` in {-1, +1}^P: exact under
    the null that the contrast is symmetric about 0. The B flips are split into independent chunks of
    ``SIGN_FLIP_CHUNK``, chunk ``i`` drawing from ``default_rng([seed, i])``, so the result is fixed by
    ``(contrast, B, seed)`` and identical for every ``n_jobs`` (chunks run through W0's ``parallel_map``).
    """
    c = _check_contrast(contrast)
    if isinstance(B, bool) or not isinstance(B, (int, np.integer)) or B < 1:
        raise ValueError(f"B must be a positive integer, got {B!r}")
    items = [(i, min(SIGN_FLIP_CHUNK, B - i * SIGN_FLIP_CHUNK), int(seed)) for i in range(math.ceil(B / SIGN_FLIP_CHUNK))]
    return np.concatenate(parallel_map(partial(_flip_chunk, c), items, n_jobs=n_jobs, mode="outer"))


def sign_flip_test(contrast: np.ndarray, n_pairs: int, B: int, seed: int, n_jobs: int = 1) -> dict:
    """The declared group-contrast test: ``contrast`` (P,) holds one value per declared pair, and P must be ``n_pairs``.

    Returns ``statistic`` (the mean contrast), the two-sided ``p_value = (#{|null| >= |statistic|} + 1) / (B + 1)``
    (the observed labelling counts as one flip), the ``null`` (B,), and the declared ``n_pairs``, ``B`` and ``seed``.
    """
    c = _check_contrast(contrast)
    if c.size != n_pairs:
        raise ValueError(f"the declared subsample is n_pairs={n_pairs} but the contrast has {c.size} values")
    null = sign_flip_null(c, B, seed, n_jobs)
    statistic = float(c.mean())
    extreme = int((np.abs(null) >= abs(statistic) - 1e-12 * (1.0 + abs(statistic))).sum())
    return {"statistic": statistic, "p_value": (extreme + 1) / (B + 1), "null": null, "n_pairs": int(n_pairs),
            "B": int(B), "seed": int(seed)}
