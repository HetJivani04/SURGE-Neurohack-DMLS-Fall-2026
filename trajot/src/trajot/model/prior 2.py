"""Priors on the coupling and on the population parameters ``theta = {B, F_bar, eps_{1:S}}``."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _is_torch(x: Any) -> bool:
    return type(x).__module__.startswith("torch")


def gibbs_prior(pi: Any, M_s0: Any, eps_s: Any) -> Any:
    """``- < pi , M_s0 > / eps_s``: the log of the Gibbs prior on the polytope, up to its normalizer.

    ``M_s0 (V, K)`` float64 is the anatomical anchor from ``geometry.anatomical_cost`` (geodesic distance after
    surface registration); ``eps_s`` is the subject's sharpness, itself drawn from the population hyperprior, so
    sharpness is partially pooled. ``pi`` may carry leading batch axes ``(..., V, K)``.
    """
    return -(pi * M_s0).sum((-2, -1)) / eps_s


def dominant_frequency(x: Any, dt: float) -> Any:
    """Per-row dominant frequency in Hz of ``x (n, L)`` sampled every ``dt`` seconds: the largest non-DC
    Fourier component of each demeaned row."""
    if _is_torch(x):
        import torch

        spectrum = torch.fft.rfft(x - x.mean(-1, keepdim=True), dim=-1).abs()
        peak = torch.argmax(spectrum[..., 1:], dim=-1) + 1
        return peak.to(x.dtype) / (x.shape[-1] * dt)
    x = np.asarray(x, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(x - x.mean(-1, keepdims=True), axis=-1))
    return (np.argmax(spectrum[..., 1:], axis=-1) + 1) / (x.shape[-1] * dt)


def band_penalty_matrix(A_s: Any, A_bar: Any, dt: float, band_seconds: tuple[float, float], bandwidth: float) -> Any:
    """``(V, K)`` band penalty for every vertex / template-node pair (see :func:`band_prior`)."""
    low_seconds, high_seconds = band_seconds
    f_i = dominant_frequency(A_s, dt)  # (V,)
    g_k = dominant_frequency(A_bar, dt)  # (K,)
    lo, hi = 1.0 / high_seconds, 1.0 / low_seconds
    in_band = ((f_i >= lo) & (f_i <= hi))[:, None] & ((g_k >= lo) & (g_k <= hi))[None, :]
    gap = (f_i[:, None] - g_k[None, :])
    gap = gap.abs() if _is_torch(gap) else np.abs(gap)
    zero = gap * 0.0
    penalty = gap / bandwidth
    if _is_torch(gap):
        import torch

        return torch.where(in_band, zero, penalty)
    return np.where(in_band, zero, penalty)


def band_prior(pi: Any, A_s: Any, A_bar: Any, dt: float, band_seconds: tuple[float, float], bandwidth: float) -> Any:
    """This is a band prior, not a model of temporal dynamics: it constrains the frequency content of a
    coupling and says nothing about trajectories or a shared time axis (rest fMRI has none).

    The expected penalty ``< pi , penalty >`` with, per vertex ``i`` and template node ``k``::

        penalty[i, k] = 0                              if 1/high_seconds <= f_i, g_k <= 1/low_seconds
                      = | f_i - g_k | / bandwidth      otherwise

    where ``f_i`` and ``g_k`` are the dominant frequencies (Hz) of row ``i`` of ``A_s`` and row ``k`` of
    ``A_bar``, each row read as a sequence sampled every ``dt`` seconds (``dt`` is the repetition time from the
    contract), and the band edges given in seconds are converted to Hz as ``1/high_seconds`` and
    ``1/low_seconds``. ``pi`` may carry leading batch axes.
    """
    penalty = band_penalty_matrix(A_s, A_bar, dt, band_seconds, bandwidth)
    return (pi * penalty).sum((-2, -1))


def constrain_scale(B: Any, max_row_norm: float) -> Any:
    """Project every row of ``B (K, r)`` onto the ball of radius ``max_row_norm``; a new array of the same kind.

    The Gibbs normalizer depends on the template, so ``B`` must be scale-constrained for the generalized
    posterior to be well posed; this keeps every template entry ``|C_bar[k, l]| <= max_row_norm^2``.
    """
    if _is_torch(B):
        norms = B.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return B * (max_row_norm / norms).clamp_max(1.0)
    norms = np.maximum(np.linalg.norm(B, axis=-1, keepdims=True), 1e-12)
    return B * np.minimum(1.0, max_row_norm / norms)


def log_prior_theta(params: Any, cfg: Any) -> Any:
    """``log p(B) + log p(F_bar) + sum_s log p(eps_s)``.

    ``p(B) = N(0, sigma_B^2 I)``, ``p(F_bar) = N(F0, sigma_F^2 I)``, ``p(eps_s) = InvGamma(a_eps, b_eps)``, with
    the hyperparameters read from ``cfg`` (``model.prior.*``). ``params`` has ``B (K, r)``, ``F_bar (K, F)`` and
    ``eps (S,)`` (NumPy or torch). ``B`` is additionally scale-constrained (:func:`constrain_scale`, applied by
    the training loop) because the Gibbs normalizer depends on the template.
    """
    sigma_B, sigma_F, F0 = (float(cfg.get(f"model.prior.{k}")) for k in ("sigma_B", "sigma_F", "F0"))
    a_eps, b_eps = float(cfg.get("model.prior.a_eps")), float(cfg.get("model.prior.b_eps"))
    B, F_bar, eps = params.B, params.F_bar, params.eps

    log = np.log if not _is_torch(eps) else __import__("torch").log
    log_B = -0.5 * (B**2).sum() / sigma_B**2 - 0.5 * B.shape[0] * B.shape[1] * math.log(2 * math.pi * sigma_B**2)
    log_F = (-0.5 * ((F_bar - F0) ** 2).sum() / sigma_F**2
             - 0.5 * F_bar.shape[0] * F_bar.shape[1] * math.log(2 * math.pi * sigma_F**2))
    log_eps = (a_eps * math.log(b_eps) - math.lgamma(a_eps) - (a_eps + 1.0) * log(eps) - b_eps / eps).sum()
    return log_B + log_F + log_eps
