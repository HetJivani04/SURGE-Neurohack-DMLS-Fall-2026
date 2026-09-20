"""Log-domain Sinkhorn projection onto the transport polytope ``Pi(mu, nu)``.

Given a score field ``S`` ``(V, K)`` (higher means more likely coupled) the projection is::

    f_i <- eps * ( log mu_i - logsumexp_j( (S_ij + g_j) / eps ) )
    g_j <- eps * ( log nu_j - logsumexp_i( (S_ij + f_i) / eps ) )
    pi_ij = exp( (S_ij + f_i + g_j) / eps )

Every update is a logsumexp, so the iteration is stable at small ``eps`` and differentiable: gradients
reach ``S`` pathwise.
OT arithmetic is float64 on CPU.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def marginal_error(pi: np.ndarray, mu: np.ndarray, nu: np.ndarray) -> tuple[float, float]:
    """``(max|pi 1 - mu|, max|pi^T 1 - nu|)``."""
    return float(np.abs(pi.sum(axis=1) - mu).max()), float(np.abs(pi.sum(axis=0) - nu).max())


def sinkhorn_log(
    S: np.ndarray, mu: np.ndarray, nu: np.ndarray, eps: float, n_iter: int = 30, tol: float = 1e-6
) -> tuple[np.ndarray, dict]:
    """Project ``S`` ``(V, K)`` onto ``Pi(mu, nu)``; ``mu (V,)`` and ``nu (K,)`` are float64.

    Runs at most ``n_iter`` iterations and returns early once both marginals are within ``tol``
    (``tol = 0`` never exits early). Returns ``pi (V, K)`` float64 and a dict with ``row_error``,
    ``col_error`` (the final marginal errors) and ``n_iter`` (iterations performed). Entries with
    ``mu_i = 0`` or ``nu_k = 0`` get exactly zero mass.
    """
    S = np.asarray(S, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    with np.errstate(divide="ignore"):
        log_mu, log_nu = np.log(mu), np.log(nu)

    f = np.zeros(S.shape[0])
    g = np.zeros(S.shape[1])
    done = 0
    for done in range(1, n_iter + 1):
        f = eps * (log_mu - logsumexp((S + g[None, :]) / eps, axis=1))
        g = eps * (log_nu - logsumexp((S + f[:, None]) / eps, axis=0))
        if tol > 0:
            row_error, col_error = marginal_error(np.exp((S + f[:, None] + g[None, :]) / eps), mu, nu)
            if row_error <= tol and col_error <= tol:
                break

    pi = np.exp((S + f[:, None] + g[None, :]) / eps)
    row_error, col_error = marginal_error(pi, mu, nu)
    return pi, {"row_error": row_error, "col_error": col_error, "n_iter": done}


def sinkhorn_log_torch(S, mu, nu, eps: float, n_iter: int = 30):
    """The differentiable, unrolled projection used in training: exactly ``n_iter`` steps, no early exit.

    ``S`` is ``(..., V, K)`` (a leading batch of ``M`` draws is allowed), ``mu (V,)`` and ``nu (K,)``, all
    float64 torch tensors on CPU. The graph has fixed depth ``n_iter``. Each iteration is wrapped in
    gradient checkpointing, which keeps only the ``V + K`` dual potentials between iterations and recomputes
    the ``(V, K)`` intermediates in the backward pass: identical gradients, memory independent of ``n_iter``.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    log_mu, log_nu = torch.log(mu), torch.log(nu)
    f = torch.zeros(S.shape[:-1], dtype=S.dtype, device=S.device)
    g = torch.zeros(S.shape[:-2] + S.shape[-1:], dtype=S.dtype, device=S.device)

    def iterate(f, g, S):
        f = eps * (log_mu - torch.logsumexp((S + g.unsqueeze(-2)) / eps, dim=-1))
        g = eps * (log_nu - torch.logsumexp((S + f.unsqueeze(-1)) / eps, dim=-2))
        return f, g

    use_checkpoint = torch.is_grad_enabled() and S.requires_grad
    for _ in range(n_iter):
        f, g = checkpoint(iterate, f, g, S, use_reentrant=False) if use_checkpoint else iterate(f, g, S)
    return torch.exp((S + f.unsqueeze(-1) + g.unsqueeze(-2)) / eps)


def perturb_then_project(
    S_phi: np.ndarray, tau_phi: np.ndarray, xi: np.ndarray, mu: np.ndarray, nu: np.ndarray, eps: float, L: int = 30
) -> np.ndarray:
    """The sampling operator ``Sink_eps^L( exp[(S_phi + tau_phi * xi) / eps], mu, nu )``.

    ``S_phi (V, K)`` (float32 is promoted to float64), ``tau_phi (V,)`` a per-vertex noise scale broadcast over
    ``K``, ``xi (M, V, K)`` standard-normal draws. Returns ``(M, V, K)`` float64; every draw lies on
    ``Pi(mu, nu)`` up to the ``L``-step tolerance. ``xi`` is an input, so a draw is a deterministic function
    of it: this is a reparametrization, so gradients are pathwise.
    """
    scores = np.asarray(S_phi, dtype=np.float64)[None] + np.asarray(tau_phi, dtype=np.float64)[None, :, None] * xi
    return np.stack([sinkhorn_log(scores[m], mu, nu, eps, n_iter=L, tol=0.0)[0] for m in range(scores.shape[0])])
