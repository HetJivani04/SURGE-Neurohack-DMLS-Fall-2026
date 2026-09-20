"""The four ELBO terms, the objective, and the annealing schedules.

Conventions. ``gw``, ``feature`` and ``prior`` are *energies* (to be minimized) and enter the objective with a
minus sign::

    total = -(beta / 2) gw - feature - prior + entropy + log_prior

* ``gw``      ``E_GW(pi)`` averaged over the ``M`` draws (:func:`trajot.model.gw.gw_term`)
* ``feature`` ``- log p(Y | pi, F_bar)``, the linear-in-``pi`` transport cost of the features
* ``prior``   ``< pi , M_s0 > / eps_s`` (the negative log Gibbs prior) plus, if given, the expected band penalty
* ``entropy`` ``H[q(pi)] = - mean(log_q)``
* ``log_prior`` ``log p(theta)``

Everything is float64. NumPy inputs give NumPy scalars, torch inputs give differentiable tensors.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from trajot.model.features import fused_feature_term
from trajot.model.gw import gw_term
from trajot.model.prior import gibbs_prior


def _is_torch(x: Any) -> bool:
    return type(x).__module__.startswith("torch")


def elbo_terms(
    pi_samples: Any, A_s: Any, B: Any, nu: Any, mu_s: Any, Y_s: Any, F_bar: Any, M_s0: Any, eps_s: Any,
    sigma_f2: float, beta: float, log_q: Any, *, band_penalty: Any = None, log_prior: Any = 0.0,
) -> dict[str, Any]:
    """The per-subject ELBO terms for ``M`` posterior draws ``pi_samples (M, V, K)`` on ``Pi(mu_s, nu)``.

    ``log_q (M,)`` is ``log q(pi)`` of each draw (so ``entropy = -mean(log_q)``). ``band_penalty (V, K)`` is the
    optional band penalty matrix and ``log_prior`` the population term ``log p(theta)`` (computed once per step
    by :func:`trajot.model.prior.log_prior_theta`). Returns float64 values under ``gw``, ``feature``,
    ``prior``, ``entropy``, ``log_prior`` and ``total``.
    """
    gw = gw_term(pi_samples, A_s, B, nu, mu_s).mean()
    fused = fused_feature_term(pi_samples, Y_s, F_bar, mu_s, sigma_f2)
    feature = -(fused.linear + fused.constant).mean()
    prior = -gibbs_prior(pi_samples, M_s0, eps_s).mean()
    if band_penalty is not None:
        prior = prior + (pi_samples * band_penalty).sum((-2, -1)).mean()
    entropy = -log_q.mean()
    if _is_torch(pi_samples):
        import torch

        log_prior = torch.as_tensor(log_prior, dtype=pi_samples.dtype, device=pi_samples.device)
    else:
        log_prior = np.float64(log_prior)
    total = -0.5 * beta * gw - feature - prior + entropy + log_prior
    return {"gw": gw, "feature": feature, "prior": prior, "entropy": entropy, "log_prior": log_prior, "total": total}


def elbo(terms: Sequence[dict[str, Any]]) -> Any:
    """The objective ``mean_s mean_m [ -(beta/2) E_GW - feature - prior ] + H[q(pi_s)] + log p(theta)``:
    the mean of the per-subject ``total`` of :func:`elbo_terms` (each already includes ``log_prior`` once)."""
    totals = [t["total"] for t in terms]
    if _is_torch(totals[0]):
        import torch

        return torch.stack(totals).mean()
    return np.float64(np.mean(totals))


def beta_schedule(step: int, total_steps: int, beta_target: float, warmup_frac: float = 0.3) -> float:
    """``beta_target * min(1, step / (warmup_frac * total_steps))``: beta is tempered linearly from 0 and
    equals ``beta_target`` exactly from ``warmup_frac`` of the steps on (30% by default)."""
    warmup = warmup_frac * total_steps
    if warmup <= 0 or step >= warmup * (1.0 - 1e-12):
        return float(beta_target)
    return float(beta_target * step / warmup)


def eps_schedule(step: int, total_steps: int, eps_start: float = 0.1, eps_end: float = 0.01) -> float:
    """Geometric annealing of the Sinkhorn temperature from ``eps_start`` (step 0) to ``eps_end``
    (step ``total_steps``): ``eps_start * (eps_end / eps_start) ** (step / total_steps)``. Both endpoints are
    returned exactly."""
    if step <= 0:
        return float(eps_start)
    if step >= total_steps:
        return float(eps_end)
    return float(eps_start * math.exp(math.log(eps_end / eps_start) * step / total_steps))
