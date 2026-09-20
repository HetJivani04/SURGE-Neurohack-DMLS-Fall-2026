"""The feature term: features transport *linearly* in the coupling.

With the Gaussian observation model ``y_si ~ N(F_bar[k], sigma_f^2 I)`` for a vertex coupled to template node
``k``, the log-likelihood of a coupling is ``-(1 / 2 sigma_f^2) sum_ik pi_ik ||y_i - F_bar_k||^2``, which is
**linear in pi**. Linearity is the point: a term linear in the coupling selects a label assignment, and that is
what reduces the isometry orbit of the Gromov-Wasserstein term to finitely many optima (never "a unique
minimizer").

Like :mod:`trajot.model.gw` these functions accept NumPy arrays or torch tensors (then differentiable).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


class FusedFeature(NamedTuple):
    """Log-likelihood pieces: ``log p(Y | pi, F_bar) = linear + constant``."""

    linear: Any  # (1 / sigma_f2) < pi , Y F_bar^T >
    constant: Any  # the pi-independent remainder, so the ELBO can log it

    @property
    def total(self) -> Any:
        return self.linear + self.constant


def fused_feature_term(pi: Any, Y_s: Any, F_bar: Any, mu_s: Any, sigma_f2: float) -> FusedFeature:
    """Feature log-likelihood of ``pi (V, K)`` for ``Y_s (V, F)``, ``F_bar (K, F)``, ``mu_s (V,)``, float64.

    Expanding ``||y_i - F_bar_k||^2`` gives::

        linear   = + (1 / sigma_f2) < pi , Y_s F_bar^T >                       (the term linear in pi)
        constant = - (1 / 2 sigma_f2) ( sum_i mu_i ||y_i||^2 + sum_k nu_k ||F_bar_k||^2 )

    with ``nu`` the column sums of ``pi``. The two parts are returned separately (a ``(linear, constant)``
    named tuple; ``.total`` is their sum). ``pi`` may carry leading batch axes ``(..., V, K)``.
    """
    linear = (pi * (Y_s @ F_bar.T)).sum((-2, -1)) / sigma_f2
    constant = -0.5 / sigma_f2 * ((mu_s * (Y_s**2).sum(-1)).sum() + (pi.sum(-2) * (F_bar**2).sum(-1)).sum(-1))
    return FusedFeature(linear, constant)


def feature_residual(pi: np.ndarray, Y_s: np.ndarray, F_bar: np.ndarray, mu_s: np.ndarray) -> np.ndarray:
    """``(V, F)`` float64 residual ``Y_s - diag(mu_s)^{-1} pi F_bar`` (the barycentric image of the template
    features), for reporting the per-vertex fit."""
    return Y_s - (pi @ F_bar) / mu_s[:, None]


def gauge_feature_cost(Y_gauge: Any, F_bar_gauge: Any) -> Any:
    """``(V, K)`` float64 squared-Euclidean cost between gauge-augmented features.

    ``Y_gauge (V, 2d)`` is ``f_i = [z_i ; beta * v_i]`` from ``geometry.gauge_features`` and
    ``F_bar_gauge (K, 2d)`` the template counterpart. ``beta`` is the calibrated inverse temperature, so the
    weight of the velocity channel is set by scan-rescan reliability, not tuned. The cost is linear in a
    coupling: ``< pi , cost >``.
    """
    return (Y_gauge**2).sum(-1)[:, None] + (F_bar_gauge**2).sum(-1)[None, :] - 2.0 * (Y_gauge @ F_bar_gauge.T)
