"""The Gromov-Wasserstein term, in factored form.

For a coupling ``pi`` ``(V, K)``, subject geometry ``C_s = A_s A_s^T`` (``A_s`` is ``(V, r)``) and template
geometry ``C_bar = B B^T`` (``B`` is ``(K, r)``)::

    E_GW(pi) = sum_{ijkl} (C_s[i,j] - C_bar[k,l])^2 pi[i,k] pi[j,l]
             = < C_s^{o2} mu_s 1^T + 1 nu^T (C_bar^{o2})^T , pi >  -  2 < C_s pi C_bar , pi >

where ``C^{o2}`` is the elementwise square and the identity holds on the transport polytope
``pi 1 = mu_s``, ``pi^T 1 = nu``. Everything is evaluated through ``A_s`` and ``B``: **no ``(V, V)`` array is
ever formed**, so the cost is ``O(V K r + V r^2 + K r^2)`` and the peak memory does not grow with ``V^2``.

The functions accept NumPy arrays (and return Python floats / arrays) or torch tensors (and return tensors, so
that autograd flows to ``pi``, ``B`` and anything upstream). OT and GW arithmetic is float64.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _is_torch(x: Any) -> bool:
    return type(x).__module__.startswith("torch")


def _marginal_terms(A_s: Any, B: Any, nu: Any, mu_s: Any) -> tuple[Any, Any]:
    """``u = C_s^{o2} mu_s`` ``(V,)`` and ``w = C_bar^{o2} nu`` ``(K,)``, without forming ``C_s`` or ``C_bar``.

    ``u_i = sum_j mu_j (a_i . a_j)^2 = a_i^T (A^T diag(mu) A) a_i``, an ``r x r`` quadratic form per row.
    """
    q_subject = A_s.T @ (mu_s[:, None] * A_s)  # (r, r)
    q_template = B.T @ (nu[:, None] * B)  # (r, r)
    u = ((A_s @ q_subject) * A_s).sum(-1)  # (V,)
    w = ((B @ q_template) * B).sum(-1)  # (K,)
    return u, w


def gw_cross(pi: Any, A_s: Any, B: Any) -> Any:
    """``C_s pi C_bar`` ``(V, K)`` float64, as ``A_s @ (A_s.T @ pi @ B) @ B.T``.

    Evaluated left to right, so the intermediates are ``(r, K)``, ``(r, r)`` and ``(V, r)`` (with a leading
    batch axis if ``pi`` has one).
    """
    return A_s @ (A_s.T @ pi @ B) @ B.T


def gw_term(pi: Any, A_s: Any, B: Any, nu: Any, mu_s: Any) -> Any:
    """``E_GW(pi)`` for ``pi (V, K)``, ``A_s (V, r)``, ``B (K, r)``, ``nu (K,)``, ``mu_s (V,)``, all float64.

    Returns a Python float for NumPy input and a 0-d tensor (differentiable) for torch input.
    ``mu_s`` and ``nu`` enter as the marginals of the polytope; ``pi`` is contracted with them exactly as in
    the formula, so the gradient with respect to ``pi`` is :func:`gw_gradient`. ``pi`` may carry leading batch
    axes ``(..., V, K)`` (e.g. ``M`` posterior draws); the result then has shape ``(...)``.
    """
    u, w = _marginal_terms(A_s, B, nu, mu_s)
    linear = (u * pi.sum(-1)).sum(-1) + (w * pi.sum(-2)).sum(-1)  # < u 1^T + 1 w^T , pi >
    cross = (gw_cross(pi, A_s, B) * pi).sum((-2, -1))
    value = linear - 2.0 * cross
    if _is_torch(pi):
        return value
    return float(value) if np.ndim(value) == 0 else value


def gw_gradient(pi: np.ndarray, A_s: np.ndarray, B: np.ndarray, nu: np.ndarray, mu_s: np.ndarray) -> np.ndarray:
    """``dE_GW / dpi = C_s^{o2} mu_s 1^T + 1 nu^T (C_bar^{o2})^T - 4 C_s pi C_bar``, ``(V, K)`` float64."""
    u, w = _marginal_terms(A_s, B, nu, mu_s)
    return u[:, None] + w[None, :] - 4.0 * gw_cross(pi, A_s, B)


def gw_reference_bruteforce(pi: np.ndarray, C_s: np.ndarray, C_bar: np.ndarray) -> float:
    """The four-index double sum ``sum_{ijkl} (C_s[i,j] - C_bar[k,l])^2 pi[i,k] pi[j,l]``.

    Takes the dense ``(V, V)`` and ``(K, K)`` matrices and materializes a ``(V, V, K, K)`` array: for tests only.
    """
    squared_difference = (C_s[:, :, None, None] - C_bar[None, None, :, :]) ** 2  # (V, V, K, K)
    return float(np.einsum("ijkl,ik,jl->", squared_difference, pi, pi))
