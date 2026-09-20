"""Entropy of the pushed-forward posterior: ``H[q(pi)] = H[q(xi)] + E[log |det J_f(xi)|]``.

``pi = f_phi(xi)`` is the perturb-then-project map, so the change-of-variables term needs the log-determinant
of its Jacobian. It is estimated from Jacobian-vector products only, with Rademacher probes:
:func:`hutchinson_logdet` (the trace estimator) and :func:`slq_logdet` (stochastic Lanczos quadrature, the
estimator the entropy term uses). All arithmetic is float64.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

Matvec = Callable[[np.ndarray], np.ndarray]


def _rademacher(dim: int, generator: np.random.Generator) -> np.ndarray:
    return generator.choice(np.array([-1.0, 1.0]), size=dim)


def hutchinson_logdet(matvec: Matvec, dim: int, n_probes: int = 4, generator: np.random.Generator | None = None) -> float:
    """Hutchinson estimate ``E[z^T A z] = tr(A)`` with Rademacher probes ``z``, for the operator ``A = J_f``
    given by ``matvec``. It is unbiased for ``tr(A)``; for a Jacobian near the identity, ``tr(J_f - I)`` is the
    first-order term of ``log |det J_f|``. Use :func:`slq_logdet` for the full log-determinant."""
    generator = generator or np.random.default_rng()
    return float(np.mean([z @ matvec(z) for z in (_rademacher(dim, generator) for _ in range(n_probes))]))


def slq_logdet(
    matvec: Matvec, dim: int, n_probes: int = 4, n_lanczos: int = 20, generator: np.random.Generator | None = None
) -> float:
    """Stochastic Lanczos quadrature estimate of ``log det A`` for a symmetric positive-definite operator.

    For each Rademacher probe ``z``, ``n_lanczos`` Lanczos steps (with full reorthogonalization) give a
    tridiagonal ``T`` and ``z^T log(A) z ~ ||z||^2 e_1^T log(T) e_1``; the probes are averaged. For a
    non-symmetric Jacobian ``J`` pass the Gram operator ``J^T J`` and halve the result.
    """
    generator = generator or np.random.default_rng()
    steps = min(n_lanczos, dim)
    total = 0.0
    for _ in range(n_probes):
        z = _rademacher(dim, generator)
        basis = np.zeros((steps, dim))
        alpha, beta = np.zeros(steps), np.zeros(max(steps - 1, 0))
        basis[0] = z / np.linalg.norm(z)
        m = steps
        for j in range(steps):
            w = matvec(basis[j])
            alpha[j] = basis[j] @ w
            w = w - alpha[j] * basis[j] - (beta[j - 1] * basis[j - 1] if j > 0 else 0.0)
            for _ in range(2):  # reorthogonalize twice against the whole basis
                w = w - basis[: j + 1].T @ (basis[: j + 1] @ w)
            if j == steps - 1:
                break
            beta[j] = np.linalg.norm(w)
            if beta[j] < 1e-12:  # invariant subspace found: the quadrature is exact
                m = j + 1
                break
            basis[j + 1] = w / beta[j]
        tridiagonal = np.diag(alpha[:m]) + np.diag(beta[: m - 1], 1) + np.diag(beta[: m - 1], -1)
        nodes, vectors = np.linalg.eigh(tridiagonal)
        total += dim * float((vectors[0] ** 2 * np.log(np.clip(nodes, 1e-300, None))).sum())
    return total / n_probes


def entropy_estimator(log_q_xi: np.ndarray, logdet_terms: np.ndarray) -> float:
    """``H[q(pi)] = H[q(xi)] + E[log |det J_f(xi)|]`` from draws: ``H[q(xi)] = -mean(log q(xi))`` and
    ``logdet_terms`` the per-draw ``log |det J_f(xi)|``."""
    return float(-np.mean(log_q_xi) + np.mean(logdet_terms))


def projected_sinkhorn_logdet(
    score: np.ndarray,
    mu: np.ndarray,
    nu: np.ndarray,
    eps: float,
    n_iter: int = 30,
    subspace_dim: int = 12,
    ridge: float = 1e-3,
    fd_step: float = 1e-3,
    clip: float = 80.0,
    generator: np.random.Generator | None = None,
) -> float:
    """Detached Hutchinson-style estimate of ``log |det J|`` for ``s |-> Sinkhorn_eps(s)``.

    The Sinkhorn map ``R^{V×K} -> Pi(mu, nu)`` is rank-deficient on the ambient space (marginal constraints),
    so a full SLQ logdet is ``-inf``. This estimator draws a random ``q``-dimensional subspace, forms
    finite-difference Jacobian-vector products ``J Q`` via the NumPy Sinkhorn solver, and applies
    :func:`slq_logdet` to the regularized Gram operator ``Q^T J^T J Q + ridge I`` on that subspace. The
    subspace logdet is rescaled by ``dim/q`` as a crude full-space volume estimate and clipped to
    ``[-clip, clip]`` for training stability.

    Returns a finite float64. Computation is NumPy / CPU only; callers should treat the result as a detached
    constant in any torch ELBO (pathwise differentiation through this estimator is not supported).
    """
    from trajot.inference.sinkhorn import sinkhorn_log

    score = np.asarray(score, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    V, K = score.shape
    dim = V * K
    q = int(max(1, min(subspace_dim, dim)))
    generator = generator or np.random.default_rng(0)

    basis = generator.normal(size=(dim, q))
    basis, _ = np.linalg.qr(basis)
    columns = []
    for j in range(q):
        direction = basis[:, j].reshape(V, K)
        plus, _ = sinkhorn_log(score + fd_step * direction, mu, nu, eps, n_iter=n_iter, tol=0.0)
        minus, _ = sinkhorn_log(score - fd_step * direction, mu, nu, eps, n_iter=n_iter, tol=0.0)
        columns.append(((plus - minus) / (2.0 * fd_step)).ravel())
    jq = np.stack(columns, axis=1)
    gram = jq.T @ jq + ridge * np.eye(q)

    def matvec(v: np.ndarray) -> np.ndarray:
        return gram @ v

    logdet_gram = slq_logdet(matvec, q, n_probes=2, n_lanczos=min(q, 10), generator=generator)
    logdet = 0.5 * (dim / q) * logdet_gram
    return float(np.clip(logdet, -clip, clip))
