from __future__ import annotations

import numpy as np
import pytest

from trajot.inference.entropy import entropy_estimator, hutchinson_logdet, slq_logdet


def spd(n: int, spread: float, seed: int = 0) -> np.ndarray:
    G = np.random.default_rng(seed).normal(size=(n, n)) / np.sqrt(n)
    return np.eye(n) + spread * (G + G.T) / 2  # eigenvalues within 1 +- spread * ~1


def test_hutchinson_is_an_unbiased_trace_estimator() -> None:
    A = spd(40, 0.4)
    rng = np.random.default_rng(1)
    many = hutchinson_logdet(lambda v: A @ v, 40, n_probes=4000, generator=rng)
    assert many == pytest.approx(np.trace(A), rel=0.01)
    assert np.mean([hutchinson_logdet(lambda v: A @ v, 40, 4, np.random.default_rng(s)) for s in range(300)]) == pytest.approx(
        np.trace(A), rel=0.01)


def single_probe_std(A: np.ndarray) -> float:
    """Std of one Rademacher probe's estimate of tr(log A): sqrt(2 * sum_{i != j} log(A)_ij^2)."""
    w, V = np.linalg.eigh(A)
    L = V @ np.diag(np.log(w)) @ V.T
    return float(np.sqrt(2 * ((L**2).sum() - (np.diag(L) ** 2).sum())))


def test_slq_is_unbiased_and_its_error_matches_the_hutchinson_probe_variance() -> None:
    A = spd(30, 0.4)
    truth = np.linalg.slogdet(A)[1]
    std = single_probe_std(A)

    def run(probes: int, lanczos: int, seed: int) -> float:
        return slq_logdet(lambda v: A @ v, 30, n_probes=probes, n_lanczos=lanczos, generator=np.random.default_rng(seed))

    # 30 Lanczos steps on a 30-dimensional operator: the quadrature is exact, only the probes add error
    assert abs(run(6, 30, 0) - truth) < 4 * std / np.sqrt(6)
    assert abs(run(400, 20, 1) - truth) < 4 * std / np.sqrt(400)
    estimates = np.array([run(4, 20, s) for s in range(150)])  # the stated default: 4 probes, 20 Lanczos steps
    assert abs(estimates.mean() - truth) < 4 * std / np.sqrt(4 * 150)  # unbiased
    assert estimates.std() == pytest.approx(std / np.sqrt(4), rel=0.25)  # and the spread is the probe variance


def test_slq_is_exact_on_a_diagonal_operator_with_enough_lanczos_steps() -> None:
    d = np.linspace(0.5, 3.0, 12)
    # every Rademacher probe has ||z||^2 = dim and z^T log(D) z = sum log d (z_i^2 = 1): zero probe variance
    assert slq_logdet(lambda v: d * v, 12, n_probes=1, n_lanczos=12, generator=np.random.default_rng(2)) == pytest.approx(
        np.log(d).sum(), rel=1e-10)


def test_entropy_estimator_reproduces_the_analytic_gaussian_entropy_within_its_probe_count() -> None:
    """pi = A xi + b, xi ~ N(0, I): H = (n/2)(1 + log 2 pi) + log |det A|; log |det J| = 0.5 logdet(A^T A) by SLQ."""
    n, n_probes = 30, 4
    A = spd(n, 0.25, seed=3)
    truth = 0.5 * n * (1 + np.log(2 * np.pi)) + np.linalg.slogdet(A)[1]

    rng = np.random.default_rng(4)
    N = 20000
    xi = rng.normal(size=(N, n))
    log_q_xi = -0.5 * (xi**2).sum(1) - 0.5 * n * np.log(2 * np.pi)
    gram = lambda v: A.T @ (A @ v)  # noqa: E731
    logdet = 0.5 * slq_logdet(gram, n, n_probes=n_probes, n_lanczos=20, generator=np.random.default_rng(5))
    estimate = entropy_estimator(log_q_xi, np.full(N, logdet))

    # the estimator's own sampling error: Monte Carlo over xi plus the Hutchinson probe variance of 0.5 tr(log A^T A)
    probe_std = 0.5 * single_probe_std(A.T @ A) / np.sqrt(n_probes)
    mc_std = np.sqrt(n / 2 / N)
    assert abs(estimate - truth) < 3 * np.hypot(probe_std, mc_std)

    many = 0.5 * slq_logdet(gram, n, n_probes=200, n_lanczos=20, generator=np.random.default_rng(6))
    assert abs(entropy_estimator(log_q_xi, np.full(N, many)) - truth) / truth < 0.01  # more probes: within 1%


def test_entropy_estimator_combines_the_two_terms() -> None:
    assert entropy_estimator(np.array([-1.0, -3.0]), np.array([0.5, 1.5])) == pytest.approx(2.0 + 1.0)
