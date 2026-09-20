from __future__ import annotations

import numpy as np
import pytest
import torch

from trajot.inference.sinkhorn import marginal_error, perturb_then_project, sinkhorn_log, sinkhorn_log_torch


def problem(V=100, K=50, scale=0.3, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.5, 1.5, V)
    mu /= mu.sum()
    nu = rng.uniform(0.5, 1.5, K)
    nu /= nu.sum()
    return rng.normal(size=(V, K)) * scale, mu, nu


def test_marginals_are_within_1e_minus_4_of_mu_and_nu_at_L_30() -> None:
    S, mu, nu = problem()
    pi, info = sinkhorn_log(S, mu, nu, eps=0.1, n_iter=30, tol=0.0)
    assert pi.shape == S.shape and pi.dtype == np.float64 and (pi >= 0).all()
    assert info["n_iter"] == 30
    row, col = marginal_error(pi, mu, nu)
    assert row < 1e-4 and col < 1e-4
    assert (row, col) == (info["row_error"], info["col_error"])


def test_early_exit_and_the_reported_iteration_count() -> None:
    S, mu, nu = problem()
    _, loose = sinkhorn_log(S, mu, nu, eps=0.1, n_iter=30, tol=1e-3)
    _, tight = sinkhorn_log(S, mu, nu, eps=0.1, n_iter=30, tol=1e-9)
    assert loose["n_iter"] < tight["n_iter"] <= 30
    assert max(loose["row_error"], loose["col_error"]) <= 1e-3


def test_one_marginal_is_exact_and_the_other_converges_with_iterations() -> None:
    """The last update fixes the columns exactly; the row error is what L controls (and it grows as eps shrinks)."""
    S, mu, nu = problem(scale=1.0)
    errors = [sinkhorn_log(S, mu, nu, 0.1, n_iter=n, tol=0.0)[1] for n in (2, 8, 30, 120)]
    assert all(e["col_error"] < 1e-15 for e in errors)
    rows = [e["row_error"] for e in errors]
    assert rows == sorted(rows, reverse=True) and rows[-1] < 1e-6 < rows[0]


def test_it_stays_finite_at_very_small_eps_with_large_scores() -> None:
    S, mu, nu = problem(scale=5.0)
    pi, info = sinkhorn_log(S, mu, nu, eps=1e-3, n_iter=30, tol=0.0)
    assert np.isfinite(pi).all() and np.isfinite(info["row_error"]) and info["col_error"] < 1e-12


def test_zero_mass_vertices_and_nodes_get_exactly_zero_coupling() -> None:
    S, mu, nu = problem(V=20, K=10)
    mu[3], nu[2] = 0.0, 0.0
    mu, nu = mu / mu.sum(), nu / nu.sum()
    pi, _ = sinkhorn_log(S, mu, nu, 0.1, n_iter=60, tol=0.0)
    assert np.all(pi[3] == 0) and np.all(pi[:, 2] == 0) and np.isfinite(pi).all()


def test_torch_version_matches_numpy_and_supports_a_batch_of_draws() -> None:
    S, mu, nu = problem(V=40, K=20)
    expected, _ = sinkhorn_log(S, mu, nu, 0.1, n_iter=30, tol=0.0)
    got = sinkhorn_log_torch(torch.from_numpy(S), torch.from_numpy(mu), torch.from_numpy(nu), 0.1, n_iter=30)
    assert got.dtype == torch.float64 and np.allclose(got.numpy(), expected, atol=1e-12)

    batch = np.stack([S, S * 0.5, -S])
    got_batch = sinkhorn_log_torch(torch.from_numpy(batch), torch.from_numpy(mu), torch.from_numpy(nu), 0.1, 30)
    assert got_batch.shape == (3, 40, 20)
    for m in range(3):
        assert np.allclose(got_batch[m].numpy(), sinkhorn_log(batch[m], mu, nu, 0.1, 30, 0.0)[0], atol=1e-12)


def test_gradients_reach_the_scores_and_match_finite_differences() -> None:
    S, mu, nu = problem(V=12, K=6, scale=0.5)
    weights = np.random.default_rng(3).normal(size=S.shape)
    mu_t, nu_t, w_t = torch.from_numpy(mu), torch.from_numpy(nu), torch.from_numpy(weights)

    S_t = torch.from_numpy(S).requires_grad_(True)
    loss = (sinkhorn_log_torch(S_t, mu_t, nu_t, 0.1, n_iter=30) * w_t).sum()
    loss.backward()
    assert S_t.grad is not None and torch.isfinite(S_t.grad).all() and S_t.grad.abs().max() > 0

    def value(x: np.ndarray) -> float:
        return float((sinkhorn_log(x, mu, nu, 0.1, n_iter=30, tol=0.0)[0] * weights).sum())

    h = 1e-6
    for i, k in [(0, 0), (3, 2), (11, 5), (7, 1)]:
        bump = np.zeros_like(S)
        bump[i, k] = h
        assert (value(S + bump) - value(S - bump)) / (2 * h) == pytest.approx(float(S_t.grad[i, k]), rel=1e-5, abs=1e-9)


def test_perturb_then_project_draws_lie_on_the_polytope_and_are_reparametrized() -> None:
    S, mu, nu = problem(V=60, K=30, scale=0.3, seed=1)
    tau = np.full(60, 0.05)
    xi = np.random.default_rng(4).normal(size=(4, 60, 30))
    draws = perturb_then_project(S.astype(np.float32), tau, xi, mu, nu, eps=0.1, L=30)

    assert draws.shape == (4, 60, 30) and draws.dtype == np.float64
    for pi in draws:
        assert max(marginal_error(pi, mu, nu)) < 1e-4  # every posterior draw satisfies the bound
    assert np.array_equal(draws, perturb_then_project(S.astype(np.float32), tau, xi, mu, nu, 0.1, 30))  # a function of xi
    assert not np.allclose(draws[0], draws[1])

    no_noise = perturb_then_project(S.astype(np.float32), np.zeros(60), xi, mu, nu, 0.1, 30)
    assert np.allclose(no_noise[0], no_noise[3])  # tau = 0 leaves only the mean projection


def test_marginal_error() -> None:
    pi = np.array([[0.2, 0.3], [0.1, 0.4]])
    assert marginal_error(pi, np.array([0.6, 0.4]), np.array([0.3, 0.7])) == pytest.approx((0.1, 0.0), abs=1e-12)
