from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
import torch

from trajot.model.gw import gw_cross, gw_gradient, gw_reference_bruteforce, gw_term


def polytope_coupling(mu: np.ndarray, nu: np.ndarray, seed: int, iters: int = 3000) -> np.ndarray:
    """A strictly positive coupling with exactly the marginals (iterative proportional fitting)."""
    pi = np.random.default_rng(seed).uniform(0.2, 1.0, size=(mu.size, nu.size))
    for _ in range(iters):
        pi *= (mu / pi.sum(1))[:, None]
        pi *= (nu / pi.sum(0))[None, :]
    return pi


def make_problem(V: int, K: int, r: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(V, r)) * 0.4
    B = rng.normal(size=(K, r)) * 0.4
    mu = rng.uniform(0.5, 1.5, V)
    mu /= mu.sum()
    nu = rng.uniform(0.5, 1.5, K)
    nu /= nu.sum()
    return A, B, mu, nu, polytope_coupling(mu, nu, seed + 1)


def test_gw_term_matches_the_four_index_sum_to_1e_minus_8() -> None:
    A, B, mu, nu, pi = make_problem(64, 16)
    assert np.abs(pi.sum(1) - mu).max() < 1e-13 and np.abs(pi.sum(0) - nu).max() < 1e-13
    reference = gw_reference_bruteforce(pi, A @ A.T, B @ B.T)
    assert reference > 1e-3  # not a vacuous comparison
    assert abs(gw_term(pi, A, B, nu, mu) - reference) / reference < 1e-8


def test_gw_term_is_nonnegative_and_zero_for_a_perfect_match() -> None:
    A, _, mu, _, _ = make_problem(20, 20)
    identity_coupling = np.diag(mu)  # template == subject, coupled node to node
    assert gw_term(identity_coupling, A, A, mu, mu) == pytest.approx(0.0, abs=1e-12)
    A2, B2, mu2, nu2, pi2 = make_problem(20, 12, seed=4)
    assert gw_term(pi2, A2, B2, nu2, mu2) > 0.0


def test_gw_cross_is_the_dense_product_computed_through_an_r_dimensional_bottleneck() -> None:
    A, B, mu, nu, pi = make_problem(30, 10)
    assert np.allclose(gw_cross(pi, A, B), (A @ A.T) @ pi @ (B @ B.T), atol=1e-12)
    assert gw_cross(pi, A, B).shape == (30, 10) and gw_cross(pi, A, B).dtype == np.float64


def test_gw_term_returns_a_python_float_for_numpy_and_a_tensor_for_torch() -> None:
    A, B, mu, nu, pi = make_problem(16, 6)
    assert isinstance(gw_term(pi, A, B, nu, mu), float)
    value = gw_term(*(torch.from_numpy(x) for x in (pi, A, B, nu, mu)))
    assert isinstance(value, torch.Tensor) and value.dtype == torch.float64 and value.ndim == 0
    assert float(value) == pytest.approx(gw_term(pi, A, B, nu, mu), rel=1e-12)


def test_gw_gradient_matches_torch_autograd() -> None:
    A, B, mu, nu, pi = make_problem(32, 8)
    pi_t = torch.from_numpy(pi).requires_grad_(True)
    gw_term(pi_t, *(torch.from_numpy(x) for x in (A, B, nu, mu))).backward()
    assert np.allclose(pi_t.grad.numpy(), gw_gradient(pi, A, B, nu, mu), atol=1e-12)


def test_gw_gradient_matches_finite_differences() -> None:
    A, B, mu, nu, pi = make_problem(32, 8)
    grad = gw_gradient(pi, A, B, nu, mu)
    rng = np.random.default_rng(5)
    h = 1e-6
    for _ in range(6):
        direction = rng.normal(size=pi.shape)
        numeric = (gw_term(pi + h * direction, A, B, nu, mu) - gw_term(pi - h * direction, A, B, nu, mu)) / (2 * h)
        assert numeric == pytest.approx(float((grad * direction).sum()), rel=1e-6)


def test_gw_gradient_matches_the_brute_force_energy_along_feasible_directions() -> None:
    """Along directions that keep both marginals fixed, the analytic gradient is the true gradient of E_GW."""
    A, B, mu, nu, pi = make_problem(24, 6)
    rng = np.random.default_rng(6)
    raw = rng.normal(size=pi.shape)
    direction = raw - raw.mean(1, keepdims=True) - raw.mean(0, keepdims=True) + raw.mean()  # zero row and column sums
    assert np.abs(direction.sum(0)).max() < 1e-12 and np.abs(direction.sum(1)).max() < 1e-12
    h = 1e-6
    C_s, C_bar = A @ A.T, B @ B.T
    numeric = (gw_reference_bruteforce(pi + h * direction, C_s, C_bar)
               - gw_reference_bruteforce(pi - h * direction, C_s, C_bar)) / (2 * h)
    assert numeric == pytest.approx(float((gw_gradient(pi, A, B, nu, mu) * direction).sum()), rel=1e-6)


def test_peak_memory_does_not_scale_with_v_squared() -> None:
    """At V = 3000 a dense (V, V) float64 matrix is 72 MB; the factored evaluation stays far below that."""
    V, K, r = 3000, 32, 16
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(V, r)), rng.normal(size=(K, r))
    mu, nu = np.full(V, 1.0 / V), np.full(K, 1.0 / K)
    pi = np.outer(mu, nu)
    tracemalloc.start()
    tracemalloc.reset_peak()
    gw_term(pi, A, B, nu, mu)
    gw_gradient(pi, A, B, nu, mu)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 0.25 * V * V * 8  # well under one dense (V, V) matrix
    assert peak < 20 * V * K * 8  # and a small multiple of the (V, K) coupling itself
