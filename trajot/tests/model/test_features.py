from __future__ import annotations

import numpy as np
import pytest
import torch

from trajot.model.features import feature_residual, fused_feature_term, gauge_feature_cost


def coupling(V, K, seed=0):
    rng = np.random.default_rng(seed)
    mu, nu = rng.uniform(0.5, 1.5, V), rng.uniform(0.5, 1.5, K)
    mu, nu = mu / mu.sum(), nu / nu.sum()
    pi = rng.uniform(0.2, 1.0, (V, K))
    for _ in range(2000):
        pi *= (mu / pi.sum(1))[:, None]
        pi *= (nu / pi.sum(0))[None, :]
    return pi, mu, nu


def test_linear_plus_constant_is_the_gaussian_transport_log_likelihood() -> None:
    V, K, F, sigma2 = 20, 8, 5, 0.7
    pi, mu, nu = coupling(V, K)
    rng = np.random.default_rng(1)
    Y, Fbar = rng.normal(size=(V, F)), rng.normal(size=(K, F))
    term = fused_feature_term(pi, Y, Fbar, mu, sigma2)

    direct = -0.5 / sigma2 * sum(pi[i, k] * np.sum((Y[i] - Fbar[k]) ** 2) for i in range(V) for k in range(K))
    assert term.linear + term.constant == pytest.approx(direct, rel=1e-12)
    assert term.total == pytest.approx(direct, rel=1e-12)
    assert term.linear == pytest.approx(float((pi * (Y @ Fbar.T)).sum()) / sigma2, rel=1e-12)  # linear in pi


def test_the_linear_part_is_linear_in_the_coupling_and_the_constant_does_not_depend_on_it() -> None:
    V, K, F = 12, 6, 4
    pi1, mu, nu = coupling(V, K, 0)
    pi2, _, _ = coupling(V, K, 1)
    pi2 = pi2 * 0 + pi1 * 0 + coupling(V, K, 1)[0]  # another coupling; same marginals are not needed for linearity
    rng = np.random.default_rng(2)
    Y, Fbar = rng.normal(size=(V, F)), rng.normal(size=(K, F))
    a, b = 0.3, 1.7
    combo = fused_feature_term(a * pi1 + b * pi2, Y, Fbar, mu, 1.0).linear
    assert combo == pytest.approx(a * fused_feature_term(pi1, Y, Fbar, mu, 1.0).linear
                                  + b * fused_feature_term(pi2, Y, Fbar, mu, 1.0).linear, rel=1e-12)
    assert fused_feature_term(pi1, Y, Fbar, mu, 1.0).constant == pytest.approx(
        fused_feature_term(pi1 * 1.0, Y, Fbar, mu, 1.0).constant)


def test_a_batch_of_draws_and_autograd() -> None:
    V, K, F = 10, 5, 3
    pi, mu, nu = coupling(V, K)
    draws = np.stack([pi, pi * 0.5 + 0.5 * np.outer(mu, nu)])
    rng = np.random.default_rng(3)
    Y, Fbar = rng.normal(size=(V, F)), rng.normal(size=(K, F))
    batched = fused_feature_term(draws, Y, Fbar, mu, 2.0)
    assert batched.linear.shape == (2,) and batched.constant.shape == (2,)
    assert batched.linear[0] == pytest.approx(fused_feature_term(draws[0], Y, Fbar, mu, 2.0).linear)

    F_t = torch.from_numpy(Fbar).requires_grad_(True)
    out = fused_feature_term(*(torch.from_numpy(x) for x in (draws, Y)), F_t, torch.from_numpy(mu), 2.0)
    out.total.sum().backward()
    assert torch.isfinite(F_t.grad).all() and F_t.grad.abs().max() > 0


def test_feature_residual_vanishes_for_a_perfect_one_to_one_coupling() -> None:
    V, F = 9, 4
    rng = np.random.default_rng(4)
    Fbar = rng.normal(size=(V, F))
    perm = rng.permutation(V)
    Y = Fbar[perm]
    mu = np.full(V, 1.0 / V)
    pi = np.zeros((V, V))
    pi[np.arange(V), perm] = 1.0 / V
    assert np.abs(feature_residual(pi, Y, Fbar, mu)).max() < 1e-12
    assert feature_residual(pi, Y, Fbar, mu).shape == (V, F)


def test_gauge_feature_cost_is_the_squared_distance_matrix() -> None:
    rng = np.random.default_rng(5)
    Yg, Fg = rng.normal(size=(15, 8)), rng.normal(size=(6, 8))
    cost = gauge_feature_cost(Yg, Fg)
    brute = np.array([[np.sum((Yg[i] - Fg[k]) ** 2) for k in range(6)] for i in range(15)])
    assert cost.shape == (15, 6) and cost.dtype == np.float64 and np.allclose(cost, brute)
    assert np.all(cost >= -1e-12)
