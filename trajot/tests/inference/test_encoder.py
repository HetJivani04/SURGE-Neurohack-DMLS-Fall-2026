from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from trajot.inference.encoder import ScoreEncoder, SinkhornPosterior, build_inputs
from trajot.inference.sinkhorn import marginal_error

CPU = torch.device("cpu")


def test_build_inputs_layout_dtype_and_masking() -> None:
    rng = np.random.default_rng(0)
    V, T, d, m = 30, 20, 6, 4
    ts = rng.normal(size=(V, T))
    ts[5] = 0.0  # a vertex outside the field of view
    emb, coords, lap = rng.normal(size=(V, d)), rng.normal(size=(V, 3)) * 30, rng.normal(size=(V, m))
    u = build_inputs(ts, emb, coords, lap)

    assert u.shape == (V, d + m + 2) and u.dtype == np.float32
    keep = np.arange(V) != 5
    assert np.allclose(u[keep, :d], emb[keep], atol=1e-6) and np.allclose(u[keep, d : d + m], lap[keep], atol=1e-6)
    theta, phi = u[:, -2], u[:, -1]
    assert theta.min() >= 0 and theta.max() <= np.pi and phi.min() >= -np.pi and phi.max() <= np.pi
    assert np.all(u[5] == 0) and np.all(u[4] != 0)

    with_features = build_inputs(ts, emb, coords, lap, features=rng.normal(size=(V, 2)))
    assert with_features.shape == (V, d + 2 + m + 2)
    with pytest.raises(ValueError):
        build_inputs(ts[:-1], emb, coords, lap)


def test_spherical_angles_of_known_points() -> None:
    coords = np.array([[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]])  # centroid 0
    timeseries = np.random.default_rng(0).normal(size=(6, 5))
    u = build_inputs(timeseries, np.zeros((6, 1)), coords, np.zeros((6, 1)))
    assert u[0, -2] == pytest.approx(0.0, abs=1e-6) and u[1, -2] == pytest.approx(np.pi, abs=1e-6)  # theta: polar angle
    assert u[2, -2] == pytest.approx(np.pi / 2, abs=1e-6)
    assert u[2, -1] == pytest.approx(0.0, abs=1e-6) and u[4, -1] == pytest.approx(np.pi / 2, abs=1e-6)  # phi: azimuth
    assert u[3, -1] == pytest.approx(np.pi, abs=1e-6) and u[5, -1] == pytest.approx(-np.pi / 2, abs=1e-6)


def test_encoder_outputs_shapes_dtypes_and_positive_tau() -> None:
    enc = ScoreEncoder(in_dim=12, p=32, K=10, device=CPU)
    S, tau = enc(torch.randn(50, 12))
    assert S.shape == (50, 10) and tau.shape == (50,) and S.dtype == tau.dtype == torch.float32
    assert (tau > 0).all() and torch.isfinite(S).all()


def test_the_default_encoder_has_about_1_2_million_parameters() -> None:
    enc = ScoreEncoder(in_dim=44, p=152, K=512, n_blocks=4, device=CPU)
    count = sum(p.numel() for p in enc.parameters())
    assert 1.1e6 < count < 1.35e6, count


def test_the_score_is_a_bilinear_form_minus_lambda_times_the_anatomical_cost() -> None:
    enc = ScoreEncoder(in_dim=8, p=16, K=6, device=CPU, lambda_init=0.7)
    u, M0 = torch.randn(20, 8), torch.rand(20, 6)
    S_plain, tau_plain = enc(u)
    S_anat, tau_anat = enc(u, M0)
    assert torch.allclose(S_anat, S_plain - 0.7 * M0, atol=1e-6) and torch.equal(tau_plain, tau_anat)

    h = enc.norm(enc.blocks[-1](torch.zeros(1, 16)))  # sanity: the head is W h against the node embeddings g_k
    assert enc.g.shape == (6, 16) and enc.W.weight.shape == (16, 16) and h.shape == (1, 16)


def test_the_encoder_is_permutation_equivariant_over_vertices() -> None:
    torch.manual_seed(0)
    enc = ScoreEncoder(in_dim=8, p=16, K=6, device=CPU)
    u = torch.randn(40, 8)
    perm = torch.randperm(40)
    S, tau = enc(u)
    S_perm, tau_perm = enc(u[perm])
    assert torch.allclose(S[perm], S_perm, atol=1e-5) and torch.allclose(tau[perm], tau_perm, atol=1e-5)


def test_gradients_reach_every_parameter() -> None:
    enc = ScoreEncoder(in_dim=8, p=16, K=6, device=CPU)
    S, tau = enc(torch.randn(25, 8), torch.rand(25, 6))
    (S.sum() + tau.sum()).backward()
    missing = [name for name, p in enc.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert missing == []


def test_ten_thousand_vertex_tokens_without_a_v_by_v_matrix() -> None:
    enc = ScoreEncoder(in_dim=44, p=64, K=32, device=CPU)
    V = 10000
    start = time.time()
    with torch.no_grad():
        S, tau = enc(torch.randn(V, 44))
    assert S.shape == (V, 32) and tau.shape == (V,)
    assert time.time() - start < 30  # linear in V; a dense attention matrix would be 800 MB in float64


# ---- the posterior --------------------------------------------------------------------------------------
def posterior_inputs(V=40, K=20, seed=0):
    rng = np.random.default_rng(seed)
    mu, nu = np.full(V, 1.0 / V), np.full(K, 1.0 / K)
    return rng.normal(size=(V, K)).astype(np.float32) * 0.3, np.full(V, 0.05, dtype=np.float32), mu, nu


def test_sample_returns_float64_draws_on_the_polytope() -> None:
    S, tau, mu, nu = posterior_inputs()
    draws = SinkhornPosterior().sample(S, tau, mu, nu, eps=0.1, M=4, L=30, generator=np.random.default_rng(1))
    assert draws.shape == (4, 40, 20) and draws.dtype == np.float64
    assert all(max(marginal_error(pi, mu, nu)) < 1e-4 for pi in draws)
    again = SinkhornPosterior().sample(S, tau, mu, nu, 0.1, 4, 30, np.random.default_rng(1))
    assert np.array_equal(draws, again)


def test_sample_torch_is_float64_differentiable_and_matches_the_numpy_operator() -> None:
    S, tau, mu, nu = posterior_inputs(V=15, K=8)
    S_t, tau_t = torch.from_numpy(S).requires_grad_(True), torch.from_numpy(tau).requires_grad_(True)
    pi, xi = SinkhornPosterior().sample_torch(S_t, tau_t, torch.from_numpy(mu), torch.from_numpy(nu), 0.1, M=3, L=30,
                                              generator=torch.Generator().manual_seed(0))
    assert pi.dtype == torch.float64 and pi.shape == (3, 15, 8) and xi.shape == (3, 15, 8)

    from trajot.inference.sinkhorn import perturb_then_project

    expected = perturb_then_project(S, tau, xi.numpy(), mu, nu, 0.1, 30)
    assert np.allclose(pi.data.numpy(), expected, atol=1e-10)  # the same operator as the NumPy sampler

    (pi * torch.randn_like(pi)).sum().backward()
    assert S_t.grad is not None and tau_t.grad is not None
    assert torch.isfinite(S_t.grad).all() and torch.isfinite(tau_t.grad).all() and tau_t.grad.abs().max() > 0


def test_sample_torch_promotes_float32_scores_without_the_mps_combined_cast() -> None:
    """CPU float32 stands in for the MPS promotion path: combined .to(cpu, float64) is unsafe on MPS."""
    S, tau, mu, nu = posterior_inputs(V=12, K=6)
    assert S.dtype == np.float32
    S_t = torch.from_numpy(S).requires_grad_(True)
    tau_t = torch.from_numpy(tau).requires_grad_(True)
    mu_t, nu_t = torch.from_numpy(mu).to(torch.float64), torch.from_numpy(nu).to(torch.float64)
    pi, xi = SinkhornPosterior().sample_torch(
        S_t, tau_t, mu_t, nu_t, 0.1, M=2, L=20, generator=torch.Generator().manual_seed(3))
    assert pi.dtype == torch.float64 and pi.device.type == "cpu" and torch.isfinite(pi).all()
    pi.sum().backward()
    assert S_t.grad is not None and tau_t.grad is not None
    assert torch.isfinite(S_t.grad).all() and S_t.grad.abs().max() > 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_sample_torch_works_when_scores_live_on_mps() -> None:
    S, tau, mu, nu = posterior_inputs(V=10, K=5)
    device = torch.device("mps")
    S_t = torch.from_numpy(S).to(device).requires_grad_(True)
    tau_t = torch.from_numpy(tau).to(device).requires_grad_(True)
    mu_t = torch.from_numpy(mu).to(torch.float64)  # OT masses stay CPU float64
    nu_t = torch.from_numpy(nu).to(torch.float64)
    pi, xi = SinkhornPosterior().sample_torch(
        S_t, tau_t, mu_t, nu_t, 0.1, M=2, L=15, generator=torch.Generator().manual_seed(0))
    assert pi.dtype == torch.float64 and pi.device.type == "cpu" and torch.isfinite(pi).all()
    pi.sum().backward()
    assert S_t.grad is not None and torch.isfinite(S_t.grad.cpu()).all()



def test_log_prob_of_the_perturbation_is_the_gaussian_density() -> None:
    S, tau, mu, nu = posterior_inputs(V=6, K=4)
    xi = np.random.default_rng(2).normal(size=(3, 6, 4))
    tau64 = np.linspace(0.05, 0.4, 6)
    expected = (-0.5 * (xi**2).sum((1, 2)) - 0.5 * 24 * np.log(2 * np.pi)) - 4 * np.log(tau64).sum()
    assert np.allclose(SinkhornPosterior().log_prob(tau64, xi), expected)
    assert np.allclose(SinkhornPosterior().log_prob(torch.from_numpy(tau64), torch.from_numpy(xi)).numpy(), expected)
