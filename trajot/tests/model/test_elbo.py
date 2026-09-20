from __future__ import annotations

import numpy as np
import pytest
import torch

from trajot.model.elbo import beta_schedule, elbo, elbo_terms, eps_schedule
from trajot.model.features import fused_feature_term
from trajot.model.gw import gw_term
from trajot.model.prior import gibbs_prior


def setup(M=3, V=16, K=6, r=4, F=5, seed=0):
    rng = np.random.default_rng(seed)
    A, B = rng.normal(size=(V, r)) * 0.4, rng.normal(size=(K, r)) * 0.4
    mu, nu = np.full(V, 1.0 / V), np.full(K, 1.0 / K)
    draws = np.stack([np.outer(mu, nu) * (1 + 0.3 * rng.uniform(-1, 1, (V, K))) for _ in range(M)])
    for _ in range(300):
        draws *= (mu[None, :] / draws.sum(2))[:, :, None]
        draws *= (nu[None, None, :] / draws.sum(1, keepdims=True))
    return dict(pi=draws, A=A, B=B, nu=nu, mu=mu, Y=rng.normal(size=(V, F)), Fbar=rng.normal(size=(K, F)),
                M0=rng.uniform(size=(V, K)), log_q=rng.normal(size=M))


def terms_for(s, beta=5.0, sigma2=0.5, eps_s=0.3, **kw):
    return elbo_terms(s["pi"], s["A"], s["B"], s["nu"], s["mu"], s["Y"], s["Fbar"], s["M0"], eps_s, sigma2, beta,
                      s["log_q"], **kw)


def test_the_terms_are_the_documented_quantities_and_total_is_their_combination() -> None:
    s = setup()
    t = terms_for(s, log_prior=-3.0)
    assert set(t) == {"gw", "feature", "prior", "entropy", "log_prior", "total"}

    gw = np.mean([gw_term(pi, s["A"], s["B"], s["nu"], s["mu"]) for pi in s["pi"]])
    feature = -np.mean([fused_feature_term(pi, s["Y"], s["Fbar"], s["mu"], 0.5).total for pi in s["pi"]])
    prior = -np.mean([gibbs_prior(pi, s["M0"], 0.3) for pi in s["pi"]])
    assert t["gw"] == pytest.approx(gw) and t["feature"] == pytest.approx(feature) and t["prior"] == pytest.approx(prior)
    assert t["entropy"] == pytest.approx(-s["log_q"].mean()) and t["log_prior"] == -3.0
    assert t["total"] == pytest.approx(-0.5 * 5.0 * gw - feature - prior + t["entropy"] - 3.0)


def test_energies_enter_with_a_minus_sign_so_a_worse_fit_lowers_the_elbo() -> None:
    s = setup()
    base = terms_for(s)["total"]
    assert terms_for(s, beta=10.0)["total"] < base  # larger beta weights the (positive) GW energy more
    assert terms_for(s, eps_s=0.1)["total"] < base  # a sharper prior weighs the (positive) anatomical cost more
    assert terms_for(s, sigma2=0.1)["total"] != base


def test_the_band_penalty_adds_to_the_prior_energy() -> None:
    s = setup()
    penalty = np.random.default_rng(1).uniform(size=s["M0"].shape)
    with_band = terms_for(s, band_penalty=penalty)
    expected = terms_for(s)["prior"] + np.mean([(pi * penalty).sum() for pi in s["pi"]])
    assert with_band["prior"] == pytest.approx(expected)


def test_torch_terms_are_float64_tensors_and_differentiable() -> None:
    s = setup()
    tensors = {k: torch.from_numpy(v) for k, v in s.items()}
    B = tensors["B"].clone().requires_grad_(True)
    Fbar = tensors["Fbar"].clone().requires_grad_(True)
    t = elbo_terms(tensors["pi"], tensors["A"], B, tensors["nu"], tensors["mu"], tensors["Y"], Fbar, tensors["M0"],
                   0.3, 0.5, 5.0, tensors["log_q"], log_prior=-1.0)
    assert all(isinstance(v, torch.Tensor) and v.dtype == torch.float64 for v in t.values())
    t["total"].backward()
    assert torch.isfinite(B.grad).all() and torch.isfinite(Fbar.grad).all() and B.grad.abs().max() > 0

    numpy_total = terms_for(s, log_prior=-1.0)["total"]
    assert t["total"].item() == pytest.approx(numpy_total, rel=1e-12)


def test_elbo_is_the_mean_of_the_per_subject_totals() -> None:
    per_subject = [terms_for(setup(seed=k), log_prior=-2.0) for k in range(3)]
    assert elbo(per_subject) == pytest.approx(np.mean([t["total"] for t in per_subject]))
    as_tensors = [{k: torch.tensor(v, dtype=torch.float64) for k, v in t.items()} for t in per_subject]
    assert float(elbo(as_tensors)) == pytest.approx(float(elbo(per_subject)))


# ---- schedules -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("total", [10, 100, 333, 1000, 12345, 50000])
def test_beta_schedule_equals_beta_target_exactly_at_30_percent_of_the_steps(total: int) -> None:
    target = 137.5
    step = int(round(0.3 * total))
    if abs(step - 0.3 * total) < 1e-9:
        assert beta_schedule(step, total, target) == target  # exactly, not approximately
    assert beta_schedule(step + 1, total, target) == target and beta_schedule(total, total, target) == target
    assert beta_schedule(0, total, target) == 0.0


def test_beta_schedule_is_a_linear_ramp_from_zero() -> None:
    values = [beta_schedule(s, 1000, 10.0) for s in range(0, 301, 50)]
    assert values == pytest.approx([0.0, 10 / 6, 20 / 6, 5.0, 40 / 6, 50 / 6, 10.0])
    assert beta_schedule(150, 1000, 10.0, warmup_frac=0.5) == pytest.approx(3.0)


def test_eps_schedule_hits_both_endpoints_and_is_geometric() -> None:
    assert eps_schedule(0, 1000) == 0.1 and eps_schedule(1000, 1000) == 0.01
    assert eps_schedule(500, 1000) == pytest.approx(np.sqrt(0.1 * 0.01))  # geometric midpoint
    ratios = [eps_schedule(s + 1, 1000) / eps_schedule(s, 1000) for s in (1, 200, 800)]
    assert ratios == pytest.approx([ratios[0]] * 3)  # constant ratio between steps
    values = [eps_schedule(s, 100, 0.5, 0.05) for s in range(101)]
    assert values[0] == 0.5 and values[-1] == 0.05 and values == sorted(values, reverse=True)
