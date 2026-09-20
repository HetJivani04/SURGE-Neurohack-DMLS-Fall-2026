from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy import stats

from trajot.config import Config
from trajot.model import prior
from trajot.model.prior import (
    band_penalty_matrix, band_prior, constrain_scale, dominant_frequency, gibbs_prior, log_prior_theta,
)

DT = 2.5
CFG = Config({"model": {"prior": {"sigma_B": 0.8, "sigma_F": 1.3, "F0": 0.2, "a_eps": 3.0, "b_eps": 0.4}}})


def test_gibbs_prior_is_minus_the_anatomical_cost_over_epsilon() -> None:
    rng = np.random.default_rng(0)
    pi, M = rng.uniform(size=(6, 4)), rng.uniform(size=(6, 4))
    assert gibbs_prior(pi, M, 0.25) == pytest.approx(-float((pi * M).sum()) / 0.25)
    assert gibbs_prior(pi, M, 0.5) == pytest.approx(0.5 * gibbs_prior(pi, M, 0.25))  # sharper eps, stronger prior
    assert gibbs_prior(np.stack([pi, 2 * pi]), M, 1.0).shape == (2,)


def test_dominant_frequency_is_in_hz_for_a_sampling_interval_in_seconds() -> None:
    n = 40
    t = np.arange(n) * DT
    rows = np.stack([np.sin(2 * np.pi * 0.05 * t), np.sin(2 * np.pi * 0.1 * t + 1.0), 3 + np.cos(2 * np.pi * 0.03 * t)])
    assert np.allclose(dominant_frequency(rows, DT), [0.05, 0.1, 0.03])  # FFT bins are k / (n * dt) = 0.01 Hz apart
    assert np.allclose(dominant_frequency(torch.from_numpy(rows), DT).numpy(), [0.05, 0.1, 0.03])


def rows_with_period(periods_seconds, n=40):
    t = np.arange(n) * DT
    return np.stack([np.sin(2 * np.pi * t / p) for p in periods_seconds])


def test_band_penalty_is_zero_inside_the_band_and_scales_with_the_frequency_gap_outside_it() -> None:
    band = (10.0, 100.0)  # seconds, i.e. 0.01 to 0.1 Hz
    A = rows_with_period([20.0, 40.0, 5.0])  # 0.05 Hz, 0.025 Hz (both in band), 0.2 Hz (out)
    Abar = rows_with_period([20.0, 10.0, 5.0, 2.5])  # 0.05, 0.1 (band edge, in), 0.2, 0.4 Hz
    penalty = band_penalty_matrix(A, Abar, DT, band, bandwidth=0.05)
    f = dominant_frequency(A, DT)
    g = dominant_frequency(Abar, DT)
    assert penalty.shape == (3, 4)
    assert penalty[0, 0] == 0 and penalty[1, 0] == 0  # both in band
    assert penalty[0, 2] == pytest.approx(abs(f[0] - g[2]) / 0.05)  # one out of band
    assert penalty[2, 2] == 0  # both out of band at the same frequency: no gap
    assert penalty[2, 3] == pytest.approx(abs(f[2] - g[3]) / 0.05)
    assert np.all(penalty >= 0)
    half = band_penalty_matrix(A, Abar, DT, band, bandwidth=0.1)
    assert np.allclose(half, penalty / 2)  # inversely proportional to the bandwidth


def test_band_edges_are_converted_from_seconds_to_hz() -> None:
    """A row at 0.08 Hz is in band for [10, 100] s and out of band for [15, 100] s (0.0667 Hz upper edge)."""
    A = rows_with_period([12.5])  # 0.08 Hz
    Abar = rows_with_period([40.0])  # 0.025 Hz, in both bands
    assert band_penalty_matrix(A, Abar, DT, (10.0, 100.0), 1.0)[0, 0] == 0
    assert band_penalty_matrix(A, Abar, DT, (15.0, 100.0), 1.0)[0, 0] > 0


def test_band_prior_is_the_expected_penalty_under_the_coupling() -> None:
    A, Abar = rows_with_period([20.0, 5.0, 40.0]), rows_with_period([20.0, 5.0])
    pi = np.random.default_rng(1).uniform(size=(3, 2))
    penalty = band_penalty_matrix(A, Abar, DT, (10.0, 100.0), 0.05)
    assert band_prior(pi, A, Abar, DT, (10.0, 100.0), 0.05) == pytest.approx(float((pi * penalty).sum()))


def test_the_band_prior_is_documented_as_a_band_prior_and_not_dynamics() -> None:
    first_line = inspect.getdoc(band_prior).splitlines()[0].lower()
    assert "band prior" in first_line and "not a model of temporal dynamics" in first_line


def test_constrain_scale_projects_rows_onto_a_ball() -> None:
    B = np.array([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]])
    out = constrain_scale(B, 1.0)
    assert np.allclose(np.linalg.norm(out, axis=1), [1.0, 0.5, 0.0])
    assert np.allclose(out[1], B[1]) and np.allclose(out[0], [0.6, 0.8])
    assert torch.allclose(constrain_scale(torch.from_numpy(B), 1.0), torch.from_numpy(out))


def test_log_prior_theta_matches_scipy_for_numpy_and_torch() -> None:
    rng = np.random.default_rng(2)
    B, F, eps = rng.normal(size=(7, 3)), rng.normal(size=(7, 5)), rng.uniform(0.05, 0.5, 4)
    expected = (stats.norm.logpdf(B, 0.0, 0.8).sum() + stats.norm.logpdf(F, 0.2, 1.3).sum()
                + stats.invgamma.logpdf(eps, a=3.0, scale=0.4).sum())
    assert log_prior_theta(SimpleNamespace(B=B, F_bar=F, eps=eps), CFG) == pytest.approx(expected, rel=1e-12)
    as_torch = SimpleNamespace(B=torch.from_numpy(B), F_bar=torch.from_numpy(F), eps=torch.from_numpy(eps))
    assert float(log_prior_theta(as_torch, CFG)) == pytest.approx(expected, rel=1e-12)


def test_log_prior_theta_is_differentiable() -> None:
    B = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    F = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    eps = torch.full((3,), 0.2, dtype=torch.float64, requires_grad=True)
    log_prior_theta(SimpleNamespace(B=B, F_bar=F, eps=eps), CFG).backward()
    assert all(torch.isfinite(t.grad).all() for t in (B, F, eps))


def test_no_docstring_or_comment_in_model_describes_the_band_prior_as_dynamics() -> None:
    """Grep for 'dynamic' and 'trajectory' in model/: every hit must be a negation ('not a model of ... dynamics')."""
    import pathlib

    for path in pathlib.Path(prior.__file__).parent.glob("*.py"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            lowered = line.lower()
            if "dynamic" in lowered or "trajectory" in lowered or "trajectories" in lowered:
                assert "not a model of temporal dynamics" in lowered or "says nothing about trajectories" in lowered \
                    or "never" in lowered, f"{path.name}:{number}: {line}"
