from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trajot.report.group import meta_analysis_map, one_sample_ttest

import importlib.util
import sys

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_group_analysis.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_group_analysis", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_group_analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


driver = _load_driver()


def _coupling(rng: np.random.Generator, V: int, K: int) -> np.ndarray:
    pi = rng.random((V, K)) + 0.05
    for _ in range(200):
        pi *= (1.0 / V) / pi.sum(1, keepdims=True)
        pi *= (1.0 / K) / pi.sum(0, keepdims=True)
    return pi


def _synthetic_draws(
    rng: np.random.Generator,
    S: int = 12,
    M: int = 20,
    V: int = 30,
    K: int = 20,
    n_high_tau: int = 2,
    high_tau: float = 5.0,
    low_tau: float = 0.05,
):
    nu = np.full(K, 1.0 / K)
    pi_list, tau_list, z_list = [], [], []
    for s in range(S):
        base = _coupling(rng, V, K)
        spread = 3.0 if s < n_high_tau else 0.15
        draws = base[None] * np.exp(spread * rng.normal(size=(M, V, K)))
        draws = np.maximum(draws, 1e-12)
        for _ in range(40):
            draws *= (1.0 / V) / draws.sum(axis=2, keepdims=True)
            draws *= (1.0 / K) / draws.sum(axis=1, keepdims=True)
        tau = np.full(V, high_tau if s < n_high_tau else low_tau, dtype=np.float64)
        pi_list.append(draws)
        tau_list.append(tau)
        z_list.append(rng.normal(size=V))
    return pi_list, tau_list, z_list, nu


def test_run_group_from_arrays_downweights_high_tau_subjects():
    rng = np.random.default_rng(0)
    S, M, V, K = 12, 20, 30, 20
    pi_list, tau_list, z_list, nu = _synthetic_draws(rng, S=S, M=M, V=V, K=K)
    out = driver.run_group_from_arrays(pi_list, tau_list, z_list, nu, method="REML")

    assert out["n_subjects"] == S
    assert out["K"] == K
    assert out["n_eff"] < S
    assert out["n_eff_min"] <= out["n_eff"] < S
    weights = np.asarray(out["mean_weight_per_subject"], dtype=np.float64)
    # first two subjects planted with huge tau / coupling variance
    assert weights[:2].mean() < weights[2:].mean()
    assert out["mean_sigma2_per_subject"][0] > out["mean_sigma2_per_subject"][5]
    # REML SE at least as wide as naive t-test on average when alignment noise is heterogeneous
    assert out["ci_ratio"] > 0.0
    assert np.isfinite(out["reml_theta_se"])
    assert np.isfinite(out["ttest_theta_se"])


def test_run_group_from_arrays_ci_ratio_wider_reml_when_sigma2_large():
    rng = np.random.default_rng(1)
    pi_list, tau_list, z_list, nu = _synthetic_draws(rng, S=16, M=20, V=30, K=20, n_high_tau=6, high_tau=8.0)
    out = driver.run_group_from_arrays(pi_list, tau_list, z_list, nu, method="REML")
    assert out["n_eff"] < out["n_subjects"]
    assert out["ci_ratio"] >= 1.0 or out["reml_theta_se"] >= out["ttest_theta_se"] * 0.5
    # high-tau subjects carry larger alignment variance
    sig = np.asarray(out["mean_sigma2_per_subject"])
    assert sig[:6].mean() > sig[6:].mean()


def test_run_group_from_arrays_matches_library_calls():
    rng = np.random.default_rng(2)
    pi_list, tau_list, z_list, nu = _synthetic_draws(rng, S=8, M=20, V=30, K=20, n_high_tau=2)
    out = driver.run_group_from_arrays(pi_list, tau_list, z_list, nu, method="REML")
    reml = meta_analysis_map(pi_list, tau_list, z_list, nu, method="REML")
    ttest = one_sample_ttest(reml["m"])
    assert out["n_eff"] == pytest.approx(float(np.mean(reml["n_eff"])))
    assert out["reml_theta_se"] == pytest.approx(float(np.mean(reml["se_theta"])))
    assert out["ttest_theta_se"] == pytest.approx(float(np.mean(ttest["se_theta"])))


def test_run_group_cli_smoke(tmp_path: Path):
    rng = np.random.default_rng(3)
    S, M, V, K = 6, 20, 30, 20
    pi_list, tau_list, _, nu = _synthetic_draws(rng, S=S, M=M, V=V, K=K, n_high_tau=2)
    art = tmp_path / "artifacts"
    art.mkdir()
    ids = [f"{i:03d}" for i in range(S)]
    np.savez_compressed(
        art / "posterior_samples.npz",
        **{f"sub-{sid}": pi for sid, pi in zip(ids, pi_list)},
    )
    np.savez(art / "tau_phi.npz", **{f"sub-{sid}": t for sid, t in zip(ids, tau_list)})
    np.savez(art / "template.npz", B=rng.normal(size=(K, 4)), nu=nu, subject_ids=np.array(ids))

    out_json = tmp_path / "group.json"
    rc = driver.main(["--artifacts", str(art), "--out", str(out_json), "--method", "REML"])
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["n_subjects"] == S
    assert payload["n_eff"] < S
    assert "ci_ratio" in payload
    assert Path(payload["artifacts"]) == art
