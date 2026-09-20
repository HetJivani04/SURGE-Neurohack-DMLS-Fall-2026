from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "synthetic_coverage", ROOT / "scripts" / "synthetic_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_planted_generative_model_shapes_and_mix(script) -> None:
    plant = script.plant_population(N=20, R=12, beta=29.189, frac_ambiguous=0.3, seed=0)
    assert plant["C_pop"].shape == (12, 12)
    assert plant["P_star"].shape == (20, 12, 12)
    assert plant["C_obs"].shape == (2, 20, 12, 12)
    amb = plant["ambiguous"]
    assert int(amb.sum()) == 6
    # Sharp maps are (near-)permutations; ambiguous are 50/50 mixtures of two perms.
    for s in range(20):
        P = plant["P_star"][s]
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-8)
        if not amb[s]:
            assert P.max() >= 0.9
        else:
            p1, p2 = plant["mix_perms"][s]
            expected = 0.5 * np.eye(12)[p1] + 0.5 * np.eye(12)[p2]
            np.testing.assert_allclose(P, expected, atol=1e-12)
        C_true = plant["C_true"][s]
        np.testing.assert_allclose(C_true, C_true.T, atol=1e-10)
        assert np.allclose(np.diag(C_true), 0.0)
    # Observations differ from C_true by ~N(0, 1/beta) noise
    resid = plant["C_obs"][0] - plant["C_true"]
    iu = np.triu_indices(12, k=1)
    emp = float(np.var(resid[:, iu[0], iu[1]]))
    assert 0.2 * plant["sigma2"] < emp < 5.0 * plant["sigma2"]


def test_contract_roundtrip_and_truth(script, tmp_path) -> None:
    plant = script.plant_population(N=6, R=8, seed=1)
    root = script.make_planted_contract(tmp_path / "data", plant)
    truth = script.read_planted_truth(root)
    assert truth["C_pop"].shape == (8, 8)
    from trajot.io.contract import read_manifest, read_subject_run, subject_run_path

    manifest = read_manifest(root)
    assert len(manifest) == 12
    d = read_subject_run(subject_run_path(root, "001", "1"))
    assert d["connectivity"].shape == (8, 8)
    np.testing.assert_allclose(d["connectivity"], plant["C_obs"][0, 0], atol=1e-12)


def test_pool_and_hungarian(script) -> None:
    rng = np.random.default_rng(0)
    P = np.eye(6)[rng.permutation(6)]
    pi = P / 6.0
    pooled = script.pool_pi_to_regions(pi, None, 6)
    np.testing.assert_allclose(pooled, pi)
    pred = script.hungarian_perm(pi)
    assert pred.shape == (6,)
    assert script.coupling_accuracy(pi, pred, 6) == 1.0


def test_plant_and_fit_smoke_shapes(script, tmp_path) -> None:
    """Fast end-to-end: N=8 R=10 epochs=2. Assert key presence + finite metrics, not SOTA wins."""
    payload = script.plant_and_fit(
        N=8,
        R=10,
        beta=29.189,
        frac_ambiguous=0.3,
        M=2,
        epochs=2,
        seed=0,
        K=10,
        data_root=tmp_path / "synth",
        train_ablated=False,
        run_fugw=True,
        verbose=False,
    )
    required = [
        "coverage_90",
        "coverage_80",
        "auroc_tau",
        "heldout_ours",
        "heldout_noalign",
        "group_neff",
        "group_fpr_reml",
        "group_fpr_ttest",
        "recovery_error_ours_full",
        "recovery_error_noalign",
        "recovery_error_procrustes_C_pop",
        "recovery_error_emd_to_C_pop",
        "coupling_recovery_ours_full",
        "coupling_recovery_random",
        "sota_bar",
        "notes",
    ]
    for key in required:
        assert key in payload, key
        if key not in ("sota_bar", "notes"):
            val = payload[key]
            if val is not None:
                assert np.isfinite(float(val)), key
    assert 0.0 <= payload["coverage_90"] <= 1.0
    assert 0.0 <= payload["auroc_tau"] <= 1.0
    assert payload["coupling_recovery_ours_full"] >= 0.0
    assert payload["group_neff"] <= 8.0 + 1e-6
    assert payload["recovery_error_noalign"] > 0.0
    assert "recovery_error_fugw" in payload
    assert "recovery_error_ours_full_shrink" in payload

    json_path, md_path = script.write_results(payload, tmp_path / "tables")
    assert json_path.is_file() and md_path.is_file()
    loaded = json.loads(json_path.read_text())
    assert loaded["config"]["N"] == 8
    assert "recovery_error_ours_full" in loaded
    assert "SOTA bar" in md_path.read_text() or "SOTA" in md_path.read_text()


def test_cli_smoke(script, tmp_path) -> None:
    rc = script.main([
        "--smoke",
        "--data-root", str(tmp_path / "cli_synth"),
        "--out-dir", str(tmp_path / "tables"),
        "--no-ablation",
        "--no-fugw",
        "--epochs", "1",
        "--M", "2",
        "--N", "6",
        "--R", "8",
    ])
    # --smoke sets defaults but explicit N/R/epochs override; epochs=1 is allowed for CLI speed.
    # Plan requires full harness epochs>=10; CLI smoke is exempt.
    assert rc == 0
    payload = json.loads((tmp_path / "tables" / "synthetic_gap_results.json").read_text())
    assert payload["config"]["N"] == 6
    assert payload["config"]["epochs"] == 1
