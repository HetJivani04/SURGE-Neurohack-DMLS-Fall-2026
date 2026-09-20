from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trajot.inference.beta import calibrate_beta, sigma_c_squared
from trajot.io import contract


def symmetric_sign_matrix(R: int, seed: int) -> np.ndarray:
    upper = np.triu(np.random.default_rng(seed).choice([-1.0, 1.0], size=(R, R)), 1)
    return upper + upper.T


def test_sigma_c_squared_is_known_by_construction() -> None:
    """C1 = C0 + d S, C2 = C0 - d S with S a +-1 symmetric zero-diagonal matrix: sigma^2 = 2 d^2 (1 - 1/R)."""
    R, delta = 40, 0.13
    C0 = np.random.default_rng(0).normal(size=(5, R, R))
    S = np.stack([symmetric_sign_matrix(R, k) for k in range(5)])
    expected = 2 * delta**2 * (1 - 1 / R)
    assert sigma_c_squared(C0 + delta * S, C0 - delta * S) == pytest.approx(expected, rel=1e-12)
    assert sigma_c_squared(C0, C0) == 0.0


def test_sigma_c_squared_estimates_the_noise_variance_of_independent_scans() -> None:
    R, n, sigma = 60, 40, 0.09
    rng = np.random.default_rng(1)
    C0 = rng.normal(size=(n, R, R))

    def noisy():
        E = np.triu(rng.normal(0, sigma, size=(n, R, R)), 1)
        return C0 + E + E.transpose(0, 2, 1)

    assert sigma_c_squared(noisy(), noisy()) == pytest.approx(sigma**2 * (1 - 1 / R), rel=0.05)


def test_sigma_c_squared_validates_shapes() -> None:
    with pytest.raises(ValueError):
        sigma_c_squared(np.zeros((3, 4, 4)), np.zeros((3, 4, 5)))
    with pytest.raises(ValueError):
        sigma_c_squared(np.zeros((4, 4)), np.zeros((4, 4)))


def write_two_run_dataset(root: Path, n: int = 6, R: int = 12, delta: float = 0.1, extra_single: bool = True) -> None:
    rng = np.random.default_rng(2)
    rows = []
    for s in range(n):
        C0 = np.triu(rng.normal(size=(R, R)) * 0.3, 1)
        C0 = C0 + C0.T
        S = symmetric_sign_matrix(R, s)
        for run, sign in (("1", +1), ("2", -1)):
            contract.write_subject_run(
                contract.subject_run_path(root, f"{s:03d}", run), connectivity=C0 + sign * delta * S,
                timeseries=np.zeros((5, 4), np.float32), embedding=np.zeros((5, 3), np.float32),
                features=np.zeros((5, 2), np.float32), coords=np.zeros((5, 3), np.float32), tr=2.5, n_volumes=4,
                subject_id=f"{s:03d}", run_id=run)
            rows.append({"subject_id": f"{s:03d}", "run_id": run, "path": str(contract.subject_run_path(root, f"{s:03d}", run).relative_to(root)), "n_volumes": 4,
                         "tr": 2.5, "n_regions": R, "n_vertices": 5, "qc_pass": True, "contract_version": "1.0.0"})
    if extra_single:  # a subject with one run only must never enter the calibration
        contract.write_subject_run(
            contract.subject_run_path(root, "900", "1"), connectivity=np.zeros((R, R)), timeseries=np.zeros((5, 4), np.float32),
            embedding=np.zeros((5, 3), np.float32), features=np.zeros((5, 2), np.float32), coords=np.zeros((5, 3), np.float32),
            tr=2.5, n_volumes=4, subject_id="900", run_id="1")
        rows.append({"subject_id": "900", "run_id": "1", "path": str(contract.subject_run_path(root, "900", "1").relative_to(root)), "n_volumes": 4, "tr": 2.5,
                     "n_regions": R, "n_vertices": 5, "qc_pass": True, "contract_version": "1.0.0"})
    contract.write_manifest(rows, root)


def test_calibrate_beta_reproduces_the_analytic_value_and_writes_beta_json(tmp_path: Path) -> None:
    R, delta = 12, 0.1
    write_two_run_dataset(tmp_path, R=R, delta=delta)
    ids = [f"{s:03d}" for s in range(6)]
    result = calibrate_beta(tmp_path, subjects=ids, out_dir=tmp_path / "artifacts")

    sigma2 = 2 * delta**2 * (1 - 1 / R)
    assert result["sigma_hat_C_squared"] == pytest.approx(sigma2, rel=1e-12)
    assert result["beta"] == pytest.approx(1 / sigma2, rel=1e-12)
    assert result["n_regions"] == R  # D2: normalizer side is the connectome node count, not n_vertices
    assert result["n_subjects"] == 6 and result["subject_ids"] == ids
    assert np.allclose(result["per_subject_terms"], sigma2) and len(result["per_subject_terms"]) == 6
    on_disk = json.loads((tmp_path / "artifacts" / "beta.json").read_text())
    assert on_disk == result
    assert set(on_disk) == {"sigma_hat_C_squared", "beta", "n_regions", "n_subjects", "subject_ids", "per_subject_terms"}


def test_beta_docstring_says_v_denotes_connectome_nodes_not_surface_vertices() -> None:
    from trajot.inference import beta as beta_mod

    text = beta_mod.__doc__ + beta_mod.sigma_c_squared.__doc__ + beta_mod.calibrate_beta.__doc__
    assert "connectome" in text.lower()
    assert "not" in text.lower() and "surface" in text.lower()
    assert "n_regions" in beta_mod.calibrate_beta.__doc__


def test_calibrate_beta_asserts_the_real_dataset_has_83_two_run_subjects(tmp_path: Path) -> None:
    write_two_run_dataset(tmp_path)
    with pytest.raises(contract.ContractError, match="83"):
        calibrate_beta(tmp_path, out_dir=tmp_path / "artifacts")  # subjects=None: exactly 83 or it raises
    assert not (tmp_path / "artifacts").exists()


def test_calibrate_beta_refuses_a_single_run_subject(tmp_path: Path) -> None:
    write_two_run_dataset(tmp_path)
    with pytest.raises(ValueError, match="900"):
        calibrate_beta(tmp_path, subjects=["000", "900"], out_dir=tmp_path / "artifacts")
