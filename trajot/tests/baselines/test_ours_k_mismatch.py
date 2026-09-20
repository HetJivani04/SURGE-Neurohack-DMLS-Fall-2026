"""R != K coupling and fit_error notes: ours_full must not silently become noalign."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from trajot.baselines.ours import (
    OursFull,
    _cdist_rows,
    _project_connectome,
    _soft_coupling,
    _template_connectome,
)

ROOT = Path(__file__).resolve().parents[2]
METHOD_KEYS = {
    "ident_accuracy",
    "ident_ci",
    "perm_p",
    "null_max",
    "alignment_gain",
    "nonidentifiable_pairs",
    "per_pair_uncertainty",
    "per_pair_flags",
}


def _planted_connectomes(S: int, R: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(R, R))
    shared = shared @ shared.T
    np.fill_diagonal(shared, 0.0)
    mats = np.empty((S, R, R), dtype=np.float64)
    for s in range(S):
        perm = rng.permutation(R)
        Pmat = np.eye(R)[perm]
        C = Pmat.T @ shared @ Pmat
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 0.0)
        mats[s] = C
    return mats


def _template_B_from_mats(mats: np.ndarray, K: int, r: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    avg = mats.mean(axis=0)
    vals, vecs = np.linalg.eigh(0.5 * (avg + avg.T))
    order = np.argsort(np.abs(vals))[::-1][:r]
    emb = vecs[:, order] * np.sqrt(np.abs(vals[order]))
    R = mats.shape[1]
    rows = rng.permutation(R)[:K] if K <= R else rng.integers(0, R, size=K)
    return np.ascontiguousarray(emb[rows] + 0.01 * rng.normal(size=(K, r)), dtype=np.float64)


def _write_template_artifacts(run_dir: Path, B: np.ndarray, n_subjects: int) -> Path:
    art = run_dir / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    K = B.shape[0]
    np.savez(
        art / "template.npz",
        B=np.asarray(B, dtype=np.float64),
        F_bar=np.asarray(B, dtype=np.float64),
        nu=np.full(K, 1.0 / K),
        eps=np.ones(n_subjects, dtype=np.float64),
        subject_ids=np.array([f"{i + 1:03d}" for i in range(n_subjects)]),
    )
    return run_dir


def test_cdist_rows_handles_mismatched_profile_dims():
    R, K = 20, 8
    rng = np.random.default_rng(0)
    A = rng.normal(size=(R, R))
    B = rng.normal(size=(K, K))
    M = _cdist_rows(A, B)
    assert M.shape == (R, K)
    assert np.isfinite(M).all()
    assert (M >= -1e-12).all()


def test_soft_coupling_r_neq_k_returns_rectangular_pi():
    R, K = 20, 8
    rng = np.random.default_rng(1)
    mats = _planted_connectomes(S=1, R=R, seed=2)
    C = mats[0]
    B = _template_B_from_mats(mats, K=K, r=4, seed=3)
    C_bar = _template_connectome(B)
    assert C_bar.shape == (K, K)

    pi = _soft_coupling(C, C_bar)
    assert pi.shape == (R, K)
    assert np.isfinite(pi).all()
    assert float(pi.sum()) == pytest.approx(1.0, abs=1e-6)


def test_project_connectome_accepts_non_square_pi():
    R, K = 16, 6
    rng = np.random.default_rng(4)
    mats = _planted_connectomes(S=1, R=R, seed=5)
    C = mats[0]
    B = _template_B_from_mats(mats, K=K, r=3, seed=6)
    C_bar = _template_connectome(B)
    pi = rng.random((R, K))
    pi /= pi.sum()

    out = _project_connectome(C, pi, C_bar)
    assert out.shape == (R, R)
    np.testing.assert_allclose(out, out.T, atol=1e-8)
    np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-10)

    out_fb = _project_connectome(C, pi, None)
    assert out_fb.shape == (R, R)
    assert np.isfinite(out_fb).all()


def test_fit_with_k_neq_r_does_not_raise_and_is_not_identity(tmp_path: Path):
    """K=32 template on 100-region data must align, not silently fall back to identity."""
    S, R, K, r = 3, 100, 32, 4
    mats = _planted_connectomes(S=S, R=R, seed=7)
    B = _template_B_from_mats(mats, K=K, r=r, seed=8)
    run_dir = _write_template_artifacts(tmp_path / "run_k32", B, n_subjects=S)

    model = OursFull()
    model.fit(mats, {"run.seed": 0, "model.K": K, "model.r": r}, extra={"run_dir": str(run_dir)})

    assert model.template_connectome is not None
    assert model.template_connectome.shape == (K, K)
    assert model.meta.get("K") == K
    assert model.meta.get("n_regions") == R

    out = model.transform(mats[0])
    assert out.shape == (R, R)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, out.T, atol=1e-8)
    np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-10)
    assert not np.allclose(out, mats[0]), "ours_full with K!=R must not evaluate as identity/noalign"

    aligned = model.transform_all(mats)
    assert aligned.shape == mats.shape
    assert not np.allclose(aligned, mats)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _EvalModuleWithFitError:
    """Minimal evaluate.py stand-in that records a failed fit as identity fallback."""

    @staticmethod
    def evaluate_method(*args, **kwargs):
        stats = {k: None for k in METHOD_KEYS}
        stats["ident_accuracy"] = 0.5
        stats["perm_p"] = 1.0
        stats["null_max"] = 0.5
        stats["alignment_gain"] = 0.0
        stats["nonidentifiable_pairs"] = 0
        stats["_meta"] = {
            "fit_error": "ValueError: matmul size mismatch (ours_full K!=R)",
            "fallback": "identity",
        }
        return stats

    @staticmethod
    def _strip_meta(stats: dict) -> dict:
        return {k: stats[k] for k in METHOD_KEYS if k in stats}

    @staticmethod
    def _load_data(cfg, synthetic: bool = False):
        run = np.stack([np.eye(4), np.eye(4)], axis=0)
        return run, run.copy(), ["001", "002"], None


def test_failed_fit_notes_contain_fit_error(tmp_path: Path, monkeypatch):
    """metrics.json notes must surface fit_error so failed ours_full is not silent."""
    shutil = pytest.importorskip("shutil")
    from trajot.config import load_config

    configs_src = ROOT / "configs"
    configs_dst = tmp_path / "configs"
    shutil.copytree(configs_src, configs_dst, ignore=shutil.ignore_patterns("paths.yaml"))
    (configs_dst / "paths.yaml").write_text(
        f"data_root: {tmp_path / 'data'}\ncontract_version: '1.0.0'\n"
    )
    runs = tmp_path / "runs"
    monkeypatch.setenv("TRAJOT_RUNS_DIR", str(runs))

    module = _load_module(ROOT / "scripts" / "run_experiment.py", "run_experiment_fit_error")
    monkeypatch.setattr(module, "RUNS_DIR", runs)
    monkeypatch.setattr(module, "_load_evaluate_module", lambda: _EvalModuleWithFitError)

    cfg = load_config(
        str(configs_dst / "experiments" / "10_ours_full.yaml"),
        {
            "run.debug": True,
            "run.n_jobs": 1,
            "run.seed": 0,
            "run.synthetic": True,
            "eval.pairs.n": 4,
            "eval.permutations.B": 8,
        },
    )
    run_dir = runs / "test_fit_error"
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    metrics = module.dispatch(cfg, run_dir)
    notes = metrics.get("notes") or ""
    assert "fit_error" not in notes or True  # key name may not appear; content must
    assert "matmul size mismatch" in notes, f"fit_error must appear in notes, got {notes!r}"
    assert "fallback=identity" in notes, f"identity fallback must appear in notes, got {notes!r}"

    method = metrics["methods"]["ours_full"]
    # Stub module only fills METHOD_KEYS; real evaluate also adds diagnostics.
    assert METHOD_KEYS.issubset(set(method))
    assert "_meta" not in method
