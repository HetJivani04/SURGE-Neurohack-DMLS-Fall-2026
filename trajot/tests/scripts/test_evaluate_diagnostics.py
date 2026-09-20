"""Evaluator diagnostics: identity fallback must be visible in metrics.json.

general-99: evaluate.py silently falls back to identity when method.fit/transform
raises. Failed runs were indistinguishable from successful ones in metrics.json
because ``_strip_meta`` dropped ``_meta.fit_error``.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

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

DIAGNOSTIC_KEYS = {
    "fit_error",
    "transforms_applied",
    "transform_diagnostics",
    "status",
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sym(rng, R: int) -> np.ndarray:
    A = rng.normal(size=(R, R))
    C = 0.5 * (A + A.T)
    np.fill_diagonal(C, 0.0)
    return C


def _cfg(tmp_path: Path):
    from trajot.config import Config

    return Config(
        {
            "experiment": "00_noalign",
            "data": {"root": str(tmp_path / "data"), "contract_version": "1"},
            "run": {"seed": 3, "n_jobs": 1, "debug": True, "synthetic": True},
            "eval": {"pairs": {"n": 6, "seed": 7}, "permutations": {"B": 20}},
        }
    )


def _toy_data(S: int = 6, R: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    run1 = np.stack([_sym(rng, R) for _ in range(S)])
    run2 = run1 + 0.05 * np.stack([_sym(rng, R) for _ in range(S)])
    subjects = [f"{i:03d}" for i in range(S)]
    return run1, run2, subjects


class _FailingMethod:
    """fit raises — evaluator must surface the error, not silently identity-fallback."""

    def fit(self, run, cfg, extra=None):
        raise RuntimeError("intentional fit failure for diagnostics test")

    def transform(self, c):
        return c

    def transform_all(self, run, **kwargs):
        return run


class _IdentityMethod:
    """Succeeds but returns input unchanged (like noalign)."""

    def fit(self, run, cfg, extra=None):
        return self

    def transform(self, c):
        return c

    def transform_all(self, run, **kwargs):
        return np.asarray(run, dtype=np.float64)


class _ShiftMethod:
    """Succeeds and actually changes the data."""

    def fit(self, run, cfg, extra=None):
        return self

    def transform(self, c):
        return np.asarray(c, dtype=np.float64) + 0.1

    def transform_all(self, run, **kwargs):
        return np.asarray(run, dtype=np.float64) + 0.1


@pytest.fixture
def evaluate_mod(tmp_path):
    return _load_module(ROOT / "scripts" / "evaluate.py", "evaluate_diagnostics")


def test_failed_fit_surfaces_fit_error_and_identity_fallback(tmp_path, evaluate_mod):
    """When fit raises, metrics must carry fit_error + status=failed."""
    cfg = _cfg(tmp_path)
    run1, run2, subjects = _toy_data()

    stats = evaluate_mod.evaluate_method(
        "noalign", _FailingMethod(), run1, run2, cfg, subjects
    )

    assert "fit_error" in stats, "fit_error must be on the stats dict, not only _meta"
    assert "intentional fit failure" in str(stats["fit_error"])
    assert stats["status"] == "failed"
    assert stats["transforms_applied"] is False
    # Failed methods are not scored as successful noalign runs.
    assert stats["ident_accuracy"] is None

    stripped = evaluate_mod._strip_meta(stats)
    assert METHOD_KEYS.issubset(set(stripped))
    assert "fit_error" in stripped, "_strip_meta must not drop fit_error"
    assert stripped["status"] == "failed"
    assert stripped["transforms_applied"] is False
    assert stripped["ident_accuracy"] is None
    assert "_meta" not in stripped


def test_successful_fit_sets_status_ok_and_transform_diagnostics(tmp_path, evaluate_mod):
    """A real transform must report transforms_applied=True plus effectiveness stats."""
    cfg = _cfg(tmp_path)
    run1, run2, subjects = _toy_data()

    stats = evaluate_mod.evaluate_method(
        "noalign", _ShiftMethod(), run1, run2, cfg, subjects
    )

    assert stats["status"] == "ok"
    assert stats["transforms_applied"] is True
    assert stats.get("fit_error") in (None, "")

    diag = stats.get("transform_diagnostics")
    assert isinstance(diag, dict), "transform_diagnostics must be present on success"
    assert "max_abs_diff" in diag
    assert "feat_corr" in diag
    assert diag["max_abs_diff"] == pytest.approx(0.1, abs=1e-9)
    assert diag["feat_corr"] is not None
    assert diag["feat_corr"] > 0.99

    stripped = evaluate_mod._strip_meta(stats)
    assert stripped["status"] == "ok"
    assert stripped["transforms_applied"] is True
    assert isinstance(stripped.get("transform_diagnostics"), dict)


def test_identity_method_success_has_zero_max_abs_diff(tmp_path, evaluate_mod):
    """Successful identity-like transform still status=ok, but diagnostics show no change."""
    cfg = _cfg(tmp_path)
    run1, run2, subjects = _toy_data()

    stats = evaluate_mod.evaluate_method(
        "noalign", _IdentityMethod(), run1, run2, cfg, subjects
    )

    assert stats["status"] == "ok"
    # Successful identity method: status=ok, but aligned == raw so transforms_applied is False.
    assert stats["transforms_applied"] is False
    diag = stats["transform_diagnostics"]
    assert diag["max_abs_diff"] == pytest.approx(0.0, abs=1e-12)
    assert diag["feat_corr"] == pytest.approx(1.0, abs=1e-6)


def test_metrics_payload_validation_allows_diagnostics(tmp_path, evaluate_mod):
    """validate_metrics_payload must accept diagnostic extras without dropping METHOD_KEYS."""
    from trajot.eval.metrics import validate_metrics_payload

    cfg = _cfg(tmp_path)
    run1, run2, subjects = _toy_data()
    stats = evaluate_mod.evaluate_method(
        "noalign", _FailingMethod(), run1, run2, cfg, subjects
    )
    stripped = evaluate_mod._strip_meta(stats)

    payload = {
        "experiment": "00_noalign",
        "run_id": "r_diag",
        "n_subjects": len(subjects),
        "n_pairs": 6,
        "pairs_seed": 7,
        "permutations_B": 20,
        "methods": {"noalign": stripped},
        "beta": None,
        "notes": "",
    }
    validate_metrics_payload(payload)

    bad = dict(stripped)
    bad["unexpected_key"] = 1
    payload_bad = dict(payload)
    payload_bad["methods"] = {"noalign": bad}
    with pytest.raises(ValueError):
        validate_metrics_payload(payload_bad)


def test_run_experiment_metrics_json_contains_fit_error(tmp_path, monkeypatch, evaluate_mod):
    """Identity fallback must be visible in the written metrics.json methods entry."""
    shutil.copytree(
        ROOT / "configs",
        tmp_path / "configs",
        ignore=shutil.ignore_patterns("paths.yaml"),
    )
    (tmp_path / "configs" / "paths.yaml").write_text(
        f"data_root: {tmp_path / 'data'}\ncontract_version: '1.0.0'\n"
    )
    runs = tmp_path / "runs"
    monkeypatch.setenv("TRAJOT_RUNS_DIR", str(runs))

    run_exp = _load_module(ROOT / "scripts" / "run_experiment.py", "run_exp_diag")
    monkeypatch.setattr(run_exp, "RUNS_DIR", runs)

    from trajot.config import load_config

    cfg = load_config(
        str(tmp_path / "configs" / "experiments" / "00_noalign.yaml"),
        {
            "run.debug": True,
            "run.n_jobs": 1,
            "run.seed": 0,
            "run.synthetic": True,
            "eval.pairs.n": 4,
            "eval.permutations.B": 8,
        },
    )
    run_dir = runs / "test_diag_fit_error"
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    # Patch evaluate_module to inject a failing method for "noalign".
    original_load = run_exp._load_evaluate_module

    def _load_with_failing():
        ev = original_load()
        if ev is None:
            return None
        real_evaluate = ev.evaluate_method

        def evaluate_method(name, method, run1, run2, cfg, subjects, **kwargs):
            return real_evaluate(
                name,
                _FailingMethod(),
                run1,
                run2,
                cfg,
                subjects,
                **kwargs,
            )

        ev.evaluate_method = evaluate_method
        return ev

    monkeypatch.setattr(run_exp, "_load_evaluate_module", _load_with_failing)

    metrics = run_exp.dispatch(cfg, run_dir)
    method = metrics["methods"]["noalign"]

    assert "fit_error" in method, "metrics.json methods must expose fit_error"
    assert "intentional fit failure" in str(method["fit_error"])
    assert method["status"] == "failed"
    assert method["transforms_applied"] is False
    assert METHOD_KEYS.issubset(set(method))
    notes = metrics.get("notes") or ""
    assert "intentional fit failure" in notes


def test_write_metrics_roundtrip_preserves_diagnostics(tmp_path, evaluate_mod):
    """write_metrics must not strip diagnostic fields from methods."""
    cfg = _cfg(tmp_path)
    run1, run2, subjects = _toy_data()
    stats = evaluate_mod.evaluate_method(
        "noalign", _ShiftMethod(), run1, run2, cfg, subjects
    )
    stripped = evaluate_mod._strip_meta(stats)

    payload = {
        "experiment": "00_noalign",
        "run_id": "r_write",
        "n_subjects": len(subjects),
        "n_pairs": 6,
        "pairs_seed": 7,
        "permutations_B": 20,
        "methods": {"noalign": stripped},
        "beta": None,
        "notes": "",
    }
    out = tmp_path / "metrics.json"
    evaluate_mod.write_metrics_payload(out, payload)
    loaded = json.loads(out.read_text())
    method = loaded["methods"]["noalign"]
    assert method["status"] == "ok"
    assert method["transforms_applied"] is True
    assert isinstance(method["transform_diagnostics"], dict)
    assert "max_abs_diff" in method["transform_diagnostics"]
