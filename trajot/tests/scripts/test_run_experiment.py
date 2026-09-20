from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PLAN_METHOD_KEYS = {
    "ident_accuracy",
    "ident_ci",
    "perm_p",
    "null_max",
    "alignment_gain",
    "nonidentifiable_pairs",
    "per_pair_uncertainty",
    "per_pair_flags",
}


def _expected_method_keys() -> set[str]:
    try:
        from trajot.eval.metrics import METHOD_KEYS

        if "null_max" in set(METHOD_KEYS):
            return set(METHOD_KEYS)
    except Exception:
        pass
    return set(PLAN_METHOD_KEYS)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "configs", tmp_path / "configs", ignore=shutil.ignore_patterns("paths.yaml"))
    (tmp_path / "configs" / "paths.yaml").write_text(
        f"data_root: {tmp_path / 'data'}\ncontract_version: '1.0.0'\n"
    )
    runs = tmp_path / "runs"
    monkeypatch.setenv("TRAJOT_RUNS_DIR", str(runs))
    module = _load_module(ROOT / "scripts" / "run_experiment.py", "run_experiment_copy")
    monkeypatch.setattr(module, "RUNS_DIR", runs)
    return tmp_path, module, runs


def _debug_overrides() -> list[str]:
    return [
        "--override",
        "run.debug=true",
        "--override",
        "run.n_jobs=1",
        "--override",
        "run.seed=5",
        "--override",
        "eval.pairs.n=6",
        "--override",
        "eval.permutations.B=16",
    ]


def test_dispatch_00_noalign_synthetic_methods_not_empty(project):
    tmp_path, module, runs = project
    from trajot.config import load_config

    cfg = load_config(
        str(tmp_path / "configs" / "experiments" / "00_noalign.yaml"),
        {
            "run.debug": True,
            "run.n_jobs": 1,
            "run.seed": 5,
            "run.synthetic": True,
            "eval.pairs.n": 6,
            "eval.permutations.B": 16,
        },
    )
    run_dir = tmp_path / "runs" / "test_00_noalign"
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    metrics = module.dispatch(cfg, run_dir)
    assert metrics["experiment"] == "00_noalign"
    assert metrics["run_id"] == run_dir.name
    assert metrics["methods"], "00_noalign must not return empty methods on synthetic data"
    assert "noalign" in metrics["methods"]

    stats = metrics["methods"]["noalign"]
    assert set(stats) == _expected_method_keys()
    assert 0.0 <= stats["ident_accuracy"] <= 1.0
    assert isinstance(stats["nonidentifiable_pairs"], int)
    # Schema validates through evaluate.validate_payload
    ev = module._load_evaluate_module()
    ev.validate_payload(metrics)


def test_main_00_noalign_synthetic_writes_metrics(project):
    tmp_path, module, runs = project
    rc = module.main(
        [
            "--config",
            str(tmp_path / "configs" / "experiments" / "00_noalign.yaml"),
            "--synthetic",
            "--dry-run",
            *_debug_overrides(),
        ]
    )
    assert rc == 0

    run_dirs = [p for p in runs.iterdir() if p.is_dir()]
    assert run_dirs, "run directory was not created"
    metrics_path = run_dirs[0] / "metrics.json"
    assert metrics_path.is_file()
    payload = json.loads(metrics_path.read_text())
    assert payload["methods"], "synthetic 00_noalign must produce non-empty methods"
    assert "noalign" in payload["methods"]
    assert "null_max" in payload["methods"]["noalign"]
    # W0 acceptance path: dry-run still records the run in runs/index.csv
    assert (runs / "index.csv").is_file()


def test_experiment_runners_are_not_all_stubs(project):
    _, module, _ = project
    runners = module.EXPERIMENT_RUNNERS
    assert set(runners) == set(module.EXPERIMENT_NAMES)
    for name, fn in runners.items():
        assert fn is not module._stub_runner, f"{name} still maps to _stub_runner"
        assert callable(fn)
