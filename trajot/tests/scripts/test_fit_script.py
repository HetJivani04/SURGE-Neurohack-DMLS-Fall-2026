from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from trajot.inference.synthetic import make_synthetic_npz
from trajot.runlog.registry import read_registry

ROOT = Path(__file__).resolve().parents[2]
TINY = dict(n_subjects=6, V=24, R=8, T=48, d=32)  # d = model.d


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("fit_script", ROOT / "scripts" / "fit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path, script, monkeypatch):
    """configs/ copied to tmp with a paths.yaml; tiny synthetic sizes; runs/ redirected."""
    shutil.copytree(ROOT / "configs", tmp_path / "configs", ignore=shutil.ignore_patterns("paths.yaml"))
    (tmp_path / "configs" / "paths.yaml").write_text(f"data_root: {tmp_path / 'data'}\ncontract_version: '1.0.0'\n")
    monkeypatch.setattr(script, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(script, "SYNTHETIC", dict(TINY))
    return tmp_path


def cli(project: Path, *extra: str, name: str = "10_ours_full") -> list[str]:
    return ["--config", str(project / "configs" / "experiments" / f"{name}.yaml"), *extra]


def only_run(project: Path) -> Path:
    (run_dir,) = [p for p in (project / "runs").iterdir() if p.is_dir()]
    return run_dir


def test_a_synthetic_smoke_fit_writes_the_full_run_record(script, project) -> None:
    assert script.main(cli(project, "--synthetic", "--epochs", "2")) == 0
    run = only_run(project)

    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["experiment"] == "10_ours_full" and manifest["n_jobs"] == 1 and manifest["end_utc"]
    assert manifest["threads"]["OMP_NUM_THREADS"] == "1" and manifest["data_hash"] == "synthetic"
    assert manifest["config"]["model"]["train"]["epochs"] == 2 and manifest["config"]["model"]["K"] == TINY["V"]  # K: 512 -> V

    metrics = json.loads((run / "metrics.json").read_text())
    assert set(metrics) == {"experiment", "run_id", "n_subjects", "n_pairs", "pairs_seed", "permutations_B",
                            "methods", "beta", "notes"}  # the metrics.json schema published by W0
    assert metrics["run_id"] == run.name and metrics["n_subjects"] == 6 and metrics["methods"] == {}
    assert metrics["beta"] == 50.0 and "entropy wired" in metrics["notes"] and "synthetic" in metrics["notes"]
    assert "shannon_pi+hutchinson_slq" in metrics["notes"] and "anchor euclidean" in metrics["notes"]

    log = (run / "log.txt").read_text()
    assert "entropy WIRED" in log and log.count("epoch ") == 2 and "finished" in log
    artifacts = sorted(p.name for p in (run / "artifacts").iterdir())
    assert artifacts == ["loss_trace.json", "posterior_samples.npz", "tau_phi.npz", "template.npz"]
    assert list(read_registry(project / "runs" / "index.csv")["status"]) == ["ok"]
    trace = json.loads((run / "artifacts" / "loss_trace.json").read_text())
    assert any(e["entropy"] != 0.0 for e in trace)


def test_beta_is_never_calibrated_on_synthetic_data(script, project) -> None:
    script.main(cli(project, "--synthetic", "--epochs", "1"))
    assert not (only_run(project) / "artifacts" / "beta.json").exists()


def test_a_real_run_calibrates_beta_from_the_two_run_subjects_and_writes_beta_json(script, project, monkeypatch) -> None:
    make_synthetic_npz(project / "data", seed=0, **TINY)  # stands in for the preprocessed dataset
    calls = []

    def fake_calibrate(root, subjects=None, out_dir=None):
        calls.append((Path(root), subjects))
        result = {"sigma_hat_C_squared": 0.08, "beta": 12.5, "n_subjects": 83, "subject_ids": [], "per_subject_terms": []}
        (Path(out_dir) / "beta.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr("trajot.inference.beta.calibrate_beta", fake_calibrate)
    monkeypatch.setattr(script, "RUNS_DIR", project / "runs")
    # the data root's K must not exceed the vertex count, so shrink K through the config the run resolves
    config = project / "configs" / "model" / "default.yaml"
    config.write_text(config.read_text().replace("K: 512 ", "K: 24  "))

    assert script.main(cli(project, "--epochs", "1")) == 0
    run = only_run(project)
    assert calls == [(project / "data", None)]  # subjects=None: the 83 two-run subjects
    assert json.loads((run / "artifacts" / "beta.json").read_text())["beta"] == 12.5
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["beta"] == 12.5 and "calibrated from 83 two-run subjects" in metrics["notes"]
    assert json.loads((run / "manifest.json").read_text())["data_hash"] not in ("synthetic", "missing")


def test_the_subjects_flag_selects_subjects(script, project) -> None:
    script.main(cli(project, "--synthetic", "--epochs", "1", "--subjects", "001,003"))
    assert json.loads((only_run(project) / "metrics.json").read_text())["n_subjects"] == 2


def test_the_ablation_config_turns_the_gauge_features_off(script, project, capsys) -> None:
    script.main(cli(project, "--synthetic", "--epochs", "1", name="11_ours_ablated"))
    assert "gauge features False" in capsys.readouterr().out
    assert json.loads((only_run(project) / "manifest.json").read_text())["experiment"] == "11_ours_ablated"


def test_the_script_pins_threads_before_numpy_or_torch_is_imported() -> None:
    import subprocess

    code = (
        "import sys, importlib.util; "
        f"spec = importlib.util.spec_from_file_location('f', {str(ROOT / 'scripts' / 'fit.py')!r}); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "assert 'numpy' not in sys.modules and 'pandas' not in sys.modules and 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
