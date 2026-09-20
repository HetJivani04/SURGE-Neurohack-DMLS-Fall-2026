import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from trajot.config import ConfigError
from trajot.runlog.registry import REGISTRY_COLUMNS, read_registry

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ["00_noalign", "01_brainsync", "02_fugw", "03_conn_srm", "10_ours_full", "11_ours_ablated"]
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

TOP_KEYS = {"experiment", "run_id", "n_subjects", "n_pairs", "pairs_seed", "permutations_B",
            "methods", "beta", "notes"}
METHOD_KEYS = {"ident_accuracy", "ident_ci", "perm_p", "alignment_gain", "nonidentifiable_pairs",
               "per_pair_uncertainty", "per_pair_flags"}


def assert_schema_valid(metrics):
    """The metrics.json schema from issue #2: every key present, null allowed for method fields."""
    assert set(metrics) == TOP_KEYS
    assert isinstance(metrics["experiment"], str) and isinstance(metrics["run_id"], str)
    assert all(isinstance(metrics[k], int) for k in ("n_subjects", "n_pairs", "pairs_seed", "permutations_B"))
    assert metrics["beta"] is None or isinstance(metrics["beta"], float)
    assert isinstance(metrics["notes"], str) and isinstance(metrics["methods"], dict)
    for entry in metrics["methods"].values():
        assert set(entry) == METHOD_KEYS
        for key in ("ident_accuracy", "perm_p", "alignment_gain", "per_pair_uncertainty"):
            assert entry[key] is None or isinstance(entry[key], float)
        assert entry["nonidentifiable_pairs"] is None or isinstance(entry["nonidentifiable_pairs"], int)
        assert entry["ident_ci"] is None or (len(entry["ident_ci"]) == 2 and all(isinstance(x, float) for x in entry["ident_ci"]))
        assert entry["per_pair_flags"] is None or all(isinstance(f, bool) for f in entry["per_pair_flags"])


@pytest.fixture
def project(tmp_path):
    """configs/ + scripts/run_experiment.py copied to tmp_path, so runs/ lands in tmp_path too."""
    shutil.copytree(ROOT / "configs", tmp_path / "configs", ignore=shutil.ignore_patterns("paths.yaml"))
    (tmp_path / "configs" / "paths.yaml").write_text(
        f"data_root: {tmp_path / 'data'}\ncontract_version: '1.0.0'\n")
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "run_experiment.py", tmp_path / "scripts")
    return tmp_path


@pytest.fixture
def entry(project):
    spec = importlib.util.spec_from_file_location("run_experiment_copy", project / "scripts" / "run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def exp_path(project):
    return lambda name="00_noalign": project / "configs" / "experiments" / f"{name}.yaml"


@pytest.fixture(autouse=True)
def restore_threads():
    import torch

    saved = {v: os.environ.get(v) for v in THREAD_VARS}
    saved_torch = torch.get_num_threads()
    yield
    for var, value in saved.items():
        os.environ.pop(var, None) if value is None else os.environ.__setitem__(var, value)
    torch.set_num_threads(saved_torch)


def run_dirs(project):
    return sorted(p for p in (project / "runs").iterdir() if p.is_dir())


def test_dry_run_creates_a_complete_run_directory(entry, exp_path, project):
    assert entry.main(["--config", str(exp_path()), "--dry-run"]) == 0

    (run_dir,) = run_dirs(project)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["experiment"] == "00_noalign" and run_dir.name.split("__")[1] == manifest["config_hash"][:8]
    assert manifest["n_jobs"] == 1 and manifest["seed"] == 0
    assert manifest["threads"]["OMP_NUM_THREADS"] == "1" and manifest["contract_digest"]
    assert manifest["data_hash"] == "missing" and manifest["start_utc"] and manifest["end_utc"]

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert_schema_valid(metrics)
    assert metrics["run_id"] == run_dir.name
    assert run_dir.name in (run_dir / "log.txt").read_text()
    assert (run_dir / "artifacts").is_dir()

    df = read_registry(project / "runs" / "index.csv")
    assert list(df.columns) == REGISTRY_COLUMNS and list(df["run_id"]) == [run_dir.name]


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_experiment_runs_through_the_stub_dispatcher(entry, exp_path, project, name):
    assert entry.main(["--config", str(exp_path(name))]) == 0
    (run_dir,) = run_dirs(project)
    assert_schema_valid(json.loads((run_dir / "metrics.json").read_text()))


def test_rerunning_one_run_id_does_not_duplicate_its_row(entry, exp_path, project, monkeypatch):
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 19, 12, 0, 0, tzinfo=tz)

    monkeypatch.setattr(entry, "datetime", Frozen)
    assert entry.main(["--config", str(exp_path()), "--dry-run"]) == 0
    assert entry.main(["--config", str(exp_path()), "--dry-run"]) == 0
    assert entry.main(["--config", str(exp_path("01_brainsync")), "--dry-run"]) == 0
    df = read_registry(project / "runs" / "index.csv")
    assert len(df) == 2 and df["run_id"].is_unique and len(run_dirs(project)) == 2


def test_overrides_and_flags_change_the_run_identity(entry, exp_path, project):
    base = ["--config", str(exp_path()), "--dry-run"]
    for extra in ([], ["--seed", "5"], ["--override", "model.K=64"], ["--n-jobs", "2"]):
        assert entry.main(base + extra) == 0
    df = read_registry(project / "runs" / "index.csv")
    assert df["config_hash"].nunique() == 4
    assert sorted(df["seed"]) == [0, 0, 0, 5] and sorted(df["n_jobs"]) == [1, 1, 1, 2]


def test_debug_flag_allows_small_pair_counts(entry, exp_path, project):
    args = ["--config", str(exp_path()), "--dry-run", "--override", "eval.pairs.n=20",
            "--override", "run.debug=false"]
    with pytest.raises(ConfigError, match="eval.pairs.n"):
        entry.main(args)
    debug_args = ["--config", str(exp_path()), "--dry-run", "--override", "eval.pairs.n=20", "--debug"]
    assert entry.main(debug_args) == 0
    (run_dir,) = run_dirs(project)
    assert json.loads((run_dir / "metrics.json").read_text())["n_pairs"] == 20


def test_missing_paths_yaml_raises(entry, exp_path, project):
    (project / "configs" / "paths.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="copy configs/paths.example.yaml to configs/paths.yaml"):
        entry.main(["--config", str(exp_path())])


def test_runner_payload_is_written_as_metrics_json(entry, exp_path, project, monkeypatch):
    payload = {"experiment": "00_noalign", "run_id": "x", "n_subjects": 83, "n_pairs": 500, "pairs_seed": 2026,
               "permutations_B": 10000, "beta": None, "notes": "",
               "methods": {"noalign": {"ident_accuracy": 0.4, "ident_ci": [0.3, 0.5], "perm_p": 0.02,
                                       "alignment_gain": None, "nonidentifiable_pairs": 12,
                                       "per_pair_uncertainty": None, "per_pair_flags": None}}}
    assert_schema_valid(payload)
    monkeypatch.setitem(entry.EXPERIMENT_RUNNERS, "00_noalign", lambda cfg, run_dir: payload)
    assert entry.main(["--config", str(exp_path())]) == 0
    (run_dir,) = run_dirs(project)
    assert json.loads((run_dir / "metrics.json").read_text()) == payload


def test_the_logger_is_closed_even_if_the_runner_fails(entry, exp_path, project, monkeypatch):
    def boom(cfg, run_dir):
        print("about to fail")
        raise RuntimeError("kaboom")

    monkeypatch.setitem(entry.EXPERIMENT_RUNNERS, "00_noalign", boom)
    stdout, stderr = sys.stdout, sys.stderr
    with pytest.raises(RuntimeError, match="kaboom"):
        entry.main(["--config", str(exp_path())])
    assert sys.stdout is stdout and sys.stderr is stderr
    (run_dir,) = run_dirs(project)
    assert "about to fail" in (run_dir / "log.txt").read_text()


def test_data_hash_is_recorded_when_data_exists(entry, exp_path, project):
    derived = project / "data" / "derivatives" / "trajot"
    derived.mkdir(parents=True)
    (derived / "sub-001_run-1.npz").write_bytes(b"x" * 100)
    assert entry.main(["--config", str(exp_path()), "--dry-run"]) == 0
    (run_dir,) = run_dirs(project)
    assert len(json.loads((run_dir / "manifest.json").read_text())["data_hash"]) == 64


def test_cli_dry_run_as_a_subprocess(exp_path, project):
    """The acceptance command, in a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, str(project / "scripts" / "run_experiment.py"), "--config", str(exp_path()), "--dry-run"],
        capture_output=True, text=True, cwd=project)
    assert result.returncode == 0, result.stderr
    (run_dir,) = run_dirs(project)
    assert all((run_dir / name).is_file() for name in ("manifest.json", "metrics.json", "log.txt"))
    assert (run_dir / "artifacts").is_dir() and run_dir.name in result.stdout
    assert json.loads((run_dir / "manifest.json").read_text())["threads"]["OPENBLAS_NUM_THREADS"] == "1"


def test_local_config_data_and_runs_are_git_ignored():
    if subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True).returncode:
        pytest.skip("not inside a git work tree")
    for path in ("configs/paths.yaml", "data/ds000243", "runs/index.csv", "runs/x__1__2/manifest.json",
                 ".venv/bin/python", "src/trajot.egg-info/PKG-INFO", "src/trajot/__pycache__/x.pyc"):
        assert subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0, path
