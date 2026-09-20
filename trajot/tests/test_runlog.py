import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from trajot.config import Config
from trajot.runlog import parallel
from trajot.runlog.logging import RunLogger
from trajot.runlog.manifest import (
    contract_digest, git_commit, hash_data_root, make_run_id, package_versions, write_manifest,
)
from trajot.runlog.registry import REGISTRY_COLUMNS, append_run, read_registry, update_run_metrics

ROOT = Path(__file__).resolve().parents[1]
THREAD_VARS = parallel.THREAD_VARS


@pytest.fixture
def restore_threads():
    import torch

    saved = {v: os.environ.get(v) for v in THREAD_VARS}
    saved_torch = torch.get_num_threads()
    yield
    for var, value in saved.items():
        os.environ.pop(var, None) if value is None else os.environ.__setitem__(var, value)
    torch.set_num_threads(saved_torch)


def square(x):
    return x * x


def make_cfg():
    return Config({
        "experiment": "00_noalign",
        "data": {"root": "/somewhere", "contract_version": "1"},
        "run": {"seed": 0, "n_jobs": 1},
        "eval": {"pairs": {"n": 500, "seed": 1}, "permutations": {"B": 10000}},
    })


# ---- parallel.py -----------------------------------------------------------
@pytest.mark.parametrize("mode,n_jobs,expected", [("serial", 4, "1"), ("outer", 4, "1"), ("inner", 4, "4")])
def test_setup_threads_modes(restore_threads, mode, n_jobs, expected):
    import torch

    snapshot = parallel.setup_threads(mode, n_jobs)
    assert all(os.environ[v] == expected for v in THREAD_VARS)
    assert torch.get_num_threads() == int(expected)
    assert snapshot == {**{v: expected for v in THREAD_VARS}, "torch_threads": expected}
    assert parallel.thread_env_snapshot() == snapshot


def test_setup_threads_rejects_unknown_mode(restore_threads):
    with pytest.raises(ValueError):
        parallel.setup_threads("bogus", 1)


def test_pick_device():
    import torch

    assert parallel.pick_device(prefer_mps=False) == torch.device("cpu")
    assert parallel.pick_device().type == ("mps" if torch.backends.mps.is_available() else "cpu")


def test_parallel_map_preserves_order(restore_threads):
    items = list(range(20))
    assert parallel.parallel_map(square, items, n_jobs=1) == [i * i for i in items]
    assert parallel.parallel_map(square, items, n_jobs=2) == [i * i for i in items]


def test_parallel_map_runs_in_process_at_one_job_and_pins_threads_first(restore_threads):
    os.environ["OMP_NUM_THREADS"] = "7"
    assert set(parallel.parallel_map(lambda _: os.getpid(), [0, 1, 2], n_jobs=1)) == {os.getpid()}
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_importing_parallel_does_not_import_numpy():
    code = ("import sys; import trajot.runlog.parallel; import trajot.config; "
            "assert 'numpy' not in sys.modules and 'pandas' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_thread_env_and_torch_device_have_a_single_owner():
    """Repository-wide grep: only runlog/parallel.py assigns the thread variables or calls torch.device()."""
    owner = ROOT / "src" / "trajot" / "runlog" / "parallel.py"
    pattern = re.compile(r"OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|VECLIB_MAXIMUM_THREADS|torch\.device\(")
    offenders = [str(p.relative_to(ROOT)) for base in ("src", "scripts") for p in (ROOT / base).rglob("*.py")
                 if p != owner and pattern.search(p.read_text())]
    assert offenders == []
    assert pattern.search(owner.read_text())


# ---- manifest.py -----------------------------------------------------------
def test_make_run_id():
    when = datetime(2026, 9, 19, 21, 5, 7, tzinfo=timezone.utc)
    assert make_run_id("10_ours_full", "abcdef0123456789", when) == "10_ours_full__abcdef01__20260919T210507Z"
    assert re.fullmatch(r"x__abcdef01__\d{8}T\d{6}Z", make_run_id("x", "abcdef0123456789"))


def test_git_commit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert git_commit() == "unknown"
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["git", "init", "-q"], ["git", "commit", "-q", "--allow-empty", "-m", "x"]):
        subprocess.run(cmd, check=True, env=env)
    assert re.fullmatch(r"[0-9a-f]{4,}", git_commit())


def test_package_versions():
    versions = package_versions()
    assert set(versions) == {"python", "numpy", "scipy", "torch", "joblib", "pandas", "pot"}
    assert versions["python"].startswith(f"{sys.version_info.major}.{sys.version_info.minor}")


def test_hash_data_root(tmp_path):
    assert hash_data_root(tmp_path / "nope") == "missing"
    (tmp_path / "b.npz").write_bytes(b"bbb")
    (tmp_path / "a.npz").write_bytes(b"aaa")
    (tmp_path / "ignored.txt").write_bytes(b"zzz")
    first = hash_data_root(tmp_path)
    assert re.fullmatch(r"[0-9a-f]{64}", first) and hash_data_root(tmp_path) == first
    (tmp_path / "ignored.txt").write_bytes(b"changed")
    assert hash_data_root(tmp_path) == first
    (tmp_path / "a.npz").write_bytes(b"aaX")
    assert hash_data_root(tmp_path) != first


def test_hash_data_root_streams_and_ignores_location(tmp_path):
    payload = os.urandom(3 * (1 << 20) + 17)
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "sub-001_run-1.npz").write_bytes(payload)
    assert hash_data_root(tmp_path / "one") == hash_data_root(tmp_path / "two")


def test_hash_data_root_byte_budget(tmp_path):
    (tmp_path / "a.npz").write_bytes(b"a" * 10)
    (tmp_path / "b.npz").write_bytes(b"b" * 10)
    unlimited, budgeted = hash_data_root(tmp_path), hash_data_root(tmp_path, max_bytes=10)
    (tmp_path / "b.npz").write_bytes(b"c" * 10)
    assert hash_data_root(tmp_path) != unlimited              # contents normally matter
    assert hash_data_root(tmp_path, max_bytes=10) == budgeted  # b.npz no longer fits: path and size only
    (tmp_path / "a.npz").write_bytes(b"z" * 10)
    assert hash_data_root(tmp_path, max_bytes=10) != budgeted  # a.npz is still read


def test_write_manifest(tmp_path, restore_threads):
    cfg = make_cfg()
    start = datetime(2026, 9, 19, 21, 0, 0, tzinfo=timezone.utc)
    threads = parallel.setup_threads("serial", 1)
    kwargs = dict(seed=3, operator="het", n_jobs=1, threads=threads, data_hash="missing", start_utc=start)

    path = write_manifest(tmp_path, cfg, **kwargs, end_utc=None)
    assert path == tmp_path / "manifest.json"
    m = json.loads(path.read_text())
    assert m["experiment"] == "00_noalign" and m["config"] == cfg.raw and m["config_hash"] == cfg.hash
    assert m["seed"] == 3 and m["operator"] == "het" and m["n_jobs"] == 1
    assert m["threads"] == threads and m["data_hash"] == "missing"
    assert m["contract_digest"] == contract_digest("1")
    assert set(m["package_versions"]) >= {"python", "numpy", "torch"}
    assert m["git_commit"] and m["start_utc"] == "2026-09-19T21:00:00Z" and m["end_utc"] is None

    write_manifest(tmp_path, cfg, **kwargs, end_utc=datetime(2026, 9, 19, 21, 5, tzinfo=timezone.utc))
    assert json.loads(path.read_text())["end_utc"] == "2026-09-19T21:05:00Z"


def test_contract_digest_changes_with_version():
    assert contract_digest("1") == contract_digest("1") != contract_digest("2")


# ---- registry.py -----------------------------------------------------------
def row(run_id="r1", **kw):
    return {"run_id": run_id, "experiment": "a", "config_hash": "1" * 64, "git_commit": "abc123",
            "data_hash": "missing", "seed": 0, "operator": "het", "n_jobs": 1,
            "start_utc": "2026-09-19T00:00:00Z", "end_utc": "2026-09-19T00:01:00Z", "status": "ok", **kw}


def test_registry_columns():
    assert REGISTRY_COLUMNS == [
        "run_id", "experiment", "config_hash", "git_commit", "data_hash", "seed", "operator",
        "n_jobs", "start_utc", "end_utc", "status",
        "ident_accuracy", "perm_p", "alignment_gain", "nonidentifiable_pairs"]


def test_read_registry_on_a_missing_file_is_an_empty_typed_frame(tmp_path):
    empty = read_registry(tmp_path / "runs" / "index.csv")
    assert list(empty.columns) == REGISTRY_COLUMNS and len(empty) == 0
    append_run(tmp_path / "other.csv", row())
    assert {c: str(t) for c, t in empty.dtypes.items()} == \
        {c: str(t) for c, t in read_registry(tmp_path / "other.csv").dtypes.items()}


def test_append_creates_the_file_with_a_header(tmp_path):
    index = tmp_path / "runs" / "index.csv"
    append_run(index, row())
    lines = index.read_text().splitlines()
    assert lines[0] == ",".join(REGISTRY_COLUMNS) and len(lines) == 2
    df = read_registry(index)
    assert df.loc[0, "run_id"] == "r1" and df.loc[0, "seed"] == 0 and df["seed"].dtype == "Int64"
    assert pd.isna(df.loc[0, "ident_accuracy"])


def test_append_is_idempotent_by_run_id(tmp_path):
    index = tmp_path / "index.csv"
    append_run(index, row("r1", status="first"))
    append_run(index, row("r2"))
    append_run(index, row("r1", status="second", ident_accuracy=0.5))
    df = read_registry(index)
    assert list(df["run_id"]) == ["r1", "r2"]
    assert df.loc[0, "status"] == "second" and df.loc[0, "ident_accuracy"] == 0.5


def test_update_run_metrics(tmp_path):
    index = tmp_path / "index.csv"
    append_run(index, row("r1"))
    append_run(index, row("r2"))
    update_run_metrics(index, "r2", {"ident_accuracy": 0.9, "perm_p": 0.001, "alignment_gain": 0.12,
                                     "nonidentifiable_pairs": 7})
    df = read_registry(index).set_index("run_id")
    assert df.loc["r2", "ident_accuracy"] == 0.9 and df.loc["r2", "nonidentifiable_pairs"] == 7
    assert pd.isna(df.loc["r1", "ident_accuracy"])
    with pytest.raises(KeyError):
        update_run_metrics(index, "missing", {"perm_p": 0.5})
    with pytest.raises(ValueError):
        update_run_metrics(index, "r1", {"seed": 5})


# ---- logging.py ------------------------------------------------------------
def test_run_logger_tees_stdout_and_stderr(tmp_path, capsys):
    out_before, err_before = sys.stdout, sys.stderr
    logger = RunLogger(tmp_path, "myrun")
    print("to stdout")
    print("to stderr", file=sys.stderr)
    logger.log("hello")
    logger.close()
    assert sys.stdout is out_before and sys.stderr is err_before

    text = (tmp_path / "log.txt").read_text()
    assert "to stdout" in text and "to stderr" in text and "[myrun] hello" in text
    captured = capsys.readouterr()
    assert "to stdout" in captured.out and "hello" in captured.out and "to stderr" in captured.err
