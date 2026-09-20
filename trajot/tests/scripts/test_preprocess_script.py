from __future__ import annotations

from pathlib import Path

from trajot.io.dataset import RunSpec

import scripts.preprocess as preprocess_script


def _dummy_spec(tmp_path: Path) -> RunSpec:
    return RunSpec(
        subject_id="001",
        run_id="1",
        bold_path=tmp_path / "dummy.nii.gz",
        json_path=tmp_path / "dummy.json",
        n_volumes=10,
        tr=2.5,
    )


def test_main_dry_run(monkeypatch, tmp_path: Path) -> None:
    spec = _dummy_spec(tmp_path)
    monkeypatch.setattr(preprocess_script, "list_bold_runs", lambda _root: [spec])

    rc = preprocess_script.main(["--data-root", str(tmp_path), "--dry-run"])
    assert rc == 0


def test_main_calls_write_and_manifest(monkeypatch, tmp_path: Path) -> None:
    spec = _dummy_spec(tmp_path)

    calls = {"write": 0, "manifest": 0}

    monkeypatch.setattr(preprocess_script, "list_bold_runs", lambda _root: [spec])

    def _fake_write(root: Path, s: RunSpec, force: bool, cfg):
        calls["write"] += 1
        assert root == tmp_path
        assert s.subject_id == "001"
        return tmp_path / "derivatives" / "trajot" / "sub-001_run-1.npz"

    def _fake_manifest(root: Path):
        calls["manifest"] += 1
        assert root == tmp_path
        return tmp_path / "derivatives" / "trajot" / "manifest.parquet"

    monkeypatch.setattr(preprocess_script, "_write_one", _fake_write)
    monkeypatch.setattr(preprocess_script, "_rebuild_manifest", _fake_manifest)

    rc = preprocess_script.main(["--data-root", str(tmp_path), "--subjects", "001", "--runs", "1"])

    assert rc == 0
    assert calls["write"] == 1
    assert calls["manifest"] == 1
