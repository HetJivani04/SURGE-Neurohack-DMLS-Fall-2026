from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from trajot.io import contract
from trajot.io.dataset import RunSpec
from trajot.io.preprocess import SurfaceTemplate, qc_path
from trajot.runlog.registry import read_registry

ROOT = Path(__file__).resolve().parents[2]
V, R, T = 60, 6, 50
ALL_FAKE_RUNS = ("--runs", "1,2")  # a filter selects every fake run and skips the full-dataset 83-subject check


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("preprocess_script", ROOT / "scripts" / "preprocess.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def project(tmp_path, script, monkeypatch):
    """configs/ copied to tmp with a paths.yaml; small model/preprocess sizes; runs/ redirected."""
    shutil.copytree(ROOT / "configs", tmp_path / "configs", ignore=shutil.ignore_patterns("paths.yaml"))
    (tmp_path / "configs" / "paths.yaml").write_text(f"data_root: {tmp_path / 'data'}\ncontract_version: '1.0.0'\n")
    for name in ("model/default.yaml",):
        path = tmp_path / "configs" / name
        model = yaml.safe_load(path.read_text())
        model.update(d=3, r=4)
        path.write_text(yaml.safe_dump(model))
    exp = tmp_path / "configs" / "experiments" / "10_ours_full.yaml"
    config = yaml.safe_load(exp.read_text())
    config["preprocess"]["n_regions"] = R
    exp.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(script, "RUNS_DIR", tmp_path / "runs")
    return tmp_path


def fake_surface() -> SurfaceTemplate:
    rng = np.random.default_rng(0)
    half = V // 2
    faces = np.array([[i, i + 1, i + 2] for i in range(half - 2)] + [[half + i, half + i + 1, half + i + 2] for i in range(half - 2)])
    pial = rng.normal(size=(V, 3)) * 30
    return SurfaceTemplate("fake", half, pial, pial * 0.9, faces, rng.normal(size=(V, 2)).astype(np.float32),
                           np.arange(V) % R, R)


def make_specs(root: Path, runs=(("001", "1"), ("001", "2"), ("002", "1"))) -> list[RunSpec]:
    return [RunSpec(s, r, root / f"sub-{s}_{r}.nii.gz", root / "x.json", T, 2.5) for s, r in runs]


@pytest.fixture
def fakes(project, monkeypatch):
    """Replace only the heavy parts: BOLD discovery, preprocess_run, and the nilearn surface template."""
    calls = {"preprocess": []}
    specs = make_specs(project / "data")
    monkeypatch.setattr("trajot.io.dataset.list_bold_runs", lambda _root: specs)
    monkeypatch.setattr("trajot.io.preprocess.load_surface_template", lambda *a, **k: fake_surface())

    def fake_preprocess_run(spec, cfg):
        calls["preprocess"].append((spec.subject_id, spec.run_id))
        rng = np.random.default_rng(int(spec.subject_id) * 10 + int(spec.run_id))
        ts = rng.normal(size=(V, T)).astype(np.float32)
        ts[:5] = 0.0  # vertices outside the field of view
        record = {"qc_pass": spec.subject_id != "002", "bandpass": {
            "low": 0.01, "high": 0.1, "freqs_hz": [0.0, 0.05, 0.1, 0.15, 0.2], "psd_before": [1, 1, 1, 1, 1],
            "psd_after": [0.1, 1, 0.8, 1e-3, 1e-6], "in_band_power_ratio": 0.8, "out_of_band_power_ratio": 1e-4}}
        path = qc_path(Path(cfg.data_root), spec.subject_id, spec.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
        return ts, fake_surface().coords, 2.5

    monkeypatch.setattr("trajot.io.preprocess.preprocess_run", fake_preprocess_run)
    return calls, specs


def cli(project: Path, *extra: str) -> list[str]:
    return ["--config", str(project / "configs" / "experiments" / "10_ours_full.yaml"), *extra]


# ---------------------------------------------------------------------------
def test_dry_run_lists_runs_and_creates_nothing(script, project, fakes, capsys) -> None:
    assert script.main(cli(project, "--dry-run", "--subjects", "001")) == 0
    out = capsys.readouterr().out
    assert "would process sub-001 run-1" in out and "would process sub-001 run-2" in out and "sub-002" not in out
    assert not (project / "runs").exists() and not (project / "data").exists()


def test_main_writes_the_contract_outputs_manifest_and_run_record(script, project, fakes) -> None:
    calls, _ = fakes
    assert script.main(cli(project, *ALL_FAKE_RUNS)) == 0

    root = project / "data"
    out = root / "derivatives" / "trajot"
    for subject, run in (("001", "1"), ("001", "2"), ("002", "1")):
        d = contract.read_subject_run(contract.subject_run_path(root, subject, run))  # validates the frozen schema
        assert d["timeseries"].shape == (V, T) and d["embedding"].shape == (V, 3) and d["connectivity"].shape == (R, R)
        assert np.all(d["embedding"][:5] == 0) and np.any(d["embedding"][5:] != 0)  # out-of-view vertices stay zero
        geometry = np.load(out / f"sub-{subject}_run-{run}_geometry.npz")
        assert geometry["A"].shape == (R, 4) and 0 < float(geometry["retained_variance"]) <= 1
        assert geometry["mass"].sum() == 1.0 and np.all(geometry["mass"][:5] == 0)  # uniform over valid vertices
        assert np.allclose(geometry["mass"][5:], 1.0 / (V - 5))
    template = np.load(out / "template_geometry.npz")
    assert template["anatomical_cost"].shape == (V, V) and template["anatomical_cost"].dtype == np.float64

    manifest = contract.read_manifest(root)
    assert len(manifest) == 3 and list(manifest["n_regions"]) == [R] * 3 and list(manifest["n_vertices"]) == [V] * 3
    assert dict(zip(manifest["subject_id"] + "_" + manifest["run_id"], manifest["qc_pass"])) == {
        "001_1": True, "001_2": True, "002_1": False}
    assert contract.two_run_subjects(manifest, strict=False) == ["001"]
    assert len(contract.load_connectomes(root, run="1")[1]) == 2

    (run_dir,) = [p for p in (project / "runs").iterdir() if p.is_dir()]
    assert run_dir.name.startswith("preprocess__")
    record = json.loads((run_dir / "manifest.json").read_text())
    assert record["experiment"] == "preprocess" and record["end_utc"] and record["config"]["preprocess"]["n_regions"] == R
    assert (run_dir / "artifacts" / "bandpass_check.png").stat().st_size > 1000
    assert "band-pass check" in (run_dir / "log.txt").read_text()
    assert list(read_registry(project / "runs" / "index.csv")["status"]) == ["ok"]
    assert len(calls["preprocess"]) == 3


def test_main_is_resumable_and_force_reprocesses(script, project, fakes) -> None:
    calls, _ = fakes
    assert script.main(cli(project, *ALL_FAKE_RUNS)) == 0
    assert len(calls["preprocess"]) == 3
    assert script.main(cli(project, *ALL_FAKE_RUNS)) == 0  # everything is written: nothing to redo
    assert len(calls["preprocess"]) == 3
    assert script.main(cli(project, "--force", "--subjects", "002")) == 0
    assert len(calls["preprocess"]) == 4


def test_an_interrupted_run_is_not_treated_as_finished(script, project, fakes) -> None:
    calls, specs = fakes
    assert script.main(cli(project, "--subjects", "002")) == 0
    root = project / "data"
    contract.subject_run_path(root, "002", "1").unlink()  # the contract npz is written last; a kill leaves the rest
    (contract.subject_run_path(root, "002", "1").with_name("sub-002_run-1.partial.npz")).write_bytes(b"truncated")
    assert not script._is_done(root, specs[2])
    assert script.main(cli(project, "--subjects", "002")) == 0
    assert len(calls["preprocess"]) == 2
    assert contract.read_subject_run(contract.subject_run_path(root, "002", "1"))["subject_id"] == "002"


def test_rebuild_manifest_ignores_sidecar_and_partial_files(script, project) -> None:
    """Regression: the *_factor.npz sidecar matched the glob and crashed the manifest with KeyError."""
    root = project / "data"
    rng = np.random.default_rng(0)
    for subject, run in (("001", "1"), ("001", "2"), ("002", "1")):
        contract.write_subject_run(
            contract.subject_run_path(root, subject, run), connectivity=np.zeros((R, R)),
            timeseries=rng.normal(size=(V, T)), embedding=np.zeros((V, 3)), features=np.zeros((V, 2)),
            coords=np.zeros((V, 3)), tr=2.5, n_volumes=T, subject_id=subject, run_id=run)
    out = root / "derivatives" / "trajot"
    np.savez(out / "sub-001_run-1_geometry.npz", A=np.zeros((R, 4)))
    np.savez(out / "sub-001_run-1.partial.npz", junk=np.zeros(3))
    np.savez(out / "template_geometry.npz", anatomical_cost=np.zeros((V, V)))
    (out / "sub-001_run-1_qc.json").write_text(json.dumps({"qc_pass": True}))

    script._rebuild_manifest(root)
    manifest = contract.read_manifest(root)
    assert len(manifest) == 3
    assert dict(zip(manifest["subject_id"] + "_" + manifest["run_id"], manifest["qc_pass"])) == {
        "001_1": True, "001_2": False, "002_1": False}  # no QC record: not passed


def test_filters_and_csv_parsing(script) -> None:
    specs = make_specs(Path("x"))
    assert script._parse_csv(None) is None and script._parse_csv(" ,") is None
    assert script._parse_csv("001, 002,") == {"001", "002"}
    assert [(s.subject_id, s.run_id) for s in script._filter_specs(specs, subjects={"001"}, runs={"2"})] == [("001", "2")]
    assert len(script._filter_specs(specs, subjects=None, runs=None)) == 3


def test_a_full_run_checks_the_83_two_run_subjects(script, project, fakes) -> None:
    import sys

    stdout = sys.stdout
    with pytest.raises(contract.ContractError, match="83"):
        script.main(cli(project))  # unfiltered, so the manifest must contain exactly 83 two-run subjects
    assert sys.stdout is stdout  # the run logger was released despite the failure


def test_a_stale_contract_version_pin_fails_fast(script, project, fakes) -> None:
    (project / "configs" / "paths.yaml").write_text(f"data_root: {project / 'data'}\ncontract_version: '0.9'\n")
    with pytest.raises(contract.ContractError, match="contract_version"):
        script.main(cli(project, *ALL_FAKE_RUNS))
    assert not (project / "data").exists() and not (project / "runs").exists()


def test_the_preflight_reports_ram_and_aborts_when_it_is_short(script, project, fakes, monkeypatch) -> None:
    import psutil
    from collections import namedtuple

    memory = namedtuple("memory", "available")
    lines: list[str] = []
    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory(64 * 2**30))
    script._preflight(4, lines.append)
    assert "available RAM 64.0 GiB" in lines[0] and "8.0 GiB needed for 4 worker" in lines[0]

    monkeypatch.setattr(psutil, "virtual_memory", lambda: memory(3 * 2**30))
    with pytest.raises(MemoryError, match="lower --n-jobs"):
        script._preflight(4, lines.append)

    with pytest.raises(MemoryError, match="only 3.0 GiB"):
        script.main(cli(project, *ALL_FAKE_RUNS, "--n-jobs", "4"))
    assert not list((project / "data").glob("derivatives/trajot/*.npz")) if (project / "data").exists() else True


def test_the_script_pins_threads_before_numpy_is_imported() -> None:
    import subprocess
    import sys

    code = (
        "import sys, importlib.util; "
        f"spec = importlib.util.spec_from_file_location('p', {str(ROOT / 'scripts' / 'preprocess.py')!r}); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "assert 'numpy' not in sys.modules and 'pandas' not in sys.modules, 'numpy imported at module level'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_region_outside_the_field_of_view_gives_zero_connectome_rows_not_a_crash(script, project, fakes, monkeypatch) -> None:
    """Regression: an empty atlas region raised ValueError and stopped the whole batch."""
    calls, _ = fakes
    inner = __import__("trajot.io.preprocess", fromlist=["preprocess_run"]).preprocess_run

    def missing_region(spec, cfg):
        ts, coords, tr = inner(spec, cfg)
        ts[fake_surface().labels == 2] = 0.0  # every vertex of region 2 is outside the field of view
        return ts, coords, tr

    monkeypatch.setattr("trajot.io.preprocess.preprocess_run", missing_region)
    assert script.main(cli(project, *ALL_FAKE_RUNS)) == 0

    d = contract.read_subject_run(contract.subject_run_path(project / "data", "001", "1"))
    C = d["connectivity"]
    assert np.all(C[2] == 0) and np.all(C[:, 2] == 0)  # the uncovered region carries no connectivity
    assert np.count_nonzero(C[np.arange(R) != 2][:, np.arange(R) != 2]) > 0  # the others are unaffected
    assert np.allclose(C, C.T) and np.all(np.diag(C) == 0)


def test_region_series_matches_parcellate_when_every_region_has_data(script) -> None:
    from trajot.geometry.connectivity import parcellate

    rng = np.random.default_rng(0)
    ts = rng.normal(size=(V, T)).astype(np.float32)
    labels = np.arange(V) % R
    valid = np.ones(V, dtype=bool)
    assert np.array_equal(script._region_series(ts, labels, valid, R), parcellate(ts, labels, R))
