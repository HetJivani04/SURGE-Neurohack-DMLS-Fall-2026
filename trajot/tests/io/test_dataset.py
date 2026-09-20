from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from trajot.io.dataset import RunSpec, iter_bold_chunks, list_bold_runs, read_bold_geometry, read_bold_json


def _write_run(root: Path, subject: str = "001", run: str = "1") -> tuple[Path, Path]:
    func_dir = root / f"sub-{subject}" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)

    bold_path = func_dir / f"sub-{subject}_task-rest_run-{run}_bold.nii.gz"
    data = np.random.default_rng(0).normal(size=(4, 4, 32, 12)).astype(np.float32)
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), str(bold_path))

    json_path = func_dir / f"sub-{subject}_task-rest_run-{run}_bold.json"
    slice_timing = [float(i) * (2.5 / 32.0) for i in range(32)]
    json_path.write_text(
        "{\n"
        '  "RepetitionTime": 2.5,\n'
        '  "EchoTime": 0.027,\n'
        f'  "SliceTiming": {slice_timing}\n'
        "}\n",
        encoding="utf-8",
    )
    return bold_path, json_path


def test_read_bold_json_requires_slice_timing_len_32(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"RepetitionTime": 2.5, "EchoTime": 0.027, "SliceTiming": [0.0, 0.1]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SliceTiming"):
        read_bold_json(bad)


def test_list_bold_runs_reads_nvolumes_and_tr(tmp_path: Path) -> None:
    _write_run(tmp_path, subject="001", run="1")
    specs = list_bold_runs(tmp_path)

    assert len(specs) == 1
    spec = specs[0]
    assert spec.subject_id == "001"
    assert spec.run_id == "1"
    assert spec.n_volumes == 12
    assert spec.tr == 2.5


def test_iter_bold_chunks_streams_float32_blocks(tmp_path: Path) -> None:
    bold_path, json_path = _write_run(tmp_path, subject="002", run="2")
    spec = RunSpec(
        subject_id="002",
        run_id="2",
        bold_path=bold_path,
        json_path=json_path,
        n_volumes=12,
        tr=2.5,
    )

    chunks = list(iter_bold_chunks(spec, chunk=20))
    assert chunks
    assert all(c.dtype == np.float32 for c in chunks)

    stacked = np.vstack(chunks)
    assert stacked.shape[1] == 12
    assert stacked.shape[0] == 4 * 4 * 32


def test_read_bold_json_slice_timing_is_optional_only_when_relaxed(tmp_path: Path) -> None:
    """The real ds000243 sidecar has no SliceTiming: strict by default, relaxed on request."""
    sidecar = tmp_path / "task-rest_bold.json"
    sidecar.write_text('{"RepetitionTime": 2.5, "EchoTime": 0.027}', encoding="utf-8")

    with pytest.raises(ValueError, match="SliceTiming"):
        read_bold_json(sidecar)
    assert read_bold_json(sidecar, require_slice_timing=False)["RepetitionTime"] == 2.5

    sidecar.write_text('{"RepetitionTime": 2.5}', encoding="utf-8")
    with pytest.raises(ValueError, match="EchoTime"):
        read_bold_json(sidecar, require_slice_timing=False)  # the other keys stay required

    sidecar.write_text('{"RepetitionTime": 2.5, "EchoTime": 0.027, "SliceTiming": [0.0]}', encoding="utf-8")
    with pytest.raises(ValueError, match="length 32"):
        read_bold_json(sidecar, require_slice_timing=False)  # a SliceTiming that is present is still validated


def test_list_bold_runs_works_on_a_dataset_level_sidecar_without_slice_timing(tmp_path: Path) -> None:
    func_dir = tmp_path / "sub-007" / "func"
    func_dir.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 32, 9), dtype=np.float32), np.eye(4)),
             str(func_dir / "sub-007_task-rest_run-2_bold.nii.gz"))
    (tmp_path / "task-rest_bold.json").write_text('{"RepetitionTime": 2.5, "EchoTime": 0.027}', encoding="utf-8")

    (spec,) = list_bold_runs(tmp_path)
    assert (spec.subject_id, spec.run_id, spec.n_volumes, spec.tr) == ("007", "2", 9, 2.5)
    assert spec.json_path == tmp_path / "task-rest_bold.json"


def test_read_bold_geometry_reads_only_the_header(tmp_path: Path) -> None:
    bold_path, json_path = _write_run(tmp_path)
    spec = RunSpec("001", "1", bold_path, json_path, 12, 2.5)
    shape, voxel_size, affine = read_bold_geometry(spec)
    assert shape == (4, 4, 32, 12)
    assert voxel_size == (1.0, 1.0, 1.0)
    assert affine.shape == (4, 4) and np.allclose(affine, np.eye(4))


def test_iter_bold_chunks_concatenate_to_the_c_order_flattened_run(tmp_path: Path) -> None:
    """Block k is a run of consecutive voxels; the pipeline relies on index = (x * Y + y) * Z + z."""
    bold_path, json_path = _write_run(tmp_path)
    spec = RunSpec("001", "1", bold_path, json_path, 12, 2.5)
    data = np.asarray(nib.load(str(bold_path)).dataobj, dtype=np.float32)
    for chunk in (1, 20, 200, 10**6):
        stacked = np.concatenate(list(iter_bold_chunks(spec, chunk=chunk)), axis=0)
        assert np.array_equal(stacked, data.reshape(-1, 12))
