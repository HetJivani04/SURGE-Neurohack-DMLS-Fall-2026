"""Issue #3's acceptance criteria against the real ds000243, skipped where the data is not present.

The raw checks need the dataset at ``data_root`` (see ``configs/paths.yaml``); the derivative checks
also need ``python scripts/preprocess.py`` to have run over all subjects.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from trajot.config import load_config
from trajot.io import contract
from trajot.io.dataset import list_bold_runs, read_bold_json

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def data_root() -> Path:
    try:
        root = load_config(ROOT / "configs" / "experiments" / "10_ours_full.yaml").data_root
    except FileNotFoundError:
        pytest.skip("configs/paths.yaml is missing")
    if not (root / "sub-001" / "func").exists() or not any(root.glob("sub-001/func/*_bold.nii.gz")):
        pytest.skip(f"ds000243 is not downloaded at {root}")
    if not (root / "sub-001" / "func" / "sub-001_task-rest_run-1_bold.nii.gz").exists():
        pytest.skip("the BOLD files are unfetched annex links")
    return root


def test_the_dataset_has_the_documented_runs(data_root: Path) -> None:
    specs = list_bold_runs(data_root)
    per_subject = Counter(s.subject_id for s in specs)
    assert len(specs) == 203 and len(per_subject) == 120
    assert sum(n == 2 for n in per_subject.values()) == 83 and sum(n == 1 for n in per_subject.values()) == 37

    lengths = {}
    for s in specs:
        lengths.setdefault(s.subject_id, set()).add(s.n_volumes)
    assert sum(len(v) == 1 for k, v in lengths.items() if per_subject[k] == 2) == 83  # equal within every pair
    volumes = sorted(s.n_volumes for s in specs)
    assert (volumes[len(volumes) // 2], volumes[0], volumes[-1]) == (132, 130, 724)
    assert {s.tr for s in specs} == {2.5}
    assert "SliceTiming" not in read_bold_json(data_root / "task-rest_bold.json", require_slice_timing=False)


def test_the_derivatives_satisfy_the_contract_for_all_subjects(data_root: Path) -> None:
    if not (data_root / "derivatives" / "trajot" / "manifest.parquet").exists():
        pytest.skip("scripts/preprocess.py has not been run")
    manifest = contract.read_manifest(data_root)
    if len(manifest) != 203:
        pytest.skip(f"preprocessing incomplete: {len(manifest)} of 203 runs")

    assert len(contract.two_run_subjects(manifest)) == 83  # strict: exactly 83, equal n_volumes within each
    assert manifest.groupby("subject_id").size().isin([1, 2]).all()
    assert list(manifest.columns) == contract.MANIFEST_COLUMNS

    for row in manifest.itertuples():
        d = contract.read_subject_run(data_root / row.path)  # validates every array against NPZ_SCHEMA
        assert (str(d["subject_id"]), str(d["run_id"])) == (row.subject_id, row.run_id)
        assert int(d["n_volumes"]) == row.n_volumes == d["timeseries"].shape[1]
        assert d["connectivity"].shape == (row.n_regions, row.n_regions) and d["timeseries"].shape[0] == row.n_vertices

    connectomes, subjects = contract.load_connectomes(data_root, run="1")
    assert connectomes.shape == (120, 100, 100) and len(subjects) == 120
    assert contract.load_embeddings(data_root, run="1").shape == (120, 5124, 32)
    assert np.isfinite(connectomes).all()
