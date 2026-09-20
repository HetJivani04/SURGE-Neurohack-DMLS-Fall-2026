from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trajot.inference.synthetic import make_synthetic_npz, read_truth
from trajot.io import contract


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    return make_synthetic_npz(tmp_path_factory.mktemp("synthetic"), n_subjects=5, V=60, R=20, T=64, d=8, seed=3)


def test_every_file_satisfies_the_frozen_contract_unchanged(dataset: Path) -> None:
    manifest = contract.read_manifest(dataset)
    assert len(manifest) == 10 and list(manifest.columns) == contract.MANIFEST_COLUMNS
    for row in manifest.itertuples():
        d = contract.read_subject_run(dataset / row.path)  # read_subject_run validates against NPZ_SCHEMA
        contract.validate_subject_run(d)
        assert d["connectivity"].shape == (20, 20) and d["timeseries"].shape == (60, 64)
        assert d["embedding"].shape == (60, 8) and d["features"].shape == (60, 2) and d["coords"].shape == (60, 3)
        assert d["timeseries"].dtype == np.float32 and d["connectivity"].dtype == np.float64
        assert (row.n_regions, row.n_vertices, row.n_volumes) == (20, 60, 64)
    assert contract.two_run_subjects(manifest, strict=False) == ["001", "002", "003", "004", "005"]


def test_default_shapes_match_the_documented_signature(tmp_path: Path) -> None:
    make_synthetic_npz(tmp_path, n_subjects=1)  # V=200, R=100, T=132, d=32
    d = contract.read_subject_run(contract.subject_run_path(tmp_path, "001", "1"))
    assert d["connectivity"].shape == (100, 100) and d["timeseries"].shape == (200, 132) and d["embedding"].shape == (200, 32)


def test_the_planted_permutation_is_recorded_and_the_data_follow_it(dataset: Path) -> None:
    truth = read_truth(dataset)
    perm = truth["perm"]
    assert perm.shape == (5, 60) and all(sorted(p) == list(range(60)) for p in perm)
    assert not np.array_equal(perm[0], perm[1]) and bool(truth["planted"])

    # vertex i of subject s carries the features of template node perm[s][i]
    d = contract.read_subject_run(contract.subject_run_path(dataset, "002", "1"))
    expected = truth["F_true"][perm[1]]
    assert np.corrcoef(d["embedding"].ravel(), expected.ravel())[0, 1] > 0.8


def test_the_two_runs_of_a_subject_are_alike_but_not_identical(dataset: Path) -> None:
    r1 = contract.read_subject_run(contract.subject_run_path(dataset, "001", "1"))
    r2 = contract.read_subject_run(contract.subject_run_path(dataset, "001", "2"))
    other = contract.read_subject_run(contract.subject_run_path(dataset, "002", "2"))
    iu = np.triu_indices(20, 1)
    same, different = (np.corrcoef(a["connectivity"][iu], b["connectivity"][iu])[0, 1] for a, b in ((r1, r2), (r1, other)))
    assert not np.array_equal(r1["timeseries"], r2["timeseries"]) and np.array_equal(r1["coords"], r2["coords"])
    assert same > different  # scan-rescan reliability


def test_without_a_planted_permutation_every_subject_has_the_identity(tmp_path: Path) -> None:
    make_synthetic_npz(tmp_path, n_subjects=3, V=30, R=10, T=32, d=4, planted_permutation=False)
    perm = read_truth(tmp_path)["perm"]
    assert np.array_equal(perm, np.tile(np.arange(30), (3, 1)))
    contract.read_manifest(tmp_path)


def test_generation_is_deterministic_in_the_seed(tmp_path: Path) -> None:
    a = make_synthetic_npz(tmp_path / "a", n_subjects=2, V=20, R=5, T=16, d=3, seed=7)
    b = make_synthetic_npz(tmp_path / "b", n_subjects=2, V=20, R=5, T=16, d=3, seed=7)
    c = make_synthetic_npz(tmp_path / "c", n_subjects=2, V=20, R=5, T=16, d=3, seed=8)
    load = lambda root: contract.read_subject_run(contract.subject_run_path(root, "001", "1"))["timeseries"]  # noqa: E731
    assert np.array_equal(load(a), load(b)) and not np.array_equal(load(a), load(c))


def test_more_regions_than_vertices_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regions"):
        make_synthetic_npz(tmp_path, n_subjects=1, V=10, R=20)
