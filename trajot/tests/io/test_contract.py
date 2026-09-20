from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trajot.io.contract import (
    ContractError,
    NPZ_SCHEMA,
    SCHEMA_VERSION,
    load_connectomes,
    load_embeddings,
    two_run_subjects,
    validate_subject_run,
    write_manifest,
    write_subject_run,
)


def _sample_subject_run(subject_id: str = "001", run_id: str = "1") -> dict[str, object]:
    V, T, R, d, F = 12, 20, 5, 4, 2
    rng = np.random.default_rng(0)

    raw = rng.normal(size=(R, R))
    conn = 0.5 * (raw + raw.T)
    np.fill_diagonal(conn, 0.0)

    return {
        "connectivity": conn.astype(np.float64),
        "timeseries": rng.normal(size=(V, T)).astype(np.float32),
        "embedding": rng.normal(size=(V, d)).astype(np.float32),
        "features": rng.normal(size=(V, F)).astype(np.float32),
        "coords": rng.normal(size=(V, 3)).astype(np.float32),
        "tr": np.float64(2.5),
        "n_volumes": int(T),
        "subject_id": subject_id,
        "run_id": run_id,
    }


def test_contract_roundtrip_preserves_keys_shapes_and_dtypes(tmp_path: Path) -> None:
    payload = _sample_subject_run()
    out_path = tmp_path / "derivatives" / "trajot" / "sub-001_run-1.npz"

    write_subject_run(out_path, **payload)

    from trajot.io.contract import read_subject_run

    loaded = read_subject_run(out_path)

    assert set(loaded.keys()) == set(NPZ_SCHEMA.keys())
    for key, (expected_shape, expected_dtype) in NPZ_SCHEMA.items():
        value = np.asarray(loaded[key])
        if expected_shape is None:
            assert value.shape == ()
        else:
            assert len(value.shape) == len(expected_shape)
        if expected_dtype == "str":
            assert value.dtype.kind in {"U", "S", "O"}
        else:
            assert value.dtype == np.dtype(expected_dtype)


def test_contract_missing_key_fails_loudly_with_key_name() -> None:
    payload = _sample_subject_run()
    payload.pop("coords")

    with pytest.raises(ContractError, match="coords"):
        validate_subject_run(payload)


def test_two_run_subjects_returns_expected_count() -> None:
    rows = []
    for sid in range(1, 84):
        subject_id = f"{sid:03d}"
        rows.append({"subject_id": subject_id, "run_id": "1", "n_volumes": 132})
        rows.append({"subject_id": subject_id, "run_id": "2", "n_volumes": 132})

    rows.append({"subject_id": "999", "run_id": "1", "n_volumes": 132})

    manifest = pd.DataFrame(rows)
    result = two_run_subjects(manifest, strict=True)

    assert len(result) == 83
    assert result[0] == "001"
    assert result[-1] == "083"


def test_load_connectomes_and_embeddings(tmp_path: Path) -> None:
    root = tmp_path

    rows = []
    for subject_id in ("001", "002"):
        payload = _sample_subject_run(subject_id=subject_id, run_id="1")
        npz_path = root / "derivatives" / "trajot" / f"sub-{subject_id}_run-1.npz"
        write_subject_run(npz_path, **payload)

        rows.append(
            {
                "subject_id": subject_id,
                "run_id": "1",
                "path": str(npz_path.relative_to(root)),
                "n_volumes": payload["n_volumes"],
                "tr": payload["tr"],
                "n_regions": payload["connectivity"].shape[0],
                "n_vertices": payload["timeseries"].shape[0],
                "qc_pass": True,
                "contract_version": SCHEMA_VERSION,
            }
        )

    write_manifest(rows, root)

    connectomes, subjects = load_connectomes(root, run="1")
    assert connectomes.shape[0] == 2
    assert connectomes.dtype == np.float64
    assert subjects == ["001", "002"]

    embeddings = load_embeddings(root, run="1")
    assert embeddings.shape[0] == 2
    assert embeddings.dtype == np.float32
