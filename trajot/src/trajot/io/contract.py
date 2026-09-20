from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0.0"

# -1 is used as a wildcard dimension in variable-size arrays.
NPZ_SCHEMA: dict[str, tuple[tuple[int, ...] | None, str]] = {
    "connectivity": ((-1, -1), "float64"),
    "timeseries": ((-1, -1), "float32"),
    "embedding": ((-1, -1), "float32"),
    "features": ((-1, -1), "float32"),
    "coords": ((-1, 3), "float32"),
    "tr": (None, "float64"),
    "n_volumes": (None, "int64"),
    "subject_id": (None, "str"),
    "run_id": (None, "str"),
}

MANIFEST_COLUMNS = [
    "subject_id",
    "run_id",
    "path",
    "n_volumes",
    "tr",
    "n_regions",
    "n_vertices",
    "qc_pass",
    "contract_version",
]


class ContractError(ValueError):
    """Raised when a subject-run file does not satisfy the frozen contract."""


@dataclass(frozen=True)
class _NormalizedRow:
    subject_id: str
    run_id: str
    path: str


def subject_run_path(root: Path, subject_id: str, run_id: str) -> Path:
    return root / "derivatives" / "trajot" / f"sub-{subject_id}_run-{run_id}.npz"


def _is_scalar(value: Any) -> bool:
    return np.asarray(value).shape == ()


def _shape_matches(found: tuple[int, ...], expected: tuple[int, ...]) -> bool:
    if len(found) != len(expected):
        return False
    return all(e == -1 or e == f for f, e in zip(found, expected, strict=True))


def _dtype_matches(value: Any, expected: str) -> bool:
    arr = np.asarray(value)
    if expected == "str":
        return arr.dtype.kind in {"U", "S", "O"}
    return arr.dtype == np.dtype(expected)


def _cast_for_schema(key: str, value: Any) -> Any:
    _, dtype = NPZ_SCHEMA[key]
    if dtype == "str":
        return str(np.asarray(value).item())
    if NPZ_SCHEMA[key][0] is None:
        return np.asarray(value, dtype=dtype).item()
    return np.asarray(value, dtype=dtype)


def validate_subject_run(d: Mapping[str, Any]) -> None:
    required = set(NPZ_SCHEMA)
    found = set(d)

    missing = sorted(required - found)
    if missing:
        raise ContractError(f"Missing key(s): {', '.join(missing)}")

    extras = sorted(found - required)
    if extras:
        raise ContractError(f"Unexpected key(s): {', '.join(extras)}")

    for key, (expected_shape, expected_dtype) in NPZ_SCHEMA.items():
        value = d[key]
        arr = np.asarray(value)

        if expected_shape is None:
            if not _is_scalar(value):
                raise ContractError(
                    f"Key '{key}' has shape {arr.shape}, expected scalar with dtype {expected_dtype}"
                )
        elif not _shape_matches(arr.shape, expected_shape):
            raise ContractError(
                f"Key '{key}' has shape {arr.shape}, expected {expected_shape} with dtype {expected_dtype}"
            )

        if not _dtype_matches(value, expected_dtype):
            raise ContractError(
                f"Key '{key}' has dtype {arr.dtype}, expected {expected_dtype}"
            )

    connectivity = np.asarray(d["connectivity"], dtype=np.float64)
    if connectivity.shape[0] != connectivity.shape[1]:
        raise ContractError(
            f"Key 'connectivity' must be square, found shape {connectivity.shape}"
        )
    if not np.allclose(connectivity, connectivity.T, atol=1e-8):
        raise ContractError("Key 'connectivity' is not symmetric")
    if not np.allclose(np.diag(connectivity), 0.0, atol=1e-8):
        raise ContractError("Key 'connectivity' diagonal must be zero")

    timeseries = np.asarray(d["timeseries"])
    embedding = np.asarray(d["embedding"])
    features = np.asarray(d["features"])
    coords = np.asarray(d["coords"])

    n_vertices = timeseries.shape[0]
    if embedding.shape[0] != n_vertices:
        raise ContractError(
            f"Key 'embedding' has first axis {embedding.shape[0]}, expected {n_vertices}"
        )
    if features.shape[0] != n_vertices:
        raise ContractError(
            f"Key 'features' has first axis {features.shape[0]}, expected {n_vertices}"
        )
    if coords.shape[0] != n_vertices:
        raise ContractError(
            f"Key 'coords' has first axis {coords.shape[0]}, expected {n_vertices}"
        )

    n_volumes = int(np.asarray(d["n_volumes"]).item())
    if n_volumes != timeseries.shape[1]:
        raise ContractError(
            f"Key 'n_volumes' is {n_volumes}, but timeseries has T={timeseries.shape[1]}"
        )


def write_subject_run(
    path: Path,
    *,
    connectivity: Any,
    timeseries: Any,
    embedding: Any,
    features: Any,
    coords: Any,
    tr: Any,
    n_volumes: Any,
    subject_id: Any,
    run_id: Any,
) -> Path:
    payload = {
        "connectivity": _cast_for_schema("connectivity", connectivity),
        "timeseries": _cast_for_schema("timeseries", timeseries),
        "embedding": _cast_for_schema("embedding", embedding),
        "features": _cast_for_schema("features", features),
        "coords": _cast_for_schema("coords", coords),
        "tr": _cast_for_schema("tr", tr),
        "n_volumes": _cast_for_schema("n_volumes", n_volumes),
        "subject_id": _cast_for_schema("subject_id", subject_id),
        "run_id": _cast_for_schema("run_id", run_id),
    }
    validate_subject_run(payload)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def read_subject_run(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as handle:
        payload = {key: handle[key] for key in handle.files}

    payload["tr"] = float(np.asarray(payload["tr"]).item())
    payload["n_volumes"] = int(np.asarray(payload["n_volumes"]).item())
    payload["subject_id"] = str(np.asarray(payload["subject_id"]).item())
    payload["run_id"] = str(np.asarray(payload["run_id"]).item())

    validate_subject_run(payload)
    return payload


def write_manifest(rows: Iterable[dict[str, Any]], root: Path) -> Path:
    df = pd.DataFrame(list(rows))
    missing = [col for col in MANIFEST_COLUMNS if col not in df.columns]
    if missing:
        raise ContractError(f"Manifest missing column(s): {', '.join(missing)}")

    df = df[MANIFEST_COLUMNS].copy()
    df = df.astype(
        {
            "subject_id": "string",
            "run_id": "string",
            "path": "string",
            "n_volumes": "int32",
            "tr": "float64",
            "n_regions": "int32",
            "n_vertices": "int32",
            "qc_pass": "bool",
            "contract_version": "string",
        }
    )

    out_path = root / "derivatives" / "trajot" / "manifest.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def read_manifest(root: Path) -> pd.DataFrame:
    path = root / "derivatives" / "trajot" / "manifest.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_parquet(path)


def _normalize_run_id(value: Any) -> str:
    token = str(value)
    token = token.replace("run-", "")
    if token.isdigit():
        return str(int(token))
    return token


def two_run_subjects(manifest: pd.DataFrame, strict: bool = True) -> list[str]:
    for column in ("subject_id", "run_id", "n_volumes"):
        if column not in manifest.columns:
            raise ContractError(f"Manifest missing required column '{column}'")

    hits: list[str] = []
    grouped = manifest.groupby("subject_id", sort=True)
    for subject_id, frame in grouped:
        if len(frame) != 2:
            continue
        if frame["n_volumes"].nunique() != 1:
            continue
        hits.append(str(subject_id))

    if strict and len(hits) != 83:
        raise ContractError(
            f"Expected exactly 83 two-run subjects with equal n_volumes, found {len(hits)}"
        )
    return sorted(hits)


def _iter_manifest_rows(
    root: Path,
    *,
    subjects: Sequence[str] | None,
    run: str,
) -> list[_NormalizedRow]:
    manifest = read_manifest(root)

    if subjects is not None:
        wanted = {str(s) for s in subjects}
        manifest = manifest[manifest["subject_id"].astype(str).isin(wanted)]

    run_norm = _normalize_run_id(run)
    run_col_norm = manifest["run_id"].astype(str).map(_normalize_run_id)
    manifest = manifest[run_col_norm == run_norm]

    if manifest.empty:
        raise ContractError(
            f"No rows found in manifest for run={run!r} and subjects={subjects!r}"
        )

    rows: list[_NormalizedRow] = []
    for _, row in manifest.sort_values(["subject_id", "run_id"]).iterrows():
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = root / path
        rows.append(
            _NormalizedRow(
                subject_id=str(row["subject_id"]),
                run_id=str(row["run_id"]),
                path=str(path),
            )
        )
    return rows


def load_connectomes(
    root: Path,
    subjects: Sequence[str] | None = None,
    run: str = "1",
) -> tuple[np.ndarray, list[str]]:
    rows = _iter_manifest_rows(root, subjects=subjects, run=run)

    mats = []
    subject_ids = []
    for row in rows:
        d = read_subject_run(Path(row.path))
        mats.append(np.asarray(d["connectivity"], dtype=np.float64))
        subject_ids.append(row.subject_id)

    return np.stack(mats, axis=0), subject_ids


def load_embeddings(
    root: Path,
    subjects: Sequence[str] | None = None,
    run: str = "1",
) -> np.ndarray:
    rows = _iter_manifest_rows(root, subjects=subjects, run=run)

    embeddings = [
        np.asarray(read_subject_run(Path(row.path))["embedding"], dtype=np.float32)
        for row in rows
    ]
    return np.stack(embeddings, axis=0)
