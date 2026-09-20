"""``runs/index.csv``: one row per run."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

METRIC_COLUMNS = ["ident_accuracy", "perm_p", "alignment_gain", "nonidentifiable_pairs"]
REGISTRY_COLUMNS = [
    "run_id", "experiment", "config_hash", "git_commit", "data_hash", "seed", "operator",
    "n_jobs", "start_utc", "end_utc", "status", *METRIC_COLUMNS,
]
_INT_COLUMNS = ["seed", "n_jobs", "nonidentifiable_pairs"]
_FLOAT_COLUMNS = ["ident_accuracy", "perm_p", "alignment_gain"]


def _read_rows(index_path: Path) -> list[dict[str, str]]:
    if not index_path.is_file():
        return []
    with index_path.open(newline="") as fh:
        return [{c: row.get(c) or "" for c in REGISTRY_COLUMNS} for row in csv.DictReader(fh)]


def _write_rows(index_path: Path, rows: list[dict[str, str]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def append_run(index_path: Path, row: dict[str, Any]) -> None:
    """Append one row, creating the file with a header row if absent.

    Idempotent by ``run_id``: a re-run with the same ``run_id`` replaces its row rather
    than duplicating it.
    """
    index_path = Path(index_path)
    new = {c: _text(row.get(c)) for c in REGISTRY_COLUMNS}
    rows = _read_rows(index_path)
    for i, existing in enumerate(rows):
        if existing["run_id"] == new["run_id"]:
            rows[i] = new
            break
    else:
        rows.append(new)
    _write_rows(index_path, rows)


def read_registry(index_path: Path) -> pd.DataFrame:
    """Typed read; an empty DataFrame with ``REGISTRY_COLUMNS`` when the file is absent."""
    frame = pd.DataFrame(_read_rows(Path(index_path)), columns=REGISTRY_COLUMNS, dtype=object)
    for col in _FLOAT_COLUMNS:
        frame[col] = pd.to_numeric(frame[col].replace("", None), errors="coerce").astype("float64")
    for col in _INT_COLUMNS:
        frame[col] = pd.to_numeric(frame[col].replace("", None), errors="coerce").astype("Int64")
    return frame


def update_run_metrics(index_path: Path, run_id: str, metrics: dict[str, Any]) -> None:
    """Patch the metric columns of one existing row."""
    index_path = Path(index_path)
    unknown = set(metrics) - set(METRIC_COLUMNS)
    if unknown:
        raise ValueError(f"not metric columns: {sorted(unknown)}")
    rows = _read_rows(index_path)
    for row in rows:
        if row["run_id"] == run_id:
            row.update({k: _text(v) for k, v in metrics.items()})
            break
    else:
        raise KeyError(f"run_id {run_id!r} not found in {index_path}")
    _write_rows(index_path, rows)
