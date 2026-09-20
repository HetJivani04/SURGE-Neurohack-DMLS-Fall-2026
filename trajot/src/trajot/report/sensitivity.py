"""Scan-length sensitivity scaffolding (PLAN section 6.1): the long-run subset, and the metrics reported on it.

The long runs of ds000243 (15 x 360, 5 x 480 and 6 x 724 volumes) belong to one-run subjects, so they cannot be
identified across runs, and the metrics for the subset are evaluated separately from the headline table. Such a
run is registered under ``<experiment>_long`` (``SENSITIVITY_SUFFIX``); it never becomes a headline row, and every
row of ``scan_length_metrics`` carries ``SENSITIVITY_LABEL`` so it cannot be read as a headline number.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from trajot.report.table import ROW_METHODS, TABLE_EXPERIMENTS, TABLE_ROWS, MetricsError, latest_successful_run, load_run_metrics
from trajot.runlog.registry import read_registry

SENSITIVITY_SUFFIX = "_long"
SENSITIVITY_LABEL = "sensitivity: long-run subset (not a headline number)"
SENSITIVITY_COLUMNS = ["analysis", "method", "experiment", "run_id", "n_subjects", "ident_accuracy", "ident_ci_low",
                       "ident_ci_high", "perm_p", "alignment_gain", "nonidentifiable_pairs", "per_pair_uncertainty"]


def long_run_subset(manifest: pd.DataFrame, min_volumes: int = 300) -> list[str]:
    """Sorted ids of the subjects with at least one run of ``min_volumes`` or more volumes (26 in ds000243)."""
    if isinstance(min_volumes, bool) or not isinstance(min_volumes, int) or min_volumes < 1:
        raise ValueError(f"min_volumes must be a positive integer, got {min_volumes!r}")
    for column in ("subject_id", "n_volumes"):
        if column not in manifest.columns:
            raise ValueError(f"manifest has no {column!r} column")
    return sorted({str(s) for s in manifest.loc[manifest["n_volumes"] >= min_volumes, "subject_id"]})


def scan_length_metrics(index_path: Path, subset: Iterable[str], experiments: Iterable[str] | None = None,
                        *, include_fake: bool = False) -> pd.DataFrame:
    """The metrics of the runs evaluated on ``subset``, one row per (run, method), flagged as a sensitivity analysis.

    For each of ``experiments`` (the headline experiment names; default ``TABLE_EXPERIMENTS``) it takes the latest
    successful run of ``<experiment>_long``, leaving out fake runs unless ``include_fake``. Such a run must have
    been evaluated on exactly the ``len(subset)`` subjects of the subset (``MetricsError`` otherwise). Nothing to
    report is an empty frame with the columns, not an error.
    """
    subset = [str(s) for s in subset]
    if not subset:
        raise ValueError("the long-run subset is empty")
    index_path = Path(index_path)
    records = read_registry(index_path).to_dict("records")
    rows = []
    for name in TABLE_EXPERIMENTS if experiments is None else list(experiments):
        entry = latest_successful_run(records, name + SENSITIVITY_SUFFIX, include_fake)
        if entry is None:
            continue
        run_id = entry["run_id"]
        metrics = load_run_metrics(index_path.parent, run_id)
        n_subjects = metrics.get("n_subjects")
        if n_subjects != len(subset):
            raise MetricsError(f"run {run_id}: n_subjects: evaluated on {n_subjects!r} subjects "
                               f"but the long-run subset has {len(subset)}")
        for label in TABLE_ROWS:
            stats = metrics["methods"].get(ROW_METHODS[label])
            if stats is None:
                continue
            uncertainty = stats["per_pair_uncertainty"]
            rows.append({
                "analysis": SENSITIVITY_LABEL, "method": label, "experiment": entry["experiment"], "run_id": run_id,
                "n_subjects": n_subjects, "ident_accuracy": stats["ident_accuracy"], "ident_ci_low": stats["ident_ci"][0],
                "ident_ci_high": stats["ident_ci"][1], "perm_p": stats["perm_p"], "alignment_gain": stats["alignment_gain"],
                "nonidentifiable_pairs": stats["nonidentifiable_pairs"],
                "per_pair_uncertainty": float("nan") if uncertainty is None else uncertainty,
            })
    return pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)
