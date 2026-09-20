"""The comparison table of PLAN section 7.4: validate each run's ``metrics.json``, pick runs, render 6 rows x 5 columns.

Labels, order, and the empty-cell rule are data here, so the table can never come out short or reordered. The two
right-hand columns are quantities no baseline produces: they render as ``EMPTY_CELL`` for every baseline row.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable

from trajot.runlog.registry import read_registry

TABLE_ROWS = ["No alignment", "BrainSync", "FUGW", "connectivity-SRM", "Ours (ablated)", "Ours (full)"]
TABLE_COLUMNS = ["method", "ident_accuracy", "perm_p", "per_pair_uncertainty", "nonidentifiable_pairs"]
EMPTY_CELL = "—"

# Markdown header texts, in TABLE_COLUMNS order (the layout of the table in the W4 issue).
COLUMN_TITLES = ["Method", "Identification acc.", "vs null (p)", "Per-pair uncertainty", "Non-identifiable pairs flagged"]

# The key each row has under metrics.json["methods"]: W3's baseline names, and "full" for the full model, which has
# no W3 entry yet. Only the two model rows may fill the two model-only columns; every other method key is a baseline.
ROW_METHODS = dict(zip(TABLE_ROWS, ["noalign", "brainsync", "fugw", "conn_srm", "ablated", "full"]))
MODEL_ROWS = TABLE_ROWS[4:]
MODEL_METHODS = {ROW_METHODS[label] for label in MODEL_ROWS}

# The registry experiment that produces each row (the experiment names of W0's experiment script), in row order.
TABLE_EXPERIMENTS = ["00_noalign", "01_brainsync", "02_fugw", "03_conn_srm", "11_ours_ablated", "10_ours_full"]

# A run on generated data records this as its registry data_hash (scripts/fit.py --synthetic does); such a run is a
# "fake" run: it is left out of every table unless asked for.
SYNTHETIC_DATA_HASH = "synthetic"

_TOP_LEVEL_KEYS = ("methods", "n_pairs", "pairs_seed", "permutations_B", "beta")
_METHOD_KEYS = ("ident_accuracy", "ident_ci", "perm_p", "null_max", "alignment_gain", "nonidentifiable_pairs",
                "per_pair_uncertainty", "per_pair_flags")
_MODEL_ONLY_KEYS = ("per_pair_uncertainty", "per_pair_flags")


class MetricsError(ValueError):
    """A run's ``metrics.json`` is missing a key the table needs, or a value is malformed."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_method(run_id: str, name: str, stats: Any) -> None:
    def fail(key: str, problem: str) -> None:
        raise MetricsError(f"run {run_id}: methods.{name}.{key}: {problem}")

    if not isinstance(stats, dict):
        raise MetricsError(f"run {run_id}: methods.{name}: must be an object, got {type(stats).__name__}")
    for key in _METHOD_KEYS:
        if key not in stats:
            fail(key, "missing")

    acc = stats["ident_accuracy"]
    if not _is_number(acc) or not 0.0 <= acc <= 1.0:
        fail("ident_accuracy", f"must be a number in [0, 1], got {acc!r}")
    ci = stats["ident_ci"]
    if not isinstance(ci, (list, tuple)) or len(ci) != 2 or not all(_is_number(v) for v in ci):
        fail("ident_ci", f"must be two numbers, got {ci!r}")
    p = stats["perm_p"]
    if not _is_number(p) or not 0.0 < p <= 1.0:
        fail("perm_p", f"must be a number in (0, 1], got {p!r}")
    for key in ("null_max", "alignment_gain"):
        if not _is_number(stats[key]):
            fail(key, f"must be a number, got {stats[key]!r}")
    count = stats["nonidentifiable_pairs"]
    if not _is_int(count) or count < 0:
        fail("nonidentifiable_pairs", f"must be a non-negative integer, got {count!r}")

    for key in _MODEL_ONLY_KEYS:
        value = stats[key]
        if name not in MODEL_METHODS:
            if value is not None:
                fail(key, f"a baseline cannot produce this and must report null (never 0 or empty), got {value!r}")
        elif value is None:
            fail(key, "a model row must report it, got null")
    if name in MODEL_METHODS:
        if not _is_number(stats["per_pair_uncertainty"]):
            fail("per_pair_uncertainty", f"must be a number, got {stats['per_pair_uncertainty']!r}")
        flags = stats["per_pair_flags"]
        if not isinstance(flags, list) or not flags or not all(isinstance(f, bool) for f in flags):
            fail("per_pair_flags", "must be a non-empty list of booleans")


def validate_metrics(metrics: dict[str, Any], run_id: str) -> None:
    """Raise :class:`MetricsError` naming ``run_id`` and the missing or malformed key.

    Requires ``methods`` (non-empty dict), the declared subsample (``n_pairs``, ``pairs_seed``, ``permutations_B``),
    ``beta`` (a number, or null for a baseline-only run), and for each method the keys of the table. A baseline
    reports ``null`` for ``per_pair_uncertainty`` and ``per_pair_flags``; the two model rows report values.
    """
    if not isinstance(metrics, dict):
        raise MetricsError(f"run {run_id}: metrics.json must be a JSON object, got {type(metrics).__name__}")
    for key in _TOP_LEVEL_KEYS:
        if key not in metrics:
            raise MetricsError(f"run {run_id}: {key}: missing")
    for key in ("n_pairs", "permutations_B"):
        if not _is_int(metrics[key]) or metrics[key] < 1:
            raise MetricsError(f"run {run_id}: {key}: the declared subsample must be a positive integer, got {metrics[key]!r}")
    if not _is_int(metrics["pairs_seed"]):
        raise MetricsError(f"run {run_id}: pairs_seed: the declared seed must be an integer, got {metrics['pairs_seed']!r}")
    beta = metrics["beta"]
    if beta is not None and not _is_number(beta):
        raise MetricsError(f"run {run_id}: beta: must be a number or null, got {beta!r}")
    methods = metrics["methods"]
    if not isinstance(methods, dict) or not methods:
        raise MetricsError(f"run {run_id}: methods: must be a non-empty object, got {methods!r}")
    for name, stats in methods.items():
        _validate_method(run_id, name, stats)


def load_run_metrics(runs_dir: Path, run_id: str) -> dict[str, Any]:
    """Read and validate ``<runs_dir>/<run_id>/metrics.json``; the only file of a run this package opens."""
    path = Path(runs_dir) / run_id / "metrics.json"
    try:
        metrics = json.loads(path.read_text())
    except FileNotFoundError:
        raise MetricsError(f"run {run_id}: metrics.json not found at {path}") from None
    except json.JSONDecodeError as err:
        raise MetricsError(f"run {run_id}: metrics.json is not valid JSON ({err})") from None
    validate_metrics(metrics, run_id)
    return metrics


def latest_successful_run(records: list[dict[str, Any]], experiment: str, include_fake: bool = False) -> dict[str, Any] | None:
    """The registry record with the latest ``start_utc`` among the successful runs of ``experiment`` (ties: last row)."""
    candidates = [r for r in records if r["experiment"] == experiment and r["status"] == "ok"
                  and (include_fake or r["data_hash"] != SYNTHETIC_DATA_HASH)]
    return sorted(candidates, key=lambda r: r["start_utc"])[-1] if candidates else None


def _select_runs(records: list[dict[str, Any]], index_path: Path, experiments: Iterable[str] | None,
                 run_ids: Iterable[str] | None, include_fake: bool) -> list[dict[str, Any]]:
    if run_ids is not None:
        known = {r["run_id"]: r for r in records}
        chosen = []
        for run_id in run_ids:
            if run_id not in known:
                raise ValueError(f"run_id {run_id!r} is not in {index_path}")
            if known[run_id]["status"] != "ok":
                raise ValueError(f"run {run_id} has status {known[run_id]['status']!r}, not 'ok'")
            chosen.append(known[run_id])
    else:
        names = list(TABLE_EXPERIMENTS if experiments is None else experiments)
        known_names = set(TABLE_EXPERIMENTS) | {r["experiment"] for r in records}
        for name in names:
            if name not in known_names:
                raise ValueError(f"unknown experiment {name!r}; the registry has {sorted(known_names)}")
        latest = (latest_successful_run(records, name, include_fake) for name in names)
        chosen = [r for r in latest if r is not None]
    return sorted(chosen, key=lambda r: r["start_utc"])  # oldest first: a later run overwrites an earlier one


def _missing_row(label: str) -> dict[str, Any]:
    keys = ("experiment", "run_id", "ident_accuracy", "ident_ci", "perm_p", "null_max", "alignment_gain",
            "nonidentifiable_pairs", "per_pair_uncertainty", "per_pair_flags", "n_subjects", "n_pairs", "pairs_seed",
            "permutations_B", "beta")
    return {"method": label, "missing": True, **dict.fromkeys(keys)}


def collect_rows(index_path: Path, experiments: Iterable[str] | None = None, run_ids: Iterable[str] | None = None,
                 *, include_fake: bool = False) -> list[dict[str, Any]]:
    """One dict per table row, in ``TABLE_ROWS`` order; a row with no run is ``missing`` and empty.

    Reads ``index_path`` (``runs/index.csv``) through W0's ``read_registry``. Unless explicit ``run_ids`` are given
    it takes the latest successful run of each of ``experiments`` (default: the six ``TABLE_EXPERIMENTS``), leaving
    out fake runs unless ``include_fake``. Each run's ``metrics.json`` is validated and fills every row whose
    method key it holds; among several runs the latest wins.
    """
    index_path = Path(index_path)
    records = read_registry(index_path).to_dict("records")
    rows = {label: _missing_row(label) for label in TABLE_ROWS}
    for entry in _select_runs(records, index_path, experiments, run_ids, include_fake):
        metrics = load_run_metrics(index_path.parent, entry["run_id"])
        for label, key in ROW_METHODS.items():
            if key in metrics["methods"]:
                rows[label] = {
                    "method": label, "missing": False, "experiment": entry["experiment"], "run_id": entry["run_id"],
                    **{k: metrics["methods"][key][k] for k in _METHOD_KEYS},
                    "n_subjects": metrics.get("n_subjects"), **{k: metrics[k] for k in ("n_pairs", "pairs_seed", "permutations_B", "beta")},
                }
    return [rows[label] for label in TABLE_ROWS]


def _cells(label: str, row: dict[str, Any] | None) -> list[str]:
    if row is None or row.get("missing"):
        return [f"{label} (missing)"] + [EMPTY_CELL] * 4
    acc, ci = row.get("ident_accuracy"), row.get("ident_ci")
    accuracy = EMPTY_CELL if acc is None else f"{acc:.2f}" + ("" if ci is None else f" [{ci[0]:.2f}, {ci[1]:.2f}]")
    p = row.get("perm_p")
    cells = [label, accuracy, EMPTY_CELL if p is None else f"{p:.4f}", EMPTY_CELL, EMPTY_CELL]
    if label in MODEL_ROWS:
        uncertainty, flagged = row.get("per_pair_uncertainty"), row.get("nonidentifiable_pairs")
        cells[3] = EMPTY_CELL if uncertainty is None else f"{uncertainty:.3f}"
        cells[4] = EMPTY_CELL if flagged is None else str(flagged)
    return cells


def render_table(rows: list[dict[str, Any]], fmt: str = "markdown") -> str:
    """Six fixed rows and five fixed columns in ``TABLE_ROWS`` / ``TABLE_COLUMNS`` order, whatever ``rows`` holds.

    A row with no run (absent from ``rows``, or ``missing``) shows ``EMPTY_CELL`` in every cell and a ``(missing)``
    marker on its label. ``fmt`` is ``"markdown"`` or ``"csv"``.
    """
    if fmt not in ("markdown", "csv"):
        raise ValueError(f"fmt must be 'markdown' or 'csv', got {fmt!r}")
    by_label = {row.get("method"): row for row in rows}  # a later row for the same method wins
    table = [_cells(label, by_label.get(label)) for label in TABLE_ROWS]
    if fmt == "csv":
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(TABLE_COLUMNS)
        writer.writerows(table)
        return out.getvalue()
    for cells in table:
        if cells[0] in MODEL_ROWS:
            cells[0] = f"**{cells[0]}**"
    lines = ["| " + " | ".join(COLUMN_TITLES) + " |", "|" + "|".join(["---"] * len(COLUMN_TITLES)) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in table]
    return "\n".join(lines) + "\n"


def render_table_meta(folds: str, seed: int | None, n_pairs: int | None, beta: float | None) -> str:
    """The declared subsample that must accompany any table: pair count, seed, draw procedure, fold scheme, ``beta``."""
    if n_pairs is None:
        subsample = "not declared (no run in this table)"
    else:
        subsample = f"{n_pairs} ordered subject pairs, seed {'not declared' if seed is None else seed}"
    return "\n".join([
        f"- Declared pair subsample: {subsample}",
        "- Draw procedure: ordered pairs (a, b) of distinct subjects drawn with replacement by "
        "numpy.random.default_rng(seed) (trajot.eval.folds.sample_pairs)",
        f"- Fold scheme: {folds}",
        f"- beta: {'not reported (no model run in this table)' if beta is None else beta}",
    ]) + "\n"
