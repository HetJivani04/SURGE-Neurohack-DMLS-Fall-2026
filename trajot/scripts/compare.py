#!/usr/bin/env python
"""W4 entry point: compare runs into the results table of PLAN section 7.4.

    python scripts/compare.py --experiments all
    python scripts/compare.py --runs <run_id>,<run_id> --format csv --out reports/results_table.csv
    python scripts/compare.py --experiments all --sensitivity

Reads only ``runs/index.csv`` and the ``metrics.json`` of each selected run (``--sensitivity`` also reads the dataset
manifest for the long-run subset). It never opens anything else in a run directory and never starts a run, so the
results of different developers are compared without anyone re-running anything. The table always has six rows and
five columns; a row with no run shows ``(missing)`` and dashes. Runs on generated data (registry ``data_hash``
``synthetic``) are left out unless ``--include-fake``. Failures name the run and the key and exit 1.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "10_ours_full.yaml"  # its paths.yaml gives data_root, for --sensitivity
FOLD_SCHEME = ("two-run identification: each subject's run 1 is the query against the gallery of run 2 and vice versa, "
               "both folds holding the same sorted subject list (trajot.eval.folds.make_two_run_splits)")
METRICS = ["ident_accuracy", "perm_p", "alignment_gain", "nonidentifiable_pairs", "per_pair_uncertainty"]
_METRIC_COLUMNS = {"ident_accuracy": ["ident_accuracy", "ident_ci_low", "ident_ci_high"]}
_IDENTIFYING = ["method", "experiment", "run_id", "n_subjects"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare runs into the results table (reads runs/index.csv and metrics.json only)")
    parser.add_argument("--experiments", default="all",
                        help="comma-separated experiment names, or 'all' (the six of the table); the latest successful run of each is used")
    parser.add_argument("--runs", help="comma-separated run_ids; used instead of --experiments")
    parser.add_argument("--metric", choices=METRICS, help="with --sensitivity, show only this metric (the table keeps its five columns)")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument("--out", help="also write the output to this file")
    parser.add_argument("--include-fake", action="store_true", help="include runs on generated data (registry data_hash 'synthetic')")
    parser.add_argument("--sensitivity", action="store_true", help="append the scan-length sensitivity table (long-run subset)")
    return parser


def _split(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _declared(rows: list[dict], key: str):
    """The one value the selected runs declare for ``key``, None if none does; several values are an error."""
    values: dict = {}
    for row in rows:
        if not row["missing"] and row[key] is not None:
            values.setdefault(row[key], []).append(row["run_id"])
    if len(values) > 1:
        detail = "; ".join(f"{value} ({', '.join(sorted(set(ids)))})" for value, ids in values.items())
        raise ValueError(f"the selected runs declare different {key}: {detail}")
    return next(iter(values), None)


def _comment(block: str) -> str:
    return "".join(f"# {line}\n" for line in block.splitlines())


def _cell(column: str, value, empty: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return empty
    if column == "perm_p":
        return f"{value:.4f}"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _sensitivity(args: argparse.Namespace, index_path: Path, experiments: list[str] | None) -> str:
    from trajot.config import load_config
    from trajot.io.contract import read_manifest
    from trajot.report.sensitivity import SENSITIVITY_SUFFIX, long_run_subset, scan_length_metrics
    from trajot.report.table import EMPTY_CELL

    subset = long_run_subset(read_manifest(load_config(DEFAULT_CONFIG).data_root))
    frame = scan_length_metrics(index_path, subset, experiments, include_fake=args.include_fake)
    metric_columns = _METRIC_COLUMNS.get(args.metric, [args.metric]) if args.metric else [
        "ident_accuracy", "ident_ci_low", "ident_ci_high", "perm_p", "alignment_gain", "nonidentifiable_pairs", "per_pair_uncertainty"]
    heading = "Scan-length sensitivity (not a headline number)"
    if args.format == "csv":
        return _comment(heading) + frame[["analysis", *_IDENTIFYING, *metric_columns]].to_csv(index=False)
    lines = [f"### {heading}", "", f"{len(subset)} subjects in the long-run subset.", ""]
    if frame.empty:
        lines.append(f"There are no runs of the form <experiment>{SENSITIVITY_SUFFIX} in {index_path.name}.")
        return "\n".join(lines) + "\n"
    columns = [*_IDENTIFYING, *metric_columns]
    lines += ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    lines += ["| " + " | ".join(_cell(c, row[c], EMPTY_CELL) for c in columns) + " |" for _, row in frame.iterrows()]
    return "\n".join(lines) + "\n"


def _output(args: argparse.Namespace) -> str:
    from trajot.report.table import collect_rows, render_table, render_table_meta

    index_path = RUNS_DIR / "index.csv"
    run_ids = _split(args.runs) if args.runs else None
    experiments = None if run_ids is not None or args.experiments == "all" else _split(args.experiments)
    rows = collect_rows(index_path, experiments, run_ids, include_fake=args.include_fake)
    meta = render_table_meta(FOLD_SCHEME, _declared(rows, "pairs_seed"), _declared(rows, "n_pairs"), _declared(rows, "beta"))
    sources = "Source runs:\n" + "".join(f"- {row['method']}: {row['run_id'] or '(missing)'}\n" for row in rows)
    table = render_table(rows, args.format)
    text = f"{table}\n{meta}\n{sources}" if args.format == "markdown" else table + _comment(meta + sources)
    if args.sensitivity:
        text += "\n" + _sensitivity(args, index_path, experiments)
    return text


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        text = _output(args)
    except (ValueError, FileNotFoundError) as err:  # MetricsError is a ValueError
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(text, end="")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
