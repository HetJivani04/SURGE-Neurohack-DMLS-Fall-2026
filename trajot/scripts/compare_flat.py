#!/usr/bin/env python
"""Build comparison table from runs/index.csv + runs/*/metrics.json.

    python scripts/compare.py
    python scripts/compare.py --runs /path/to/runs --out /path/to/results/tables/comparison.csv

Reads every ``metrics.json`` under ``runs/*/``, flattens ``methods`` to one row per
(experiment, method), writes CSV to ``trajot/results/tables/comparison.csv`` and prints
a markdown table. Empty/null cells stay empty in the CSV (not 0).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = PROJECT_ROOT / "runs"
DEFAULT_OUT = PROJECT_ROOT / "results" / "tables" / "comparison.csv"

COLUMNS = [
    "experiment",
    "run_id",
    "method",
    "ident_accuracy",
    "ident_ci_lo",
    "ident_ci_hi",
    "perm_p",
    "null_max",
    "alignment_gain",
    "nonidentifiable_pairs",
    "per_pair_uncertainty",
    "notes",
]


def _cell(value: Any) -> str:
    """CSV cell: empty string for None/NaN; never coerce unfilled metrics to 0."""
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return f"{value:.10g}"
    return str(value)


def _num(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_rows(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return rows
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        experiment = payload.get("experiment") or metrics_path.parent.name
        run_id = payload.get("run_id") or metrics_path.parent.name
        notes = payload.get("notes") or ""
        methods = payload.get("methods") or {}
        if not isinstance(methods, dict):
            continue
        for method, stats in methods.items():
            if not isinstance(stats, dict):
                continue
            ci = stats.get("ident_ci") or []
            ci_lo = ci[0] if len(ci) > 0 else None
            ci_hi = ci[1] if len(ci) > 1 else None
            rows.append({
                "experiment": experiment,
                "run_id": run_id,
                "method": method,
                "ident_accuracy": _num(stats.get("ident_accuracy")),
                "ident_ci_lo": _num(ci_lo),
                "ident_ci_hi": _num(ci_hi),
                "perm_p": _num(stats.get("perm_p")),
                "null_max": _num(stats.get("null_max")),
                "alignment_gain": _num(stats.get("alignment_gain")),
                "nonidentifiable_pairs": _num(stats.get("nonidentifiable_pairs")),
                "per_pair_uncertainty": _num(stats.get("per_pair_uncertainty")),
                "notes": notes,
            })
    return rows


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _cell(row.get(c)) for c in COLUMNS})
    return out_path


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ["experiment", "method", "ident_accuracy", "perm_p", "null_max",
               "alignment_gain", "nonidentifiable_pairs", "per_pair_uncertainty"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for h in headers:
            val = row.get(h)
            if val is None or val == "":
                cells.append("")
            elif isinstance(val, float):
                cells.append(f"{val:.4g}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build comparison table from experiment metrics.")
    p.add_argument("--runs", type=str, default=str(DEFAULT_RUNS), help="runs directory")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="output CSV path")
    p.add_argument("--quiet", action="store_true", help="do not print the markdown table")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs_dir = Path(args.runs)
    out_path = Path(args.out)

    rows = collect_rows(runs_dir)
    write_csv(rows, out_path)
    if not args.quiet:
        if rows:
            print(markdown_table(rows))
        else:
            print(f"No metrics.json found under {runs_dir}", file=sys.stderr)
        print(f"\nWrote {out_path} ({len(rows)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
