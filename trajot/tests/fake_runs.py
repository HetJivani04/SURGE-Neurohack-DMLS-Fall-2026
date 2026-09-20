"""Hand-written fake runs for the W4 tests: a registry (``runs/index.csv``) plus ``runs/<run_id>/metrics.json``.

A run is *fake* when its registry ``data_hash`` is ``"synthetic"``, the value ``scripts/fit.py --synthetic`` records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from trajot.runlog.registry import append_run

# experiment -> (method key inside metrics.json, is a model row); the order is the table's row order
EXPERIMENTS = {
    "00_noalign": ("noalign", False),
    "01_brainsync": ("brainsync", False),
    "02_fugw": ("fugw", False),
    "03_conn_srm": ("conn_srm", False),
    "11_ours_ablated": ("ablated", True),
    "10_ours_full": ("full", True),
}


def method_stats(model: bool = False, accuracy: float = 0.8, **override: Any) -> dict[str, Any]:
    """One method's entry of ``metrics.json``: baselines report ``null`` for the two model-only columns."""
    stats: dict[str, Any] = {
        "ident_accuracy": accuracy,
        "ident_ci": [max(0.0, accuracy - 0.05), min(1.0, accuracy + 0.05)],
        "perm_p": 0.0001,
        "null_max": 0.06,
        "alignment_gain": 0.1,
        "nonidentifiable_pairs": 12,
        "per_pair_uncertainty": 0.31 if model else None,
        "per_pair_flags": [False, True, False, False] if model else None,
    }
    stats.update(override)
    return stats


def metrics_payload(methods: dict[str, dict[str, Any]], experiment: str = "10_ours_full", run_id: str = "r",
                    n_subjects: int = 83, **override: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": experiment, "run_id": run_id, "n_subjects": n_subjects, "n_pairs": 500, "pairs_seed": 2026,
        "permutations_B": 10000, "methods": methods, "beta": 28.4, "notes": "hand-written",
    }
    payload.update(override)
    return payload


def write_run(runs_dir: Path, run_id: str, experiment: str, payload: dict[str, Any] | None, status: str = "ok",
              start_utc: str = "2026-09-20T10:00:00Z", fake: bool = False) -> Path:
    """Write ``runs/<run_id>/metrics.json`` (unless ``payload`` is None) and append the registry row."""
    runs_dir = Path(runs_dir)
    if payload is not None:
        (runs_dir / run_id).mkdir(parents=True, exist_ok=True)
        (runs_dir / run_id / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    append_run(runs_dir / "index.csv", {
        "run_id": run_id, "experiment": experiment, "config_hash": "0" * 64, "git_commit": "abc1234",
        "data_hash": "synthetic" if fake else "d" * 64, "seed": 0, "operator": "tester", "n_jobs": 1,
        "start_utc": start_utc, "end_utc": start_utc, "status": status})
    return runs_dir / "index.csv"


def write_full_registry(runs_dir: Path, fake: bool = False, start_utc: str = "2026-09-20T10:00:00Z",
                        **payload_override: Any) -> Path:
    """One successful run per experiment, each holding its own method, like six ``run_experiment`` runs."""
    for i, (experiment, (key, model)) in enumerate(EXPERIMENTS.items()):
        run_id = f"{experiment}__{i:08x}__20260920T100000Z"
        payload = metrics_payload({key: method_stats(model, 0.5 + 0.05 * i)}, experiment=experiment, run_id=run_id,
                                  **payload_override)
        write_run(runs_dir, run_id, experiment, payload, start_utc=start_utc, fake=fake)
    return Path(runs_dir) / "index.csv"


def manifest(runs: list[tuple[str, str, int]]) -> pd.DataFrame:
    """A contract-shaped manifest from (subject_id, run_id, n_volumes) triples."""
    return pd.DataFrame([{"subject_id": s, "run_id": r, "path": f"{s}_{r}.npz", "n_volumes": n, "tr": 2.5, "n_regions": 100,
                          "n_vertices": 5124, "qc_pass": True, "contract_version": "1.0.0"} for s, r, n in runs])


def dataset_like_manifest() -> tuple[pd.DataFrame, list[str]]:
    """The run-length composition of ds000243: 83 two-run subjects, and 37 one-run subjects, 26 of them long."""
    runs, expected, n = [], [], 0
    for _ in range(83):
        n += 1
        runs += [(f"{n:03d}", "1", 132), (f"{n:03d}", "2", 132)]
    for length, count in ((130, 11), (240, 10), (360, 15), (480, 5), (724, 6)):
        for _ in range(count):
            n += 1
            runs.append((f"{n:03d}", "1", length))
            if length >= 300:
                expected.append(f"{n:03d}")
    return manifest(runs), expected
