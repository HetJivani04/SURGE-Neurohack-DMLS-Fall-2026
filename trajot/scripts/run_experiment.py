#!/usr/bin/env python
"""Single entry point for every experiment.

    python scripts/run_experiment.py --config configs/experiments/10_ours_full.yaml

Every run writes ``runs/<run_id>/{manifest.json, metrics.json, log.txt, artifacts/}`` and one
row of ``runs/index.csv``.

``metrics.json`` schema (published here by W0; W2, W3, and W4 write and read it). Required keys::

    experiment, run_id, n_subjects, n_pairs, pairs_seed, permutations_B,
    methods: {<name>: {ident_accuracy: float, ident_ci: [float, float], perm_p: float,
                       alignment_gain: float, nonidentifiable_pairs: int,
                       per_pair_uncertainty: float | null, per_pair_flags: list[bool] | null}},
    beta: float | null, notes: str

``null`` is how a baseline reports a column it cannot fill: never ``0`` and never an omitted
key, because W4's table must show those cells as deliberately empty.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Only numpy-free modules are imported here: setup_threads() has to run before the first
# BLAS call, and BLAS reads its thread count at import time.
from trajot.config import Config, load_config
from trajot.runlog.parallel import setup_threads

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

EXPERIMENT_NAMES = (
    "00_noalign", "01_brainsync", "02_fugw", "03_conn_srm", "10_ours_full", "11_ours_ablated",
)


def _stub_runner(cfg: Config, run_dir: Path) -> dict[str, Any]:
    """W0 stub: an empty-but-schema-valid metrics payload."""
    return {
        "experiment": cfg.experiment,
        "run_id": run_dir.name,
        "n_subjects": 0,
        "n_pairs": cfg.get("eval.pairs.n"),
        "pairs_seed": cfg.get("eval.pairs.seed"),
        "permutations_B": cfg.get("eval.permutations.B"),
        "methods": {},
        "beta": None,
        "notes": "",
    }


# W1-W4 replace entries as they land; the signature never changes.
EXPERIMENT_RUNNERS: dict[str, Callable[[Config, Path], dict[str, Any]]] = {
    name: _stub_runner for name in EXPERIMENT_NAMES
}


def _override(text: str) -> tuple[str, Any]:
    key, sep, value = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    return key, yaml.safe_load(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], type=_override, metavar="KEY=VALUE")
    parser.add_argument("--seed", type=int, help="sets run.seed")
    parser.add_argument("--n-jobs", type=int, help="sets run.n_jobs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true", help="sets run.debug")
    return parser


def dispatch(cfg: Config, run_dir: Path) -> dict[str, Any]:
    """Look up ``EXPERIMENT_RUNNERS[cfg.experiment]`` and run it."""
    return EXPERIMENT_RUNNERS[cfg.experiment](cfg, run_dir)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = dict(args.override)
    if args.seed is not None:
        overrides["run.seed"] = args.seed
    if args.n_jobs is not None:
        overrides["run.n_jobs"] = args.n_jobs
    if args.debug:
        overrides["run.debug"] = True

    cfg = load_config(args.config, overrides)
    n_jobs = cfg.n_jobs
    threads = setup_threads("serial" if n_jobs == 1 else "outer", n_jobs)

    # After setup_threads: these import pandas / numpy.
    from trajot.runlog.logging import RunLogger
    from trajot.runlog.manifest import git_commit, hash_data_root, make_run_id, write_manifest
    from trajot.runlog.registry import append_run

    start_utc = datetime.now(timezone.utc)
    run_id = make_run_id(cfg.experiment, cfg.hash, start_utc)
    run_dir = RUNS_DIR / run_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    seed, operator = cfg.get("run.seed"), getpass.getuser()
    data_hash = hash_data_root(cfg.data_root / "derivatives" / "trajot")
    manifest = dict(seed=seed, operator=operator, n_jobs=n_jobs, threads=threads,
                    data_hash=data_hash, start_utc=start_utc)
    write_manifest(run_dir, cfg, **manifest, end_utc=None)

    logger = RunLogger(run_dir, run_id)
    try:
        logger.log(f"config hash {cfg.hash[:8]}, seed {seed}, n_jobs {n_jobs}")
        metrics = dispatch(cfg, run_dir)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        logger.log("finished")
    finally:
        logger.close()

    end_utc = datetime.now(timezone.utc)
    write_manifest(run_dir, cfg, **manifest, end_utc=end_utc)
    append_run(RUNS_DIR / "index.csv", {
        "run_id": run_id, "experiment": cfg.experiment, "config_hash": cfg.hash,
        "git_commit": git_commit(), "data_hash": data_hash, "seed": seed, "operator": operator,
        "n_jobs": n_jobs, "start_utc": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_utc": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "status": "ok",
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
