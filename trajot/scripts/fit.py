#!/usr/bin/env python
"""W2 entry point: fit the hierarchical model with amortized variational inference.

    python scripts/fit.py --config configs/experiments/10_ours_full.yaml --epochs 100
    python scripts/fit.py --synthetic --epochs 20

On real data ``beta`` is calibrated from the 83 two-run subjects (``artifacts/beta.json``); the model is trained on
run 1 with run 2 held out. ``--synthetic`` generates a small contract-exact dataset with a planted vertex
permutation inside the run directory and uses the fixed ``model.beta.synthetic``: ``beta`` is never calibrated on
synthetic data. Each invocation is recorded under ``runs/<run_id>/`` (manifest, log, artifacts, ``metrics.json``)
and in ``runs/index.csv``.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from trajot.config import load_config
from trajot.runlog.parallel import setup_threads

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "10_ours_full.yaml"
SYNTHETIC = dict(n_subjects=12, V=200, R=100, T=132, d=32)  # the --synthetic dataset; K becomes V when K > V


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="W2: fit the hierarchical population-of-couplings model")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="experiment config (default: %(default)s)")
    parser.add_argument("--synthetic", action="store_true", help="fit on generated data with a planted permutation")
    parser.add_argument("--epochs", type=int, default=None, help="overrides model.train.epochs")
    parser.add_argument("--subjects", type=str, default=None, help="comma-separated subject ids")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    overrides: dict = {}
    if args.epochs is not None:
        overrides["model.train.epochs"] = args.epochs
    if args.synthetic:
        overrides["model.K"] = min(int(load_config(args.config).get("model.K")), SYNTHETIC["V"])
    cfg = load_config(args.config, overrides)
    n_jobs = cfg.n_jobs
    threads = setup_threads("serial" if n_jobs == 1 else "inner", n_jobs)  # before the first BLAS call

    # After setup_threads: these import numpy / torch / pandas.
    from trajot.inference.beta import calibrate_beta
    from trajot.inference.synthetic import make_synthetic_npz
    from trajot.inference.train import load_train_data, save_artifacts, train
    from trajot.runlog.logging import RunLogger
    from trajot.runlog.manifest import git_commit, hash_data_root, make_run_id, write_manifest
    from trajot.runlog.parallel import pick_device
    from trajot.runlog.registry import append_run

    start_utc = datetime.now(timezone.utc)
    run_id = make_run_id(cfg.experiment, cfg.hash, start_utc)
    run_dir = RUNS_DIR / run_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    root = run_dir / "synthetic_data" if args.synthetic else Path(cfg.data_root)
    seed, operator = cfg.get("run.seed"), getpass.getuser()
    data_hash = "synthetic" if args.synthetic else hash_data_root(root / "derivatives" / "trajot")
    manifest = dict(seed=seed, operator=operator, n_jobs=n_jobs, threads=threads, data_hash=data_hash, start_utc=start_utc)
    write_manifest(run_dir, cfg, **manifest, end_utc=None)

    logger = RunLogger(run_dir, run_id)
    try:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None
        if args.synthetic:
            make_synthetic_npz(root, seed=int(seed), **SYNTHETIC)
            beta_target, beta_note = None, f"beta fixed at model.beta.synthetic ({cfg.get('model.beta.synthetic')}): synthetic data"
        else:
            beta = calibrate_beta(root, out_dir=run_dir / "artifacts")  # the only place beta is computed: 83 two-run subjects
            beta_target = beta["beta"]
            beta_note = f"beta {beta_target:.6g} calibrated from {beta['n_subjects']} two-run subjects"
            logger.log(f"calibrated {beta_note}")

        data = load_train_data(root, cfg, subjects, beta_target)
        logger.log(f"{len(data.train)} subjects, K={cfg.get('model.K')}, run 1 trains and run 2 is held out")
        result = train(cfg, data, pick_device(prefer_mps=False) if args.synthetic else pick_device())
        save_artifacts(run_dir, result)

        metrics = {
            "experiment": cfg.experiment, "run_id": run_id, "n_subjects": len(result.subject_ids),
            "n_pairs": cfg.get("eval.pairs.n"), "pairs_seed": cfg.get("eval.pairs.seed"),
            "permutations_B": cfg.get("eval.permutations.B"), "methods": {}, "beta": float(result.beta_target),
            "notes": f"first-pass fit, entropy term dropped (tau_phi carries no uncertainty yet); {beta_note}; "
                     f"final total {result.loss_trace[-1]['total']:.6g}",
        }
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
