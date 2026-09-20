#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from trajot.baselines import FakeBaseline, get_baseline
from trajot.config import Config, load_config
from trajot.eval import (
    alignment_gain,
    identify_subjects,
    nonidentifiable_pairs,
    permutation_p,
    random_permutation_control,
    validate_metrics_payload,
)
from trajot.io.contract import load_connectomes, read_manifest, two_run_subjects
from trajot.runlog.parallel import parallel_map, pick_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate baselines on two-run identifiability")
    p.add_argument("--config", required=True, help="Experiment config path")
    p.add_argument("--methods", default="noalign", help="Comma-separated method names")
    p.add_argument("--n-jobs", type=int, help="Override run.n_jobs")
    p.add_argument("--pairs", type=int, help="Override eval.pairs.n")
    p.add_argument("--seed", type=int, help="Override run.seed")
    p.add_argument("--synthetic", action="store_true", help="Use generated synthetic connectomes")
    p.add_argument("--output", type=str, help="Optional path to write metrics json")
    return p


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if args.n_jobs is not None:
        out["run.n_jobs"] = args.n_jobs
    if args.pairs is not None:
        out["eval.pairs.n"] = args.pairs
    if args.seed is not None:
        out["run.seed"] = args.seed
    return out


def _synthetic_connectomes(*, S: int, R: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    run1 = np.empty((S, R, R), dtype=np.float64)
    run2 = np.empty((S, R, R), dtype=np.float64)

    for s in range(S):
        A = rng.normal(size=(R, R))
        C = A @ A.T
        C = C / np.maximum(np.std(C), 1e-12)
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 0.0)

        N = rng.normal(scale=0.10, size=(R, R))
        N = 0.5 * (N + N.T)
        np.fill_diagonal(N, 0.0)

        run1[s] = C
        run2[s] = C + N

    subjects = [f"{i + 1:03d}" for i in range(S)]
    return run1, run2, subjects


def _load_data(cfg: Config, synthetic: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if synthetic:
        return _synthetic_connectomes(S=24, R=32, seed=int(cfg.get("run.seed", 0)))

    manifest = read_manifest(cfg.data_root)
    subjects = two_run_subjects(manifest, strict=False)
    if not subjects:
        raise ValueError("No two-run subjects available in manifest")

    run1, sub1 = load_connectomes(cfg.data_root, subjects=subjects, run="1")
    run2, sub2 = load_connectomes(cfg.data_root, subjects=subjects, run="2")
    if sub1 != sub2:
        raise ValueError("Run-1 and run-2 subject ordering differs")
    return run1, run2, sub1


def _method_instance(name: str):
    if name.lower() == "fake":
        return FakeBaseline()
    return get_baseline(name)


def _evaluate_one(name: str, run1: np.ndarray, run2: np.ndarray, cfg: Config, noalign_acc: float) -> tuple[str, dict[str, Any]]:
    method = _method_instance(name)
    method.fit(run1, cfg)

    aligned1 = np.stack([method.transform(c) for c in run1], axis=0)
    aligned2 = np.stack([method.transform(c) for c in run2], axis=0)

    ident = identify_subjects(aligned1, aligned2)
    perm = permutation_p(aligned1, aligned2, B=int(cfg.get("eval.permutations.B", 1000)), seed=int(cfg.get("eval.pairs.seed", 0)))
    ctrl = random_permutation_control(aligned1, aligned2, seed=int(cfg.get("run.seed", 0)))

    ci_radius = 1.96 * np.sqrt(max(ident.accuracy * (1.0 - ident.accuracy), 1e-12) / aligned1.shape[0])
    ci = [float(max(0.0, ident.accuracy - ci_radius)), float(min(1.0, ident.accuracy + ci_radius))]

    stats = {
        "ident_accuracy": float(ident.accuracy),
        "ident_ci": ci,
        "perm_p": float(perm.p_value),
        "alignment_gain": float(alignment_gain(ident.accuracy, noalign_acc)),
        "nonidentifiable_pairs": int(nonidentifiable_pairs(ident.correct_mask)),
        "per_pair_uncertainty": float(ctrl.per_pair_uncertainty),
        "per_pair_flags": [bool(v) for v in ident.correct_mask.tolist()],
    }
    return name.lower(), stats


def _evaluate_all(methods: list[str], run1: np.ndarray, run2: np.ndarray, cfg: Config) -> dict[str, dict[str, Any]]:
    baseline = _method_instance("noalign")
    baseline.fit(run1, cfg)
    id0 = identify_subjects(run1, run2)
    noalign_acc = float(id0.accuracy)

    def _job(name: str) -> tuple[str, dict[str, Any]]:
        return _evaluate_one(name, run1, run2, cfg, noalign_acc)

    pairs = parallel_map(_job, methods, n_jobs=cfg.n_jobs, mode="outer")
    return dict(pairs)


def build_metrics_payload(
    *,
    cfg: Config,
    run_id: str,
    n_subjects: int,
    methods: dict[str, dict[str, Any]],
    synthetic: bool,
) -> dict[str, Any]:
    payload = {
        "experiment": cfg.experiment,
        "run_id": run_id,
        "n_subjects": int(n_subjects),
        "n_pairs": int(cfg.get("eval.pairs.n")),
        "pairs_seed": int(cfg.get("eval.pairs.seed")),
        "permutations_B": int(cfg.get("eval.permutations.B")),
        "methods": methods,
        "beta": None,
        "notes": "synthetic" if synthetic else "",
    }
    validate_metrics_payload(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=_overrides(args))

    # CPU-first policy for float64-heavy OT/GW-style workloads.
    _ = pick_device(prefer_mps=False)

    run1, run2, subjects = _load_data(cfg, synthetic=bool(args.synthetic))
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise ValueError("No methods requested")

    method_stats = _evaluate_all(methods, run1, run2, cfg)
    payload = build_metrics_payload(
        cfg=cfg,
        run_id=f"{cfg.experiment}__eval",
        n_subjects=len(subjects),
        methods=method_stats,
        synthetic=bool(args.synthetic),
    )

    rendered = json.dumps(payload, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
