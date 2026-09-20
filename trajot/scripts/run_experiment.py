#!/usr/bin/env python
"""Single entry point for every experiment.

    python scripts/run_experiment.py --config configs/experiments/10_ours_full.yaml
    python scripts/run_experiment.py --config configs/experiments/00_noalign.yaml --synthetic

Every run writes ``runs/<run_id>/{manifest.json, metrics.json, log.txt, artifacts/}`` and one
row of ``runs/index.csv``.

``metrics.json`` schema (published by W0; W3/W4 write and read it). Required keys::

    experiment, run_id, n_subjects, n_pairs, pairs_seed, permutations_B,
    methods: {<name>: {ident_accuracy, ident_ci, perm_p, null_max,
                       alignment_gain, nonidentifiable_pairs,
                       per_pair_uncertainty, per_pair_flags}},
    beta: float | null, notes: str

``null`` is how a baseline reports a column it cannot fill: never ``0`` and never an omitted
key, because W4's table must show those cells as deliberately empty.

When ``scripts/evaluate.py`` is available (normal repo layout), each experiment dispatches to
real fit+eval via ``evaluate_method``. When only this file is copied (entrypoint smoke tests),
runners fall back to an empty-but-schema-valid stub payload.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.environ.get("TRAJOT_RUNS_DIR") or (PROJECT_ROOT / "runs"))

EXPERIMENT_NAMES = (
    "00_noalign", "01_brainsync", "02_fugw", "03_conn_srm", "10_ours_full", "11_ours_ablated",
)

EXPERIMENT_METHOD = {
    "00_noalign": "noalign",
    "01_brainsync": "brainsync",
    "02_fugw": "fugw",
    "03_conn_srm": "conn_srm",
    "10_ours_full": "ours_full",
    "11_ours_ablated": "ours_ablated",
}


def _evaluate_script_candidates() -> list[Path]:
    here = Path(__file__).resolve().parent
    candidates = [here / "evaluate.py"]
    env = os.environ.get("TRAJOT_EVALUATE_PY")
    if env:
        candidates.append(Path(env))
    return candidates


def _load_evaluate_module():
    """Load scripts/evaluate.py when present; None when only this file was copied."""
    for path in _evaluate_script_candidates():
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("trajot_scripts_evaluate", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            continue
        return module
    return None


def _resolve_synthetic(cfg: Config, explicit: bool = False) -> bool:
    if explicit or cfg.get("run.synthetic"):
        return True
    manifest = Path(cfg.data_root) / "derivatives" / "trajot" / "manifest.parquet"
    return not manifest.exists()


def _beta_from_config_or_artifacts(cfg: Config, run_dir: Path) -> float | None:
    candidates = [
        run_dir / "artifacts" / "beta.json",
        Path(cfg.data_root) / "derivatives" / "trajot" / "artifacts" / "beta.json",
        Path("artifacts") / "beta.json",
    ]
    raw = cfg.get("run.ours_artifacts")
    if raw:
        art = Path(str(raw))
        if art.name != "artifacts":
            art = art / "artifacts"
        candidates.insert(0, art / "beta.json")
    for artifacts in candidates:
        if artifacts.is_file():
            try:
                data = json.loads(artifacts.read_text())
                if "beta" in data:
                    return float(data["beta"])
            except Exception:
                pass
    # Prefer calibrated real-data beta over the synthetic constant when present in metrics path.
    val = cfg.get("model.beta.calibrated")
    if val is not None:
        return float(val)
    val = cfg.get("model.beta.synthetic")
    return None if val is None else float(val)


def _stub_runner(cfg: Config, run_dir: Path) -> dict[str, Any]:
    """Empty-but-schema-valid metrics payload when evaluate glue is unavailable."""
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


def _run_experiment_method(
    experiment: str,
    cfg: Config,
    run_dir: Path,
    *,
    synthetic: bool | None = None,
) -> dict[str, Any]:
    """Load data, fit/transform the experiment method, evaluate with D7 metrics."""
    ev = _load_evaluate_module()
    if ev is None or not hasattr(ev, "evaluate_method"):
        return _stub_runner(cfg, run_dir)

    method_name = EXPERIMENT_METHOD[experiment]
    synth = _resolve_synthetic(cfg) if synthetic is None else synthetic
    run1, run2, subjects, timeseries = ev._load_data(cfg, synthetic=synth)

    ours_artifacts = None
    raw = cfg.get("run.ours_artifacts")
    if raw:
        ours_artifacts = Path(str(raw))
    elif method_name.startswith("ours") and (run_dir / "artifacts").is_dir():
        ours_artifacts = run_dir / "artifacts"

    stats = ev.evaluate_method(
        method_name,
        None,
        run1,
        run2,
        cfg,
        subjects,
        timeseries=timeseries,
        ours_artifacts=ours_artifacts,
        data_root=cfg.data_root,
    )
    # Persist diagnostics at method top-level even if a stub strip drops them.
    meta = stats.get("_meta") or {}
    fit_error = stats.get("fit_error")
    if fit_error is None:
        fit_error = meta.get("fit_error")
    status = stats.get("status")
    if status is None:
        status = "identity_fallback" if fit_error else "ok"
    transforms_applied = stats.get("transforms_applied")
    if transforms_applied is None:
        transforms_applied = False if status == "identity_fallback" else True

    stripped = ev._strip_meta(stats)
    if "status" not in stripped:
        stripped["status"] = status
    if "fit_error" not in stripped and fit_error is not None:
        stripped["fit_error"] = fit_error
    if "transforms_applied" not in stripped:
        stripped["transforms_applied"] = bool(transforms_applied)
    if "transform_diagnostics" not in stripped and isinstance(stats.get("transform_diagnostics"), dict):
        stripped["transform_diagnostics"] = stats["transform_diagnostics"]
    if fit_error is not None and stripped.get("fit_error") is None:
        stripped["fit_error"] = fit_error
    if stripped.get("status") in (None, ""):
        stripped["status"] = status
    if stripped.get("transforms_applied") is None:
        stripped["transforms_applied"] = bool(transforms_applied)
    stats = stripped

    notes_parts = []
    if synth:
        notes_parts.append("synthetic")
    if stats.get("null_max") is None:
        notes_parts.append("null_max unfilled")
    if stats.get("per_pair_uncertainty") is None:
        notes_parts.append("per_pair_uncertainty unfilled")
    # Prefer the persisted diagnostics field; fall back to _meta if stripped earlier.
    fit_error = stats.get("fit_error") or fit_error
    if fit_error:
        notes_parts.append(str(fit_error))
    status = stats.get("status") or status
    if status == "identity_fallback" or meta.get("fallback"):
        notes_parts.append(f"fallback={meta.get('fallback') or 'identity'}")
        if status:
            notes_parts.append(f"status={status}")
    if stats.get("transforms_applied") is False:
        notes_parts.append("transforms_applied=False")

    payload = {
        "experiment": experiment,
        "run_id": run_dir.name,
        "n_subjects": int(len(subjects)),
        "n_pairs": int(cfg.get("eval.pairs.n")),
        "pairs_seed": int(cfg.get("eval.pairs.seed")),
        "permutations_B": int(cfg.get("eval.permutations.B")),
        "methods": {method_name: stats},
        "beta": _beta_from_config_or_artifacts(cfg, run_dir),
        "notes": "; ".join(notes_parts),
    }
    return payload


def _make_runner(experiment: str) -> Callable[[Config, Path], dict[str, Any]]:
    def _runner(cfg: Config, run_dir: Path) -> dict[str, Any]:
        return _run_experiment_method(experiment, cfg, run_dir)

    _runner.__name__ = f"run_{experiment}"
    _runner.__doc__ = f"Real fit+eval runner for {experiment} (stub if evaluate.py missing)."
    return _runner


# Each experiment maps to real fit+eval when evaluate.py is importable; else stub.
EXPERIMENT_RUNNERS: dict[str, Callable[[Config, Path], dict[str, Any]]] = {
    name: _make_runner(name) for name in EXPERIMENT_NAMES
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
    parser.add_argument("--synthetic", action="store_true", help="force synthetic data path")
    parser.add_argument("--dry-run", action="store_true",
                        help="still runs dispatch and writes the run record (W0 acceptance path)")
    parser.add_argument("--debug", action="store_true", help="sets run.debug")
    return parser


def dispatch(cfg: Config, run_dir: Path) -> dict[str, Any]:
    """Look up ``EXPERIMENT_RUNNERS[cfg.experiment]`` and run it."""
    runner = EXPERIMENT_RUNNERS.get(cfg.experiment, _stub_runner)
    return runner(cfg, run_dir)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = dict(args.override)
    if args.seed is not None:
        overrides["run.seed"] = args.seed
    if args.n_jobs is not None:
        overrides["run.n_jobs"] = args.n_jobs
    if args.debug:
        overrides["run.debug"] = True
    if args.synthetic:
        overrides["run.synthetic"] = True

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
    derivatives = cfg.data_root / "derivatives" / "trajot"
    data_hash = hash_data_root(derivatives, glob="sub-*_run-*.npz")
    manifest = dict(seed=seed, operator=operator, n_jobs=n_jobs, threads=threads,
                    data_hash=data_hash, start_utc=start_utc)
    write_manifest(run_dir, cfg, **manifest, end_utc=None)

    logger = RunLogger(run_dir, run_id)
    try:
        logger.log(f"config hash {cfg.hash[:8]}, seed {seed}, n_jobs {n_jobs}")
        metrics = dispatch(cfg, run_dir)
        # W0 contract: write the payload as-is so callers can inject schema variants.
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        logger.log("finished")
    finally:
        logger.close()

    end_utc = datetime.now(timezone.utc)
    write_manifest(run_dir, cfg, **manifest, end_utc=end_utc)

    primary = None
    methods = metrics.get("methods") or {}
    if methods:
        primary = next(iter(methods.values()))
    run_status = "ok"
    if isinstance(primary, dict) and primary.get("status") in {"failed", "identity_fallback"}:
        run_status = "failed"
    row = {
        "run_id": run_id, "experiment": cfg.experiment, "config_hash": cfg.hash,
        "git_commit": git_commit(), "data_hash": data_hash, "seed": seed, "operator": operator,
        "n_jobs": n_jobs, "start_utc": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_utc": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "status": run_status,
    }
    if isinstance(primary, dict):
        row["ident_accuracy"] = primary.get("ident_accuracy")
        row["perm_p"] = primary.get("perm_p")
        row["alignment_gain"] = primary.get("alignment_gain")
        row["nonidentifiable_pairs"] = primary.get("nonidentifiable_pairs")
    append_run(RUNS_DIR / "index.csv", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
