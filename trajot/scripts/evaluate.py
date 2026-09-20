#!/usr/bin/env python
"""Evaluate alignment methods on two-run identifiability.

    python scripts/evaluate.py --config configs/experiments/00_noalign.yaml --methods noalign
    python scripts/evaluate.py --config configs/experiments/00_noalign.yaml --synthetic --methods noalign

Metrics payload schema (D7 / eval.metrics METHOD_KEYS)::

    ident_accuracy, ident_ci, perm_p, null_max,
    alignment_gain, nonidentifiable_pairs,
    per_pair_uncertainty, per_pair_flags

``null_max`` and ``per_pair_uncertainty`` may be JSON ``null`` when a method cannot fill them.
Never omit the keys and never write ``0`` for an unfilled column.

``evaluate_method`` is the shared helper used by ``run_experiment.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from trajot.baselines import get_baseline
from trajot.config import Config, load_config
from trajot.eval.alignment_gain import alignment_gain, pair_gain
from trajot.eval.folds import make_folds
from trajot.eval.identifiability import nonidentifiable_count
from trajot.eval.identification import accuracy_ci, identification_accuracy
from trajot.eval.metrics import METHOD_KEYS, TOP_KEYS
from trajot.eval.metrics import validate_metrics_payload
from trajot.eval.metrics import write_metrics as _write_metrics
from trajot.eval.permutation import null_max as perm_null_max
from trajot.eval.permutation import permutation_p
from trajot.io.contract import load_connectomes, read_manifest, two_run_subjects
from trajot.runlog.parallel import pick_device

PLAN_METHOD_KEYS = set(METHOD_KEYS)
assert "null_max" in PLAN_METHOD_KEYS, "eval.metrics.METHOD_KEYS must include null_max"

EXPERIMENT_ALIASES = {
    "00_noalign": "noalign",
    "01_brainsync": "brainsync",
    "02_fugw": "fugw",
    "03_conn_srm": "conn_srm",
    "10_ours_full": "ours_full",
    "11_ours_ablated": "ours_ablated",
    "ablated": "ours_ablated",
    "ours": "ours_full",
}


def _method_keys() -> set[str]:
    return set(METHOD_KEYS)


def _top_keys() -> set[str]:
    return set(TOP_KEYS)


def _resolve_method(name: str):
    key = EXPERIMENT_ALIASES.get(name.lower().strip(), name.lower().strip())
    return key, get_baseline(key)


def _timeseries_kwargs(timeseries: Any, *, run: str) -> dict[str, Any]:
    if timeseries is None:
        return {}
    if isinstance(timeseries, dict):
        ts = timeseries.get(run) or timeseries.get(f"run{run}") or timeseries.get("run1")
        if ts is None:
            return {}
        return {"timeseries_run1" if run == "1" else "timeseries_run2": ts}
    arr = np.asarray(timeseries)
    if arr.ndim == 4:
        # (2, S, V, T) or (S, 2, V, T)
        if arr.shape[0] == 2:
            ts = arr[0] if run == "1" else arr[1]
        elif arr.shape[1] == 2:
            ts = arr[:, 0] if run == "1" else arr[:, 1]
        else:
            return {}
        return {"timeseries_run1" if run == "1" else "timeseries_run2": ts}
    return {"timeseries": arr}


def _declared_pairs(subjects: list[str], cfg: Config) -> tuple[list[tuple[int, int]], Any]:
    """Pair subsample via make_folds (without replacement), clamped to the pool size."""
    n = len(subjects)
    if n < 2:
        return [(i, i) for i in range(n)], None
    n_pairs = int(cfg.get("eval.pairs.n") or 500)
    seed = int(cfg.get("eval.pairs.seed") or 0)
    pool = n * (n - 1)
    n_use = max(1, min(n_pairs, pool))
    folds = make_folds(subjects, scheme="pairs_without_replacement", n_pairs=n_use, seed=seed)
    lookup = {s: i for i, s in enumerate(subjects)}
    # make_folds sorts subjects; map back to the caller's ordering
    ordered = sorted(str(s) for s in subjects)
    lookup_sorted = {s: i for i, s in enumerate(ordered)}
    # Prefer the caller's subject list order when ids match exactly
    pairs: list[tuple[int, int]] = []
    for a, b in folds.pairs:
        if a in lookup and b in lookup:
            pairs.append((lookup[a], lookup[b]))
        elif a in lookup_sorted and b in lookup_sorted:
            pairs.append((lookup_sorted[a], lookup_sorted[b]))
    if not pairs:
        pairs = [(i, i) for i in range(n)]
    return pairs, folds


def _gain_null(
    aligned1: np.ndarray,
    aligned2: np.ndarray,
    raw1: np.ndarray,
    raw2: np.ndarray,
    pairs_idx: list[tuple[int, int]],
    *,
    B: int,
    seed: int,
) -> np.ndarray:
    """Pooled method-own permutation null of per-pair gains (subject-shuffle of run2)."""
    S = aligned1.shape[0]
    rng = np.random.default_rng(seed)
    chunks = []
    for _ in range(max(1, B)):
        perm = rng.permutation(S)
        chunks.append(pair_gain(aligned1, aligned2[perm], raw1, raw2[perm], pairs_idx))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros(0, dtype=np.float64)


def _fit_transform(
    method_key: str,
    method: Any,
    run1: np.ndarray,
    run2: np.ndarray,
    cfg: Config,
    subjects: list[str],
    *,
    timeseries: Any = None,
    ours_artifacts: Path | None = None,
    data_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    meta: dict[str, Any] = {}
    extra: dict[str, Any] = {"subjects": list(subjects)}
    if data_root is not None:
        extra["data_root"] = str(data_root)
    elif cfg.get("data.root"):
        extra["data_root"] = str(cfg.data_root)
    if ours_artifacts is not None:
        extra["run_dir"] = str(Path(ours_artifacts))
        meta["ours_artifacts"] = str(Path(ours_artifacts))
    if cfg.get("run.debug"):
        extra["epochs"] = int(cfg.get("model.train.epochs") or 2)
        # Keep synthetic/debug ours fits tiny
        extra["epochs"] = min(extra["epochs"], 3)

    extra.update(_timeseries_kwargs(timeseries, run="1"))
    extra.update(_timeseries_kwargs(timeseries, run="2"))
    if isinstance(timeseries, dict):
        if "run1" in timeseries:
            extra["timeseries_run1"] = timeseries["run1"]
        if "run2" in timeseries:
            extra["timeseries_run2"] = timeseries["run2"]

    try:
        method.fit(run1, cfg, extra=extra)
    except TypeError:
        method.fit(run1, cfg)

    ts_kwargs = {
        k: extra[k]
        for k in ("timeseries_run1", "timeseries", "timeseries_run2")
        if extra.get(k) is not None
    }

    if hasattr(method, "transform_all"):
        try:
            aligned1 = np.asarray(method.transform_all(run1, **ts_kwargs), dtype=np.float64)
            ts2 = dict(ts_kwargs)
            if "timeseries_run2" in extra:
                ts2["timeseries_run1"] = extra["timeseries_run2"]
                ts2.pop("timeseries_run2", None)
            elif "timeseries_run1" in ts_kwargs and isinstance(timeseries, dict) and "run2" in timeseries:
                ts2["timeseries_run1"] = timeseries["run2"]
            aligned2 = np.asarray(method.transform_all(run2, **ts2), dtype=np.float64)
        except Exception as exc:
            meta["transform_all_error"] = str(exc)
            aligned1 = np.stack([method.transform(c) for c in run1], axis=0)
            aligned2 = np.stack([method.transform(c) for c in run2], axis=0)
    else:
        aligned1 = np.stack([method.transform(c) for c in run1], axis=0)
        aligned2 = np.stack([method.transform(c) for c in run2], axis=0)

    method_meta = getattr(method, "meta", None)
    if isinstance(method_meta, dict):
        meta.update(method_meta)
    if hasattr(method, "device"):
        meta.setdefault("device", str(getattr(method, "device")))
    return aligned1, aligned2, meta


def evaluate_method(
    name: str,
    method: Any,
    run1: np.ndarray,
    run2: np.ndarray,
    cfg: Config,
    subjects: list[str],
    folds: Any = None,
    timeseries: Any = None,
    ours_artifacts: Path | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Transform both runs and score D7 metrics via trajot.eval.

    Keys are exactly ``METHOD_KEYS``. Unfilled model-only columns are JSON ``null``.
    """
    if method is None:
        method_key, method = _resolve_method(name)
    else:
        method_key = name.lower().strip()
        method_key = EXPERIMENT_ALIASES.get(method_key, method_key)

    meta: dict[str, Any] = {}
    fit_error: str | None = None
    try:
        aligned1, aligned2, meta = _fit_transform(
            method_key,
            method,
            run1,
            run2,
            cfg,
            subjects,
            timeseries=timeseries,
            ours_artifacts=ours_artifacts,
            data_root=data_root,
        )
    except Exception as exc:
        # Keep payload schema-valid; record why the method fell back to identity.
        fit_error = f"{type(exc).__name__}: {exc}"
        meta = {"fit_error": fit_error, "fallback": "identity"}
        aligned1 = np.asarray(run1, dtype=np.float64)
        aligned2 = np.asarray(run2, dtype=np.float64)

    B = int(cfg.get("eval.permutations.B") or 1000)
    seed = int(cfg.get("run.seed") or 0)
    pairs_seed = int(cfg.get("eval.pairs.seed") or 0)
    n_boot = int(cfg.get("eval.ci.n_boot") or 10000)

    # Identification: Pearson on flattened features (D7 / eval.identification).
    ident = identification_accuracy(aligned1, aligned2, metric="pearson")
    acc = float(ident.accuracy)
    correct = np.asarray(ident.correct_mask, dtype=bool)
    lo, hi = accuracy_ci(correct, n_boot=min(n_boot, 2000), seed=seed)
    ci = [float(lo), float(hi)]

    # Permutation null + null_max (D7).
    perm = permutation_p(aligned1, aligned2, B=B, seed=pairs_seed)
    perm_p_val = float(perm.p_value)
    null_max_val = float(perm_null_max(perm))

    # Declared pair subsample.
    if folds is not None and hasattr(folds, "pairs"):
        lookup = {str(s): i for i, s in enumerate(subjects)}
        pairs_idx = [(lookup[a], lookup[b]) for a, b in folds.pairs if a in lookup and b in lookup]
        if not pairs_idx:
            pairs_idx, folds = _declared_pairs(subjects, cfg)
    else:
        pairs_idx, folds = _declared_pairs(subjects, cfg)

    if pairs_idx:
        gains = pair_gain(aligned1, aligned2, run1, run2, pairs_idx)
        alignment_gain_val = float(alignment_gain(aligned1, aligned2, run1, run2, pairs_idx))
        null_dist = _gain_null(
            aligned1, aligned2, run1, run2, pairs_idx, B=min(B, 200), seed=seed
        )
        n_nonident, flags_arr = nonidentifiable_count(gains, null_dist, alpha=0.05)
        nonident_flags = [bool(v) for v in np.asarray(flags_arr).tolist()]
    else:
        alignment_gain_val = 0.0
        n_nonident = 0
        nonident_flags = []

    # per_pair_uncertainty: model-only (tau_phi); baselines report null.
    per_pair_uncertainty: float | None = None
    tau = getattr(method, "tau_phi", None)
    if tau is not None:
        try:
            arr = np.asarray(tau, dtype=np.float64)
            per_pair_uncertainty = float(np.mean(arr)) if arr.size else None
        except Exception:
            per_pair_uncertainty = None
    if per_pair_uncertainty is None and method_key.startswith("ours"):
        artifacts = ours_artifacts or meta.get("ours_artifacts") or meta.get("template_path")
        if artifacts:
            art_dir = Path(artifacts)
            if art_dir.is_file():
                art_dir = art_dir.parent
            tau_path = art_dir / "tau_phi.npz"
            if tau_path.is_file():
                try:
                    with np.load(tau_path) as z:
                        key = "tau_phi" if "tau_phi" in z.files else z.files[0]
                        per_pair_uncertainty = float(np.mean(np.asarray(z[key], dtype=np.float64)))
                except Exception:
                    per_pair_uncertainty = None

    if nonident_flags:
        per_pair_flags = nonident_flags
    else:
        per_pair_flags = [bool(v) for v in correct.tolist()]

    stats = {
        "ident_accuracy": float(acc),
        "ident_ci": [float(ci[0]), float(ci[1])],
        "perm_p": float(perm_p_val),
        "null_max": float(null_max_val),
        "alignment_gain": float(alignment_gain_val),
        "nonidentifiable_pairs": int(n_nonident),
        "per_pair_uncertainty": None if per_pair_uncertainty is None else float(per_pair_uncertainty),
        "per_pair_flags": [bool(v) for v in per_pair_flags],
    }
    if meta:
        stats["_meta"] = meta
    return stats


def _strip_meta(stats: dict[str, Any]) -> dict[str, Any]:
    keys = _method_keys()
    return {k: stats[k] for k in keys if k in stats}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate baselines on two-run identifiability")
    p.add_argument("--config", required=True, help="Experiment config path")
    p.add_argument("--methods", default="noalign", help="Comma-separated method names")
    p.add_argument("--n-jobs", type=int, help="Override run.n_jobs")
    p.add_argument("--pairs", type=int, help="Override eval.pairs.n")
    p.add_argument("--seed", type=int, help="Override run.seed")
    p.add_argument("--synthetic", action="store_true", help="Use generated synthetic connectomes")
    p.add_argument("--ours-artifacts", type=str, default=None, help="Path to runs/<id>/artifacts for ours methods")
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


def _load_timeseries(root: Path, subjects: list[str]) -> dict[str, np.ndarray] | None:
    from trajot.io.contract import read_subject_run, subject_run_path

    ts1, ts2 = [], []
    for sid in subjects:
        try:
            d1 = read_subject_run(subject_run_path(root, sid, "1"))
            d2 = read_subject_run(subject_run_path(root, sid, "2"))
        except Exception:
            return None
        ts1.append(np.asarray(d1["timeseries"], dtype=np.float32))
        ts2.append(np.asarray(d2["timeseries"], dtype=np.float32))
    if not ts1:
        return None
    try:
        return {"run1": np.stack(ts1, axis=0), "run2": np.stack(ts2, axis=0)}
    except Exception:
        return None


def _synthetic_via_contract(cfg: Config, *, seed: int) -> tuple[np.ndarray, np.ndarray, list[str], Any] | None:
    """Write contract-exact synthetic npz under cfg.data_root and load through the real IO path."""
    try:
        from trajot.inference.synthetic import make_synthetic_npz
    except ImportError:
        return None
    root = Path(cfg.data_root)
    try:
        make_synthetic_npz(
            root,
            n_subjects=8,
            V=40,
            R=20,
            T=24,
            d=8,
            seed=seed,
            planted_permutation=False,
        )
        manifest = read_manifest(root)
        subjects = two_run_subjects(manifest, strict=False)
        if not subjects:
            return None
        run1, sub1 = load_connectomes(root, subjects=subjects, run="1")
        run2, sub2 = load_connectomes(root, subjects=subjects, run="2")
        if sub1 != sub2:
            return None
        timeseries = _load_timeseries(root, sub1)
        return run1, run2, sub1, timeseries
    except Exception:
        return None


def _load_data(cfg: Config, synthetic: bool) -> tuple[np.ndarray, np.ndarray, list[str], Any]:
    if synthetic:
        loaded = _synthetic_via_contract(cfg, seed=int(cfg.get("run.seed", 0)))
        if loaded is not None:
            return loaded
        run1, run2, subjects = _synthetic_connectomes(S=24, R=32, seed=int(cfg.get("run.seed", 0)))
        return run1, run2, subjects, None

    manifest = read_manifest(cfg.data_root)
    subjects = two_run_subjects(manifest, strict=False)
    if not subjects:
        raise ValueError("No two-run subjects available in manifest")
    run1, sub1 = load_connectomes(cfg.data_root, subjects=subjects, run="1")
    run2, sub2 = load_connectomes(cfg.data_root, subjects=subjects, run="2")
    if sub1 != sub2:
        raise ValueError("Run-1 and run-2 subject ordering differs")
    timeseries = _load_timeseries(Path(cfg.data_root), sub1)
    return run1, run2, sub1, timeseries


def _evaluate_all(
    methods: list[str],
    run1: np.ndarray,
    run2: np.ndarray,
    cfg: Config,
    subjects: list[str],
    timeseries: Any = None,
    ours_artifacts: Path | None = None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in methods:
        stats = evaluate_method(
            name,
            None,
            run1,
            run2,
            cfg,
            subjects,
            timeseries=timeseries,
            ours_artifacts=ours_artifacts,
        )
        out[name.lower()] = _strip_meta(stats)
    return out


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


def validate_payload(payload: dict[str, Any]) -> None:
    validate_metrics_payload(payload)


def write_metrics_payload(path: Path, payload: dict[str, Any]) -> Path:
    """Write metrics via trajot.eval.metrics.write_metrics (C's METHOD_KEYS include null_max)."""
    return _write_metrics(Path(path), payload)


# Name expected by run_experiment / tests.
write_metrics = write_metrics_payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=_overrides(args))
    _ = pick_device(prefer_mps=False)

    run1, run2, subjects, timeseries = _load_data(cfg, synthetic=bool(args.synthetic))
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    if not methods:
        raise ValueError("No methods requested")

    ours_artifacts = Path(args.ours_artifacts) if args.ours_artifacts else None
    method_stats = _evaluate_all(
        methods,
        run1,
        run2,
        cfg,
        subjects,
        timeseries=timeseries,
        ours_artifacts=ours_artifacts,
    )
    payload = build_metrics_payload(
        cfg=cfg,
        run_id=f"{cfg.experiment}__eval",
        n_subjects=len(subjects),
        methods=method_stats,
        synthetic=bool(args.synthetic),
    )

    rendered = json.dumps(payload, indent=2)
    if args.output:
        write_metrics_payload(Path(args.output), payload)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
