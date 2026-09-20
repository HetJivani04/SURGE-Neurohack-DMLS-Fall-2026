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
import os
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
    "full": "ours_full",
    "ours": "ours_full",
}
CANONICAL_OURS_KEYS = frozenset({"ours_full", "ours_ablated"})


def _canonical_method_key(name: str) -> str:
    key = name.lower().strip()
    return EXPERIMENT_ALIASES.get(key, key)


def _method_keys() -> set[str]:
    return set(METHOD_KEYS)


def _top_keys() -> set[str]:
    return set(TOP_KEYS)


def _resolve_method(name: str):
    key = _canonical_method_key(name)
    return key, get_baseline(key)


def _timeseries_kwargs(timeseries: Any, *, run: str) -> dict[str, Any]:
    if timeseries is None:
        return {}
    if isinstance(timeseries, dict):
        ts = None
        for key in (run, f"run{run}", "run1", "run2"):
            if key in timeseries and timeseries[key] is not None:
                candidate = timeseries[key]
                # Never boolean-test array-likes: `or` on ndarrays raises ValueError.
                ts = candidate
                break
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


def _feat_corr(flat_a: np.ndarray, flat_r: np.ndarray) -> float | None:
    """Pearson corr of flattened aligned vs raw features."""
    a = np.asarray(flat_a, dtype=np.float64).ravel()
    r = np.asarray(flat_r, dtype=np.float64).ravel()
    if a.size != r.size or a.size < 2:
        return None
    sa, sr = float(np.std(a)), float(np.std(r))
    if sa < 1e-15 or sr < 1e-15:
        return 1.0 if bool(np.allclose(a, r)) else 0.0
    return float(np.corrcoef(a, r)[0, 1])


def _aligned_changed(
    aligned1: np.ndarray,
    aligned2: np.ndarray,
    raw1: np.ndarray,
    raw2: np.ndarray,
) -> bool:
    """True when aligned outputs differ from the raw runs (identity fallback is False)."""
    for aligned, raw in ((aligned1, raw1), (aligned2, raw2)):
        a = np.asarray(aligned, dtype=np.float64)
        r = np.asarray(raw, dtype=np.float64)
        if a.shape != r.shape:
            return True
        if a.size and not np.allclose(a, r, rtol=0.0, atol=0.0, equal_nan=True):
            return True
    return False


def _transform_diagnostics(
    aligned1: np.ndarray,
    aligned2: np.ndarray,
    raw1: np.ndarray,
    raw2: np.ndarray,
) -> dict[str, Any]:
    """How much the transform actually changed the inputs.

    ``max_abs_diff``: max |aligned - raw| over both runs.
    ``feat_corr``: Pearson corr of concatenated flattened features (aligned vs raw).
    """
    diffs: list[float] = []
    for aligned, raw in ((aligned1, raw1), (aligned2, raw2)):
        a = np.asarray(aligned, dtype=np.float64)
        r = np.asarray(raw, dtype=np.float64)
        if a.shape == r.shape and a.size:
            diffs.append(float(np.max(np.abs(a - r))))
    max_abs_diff = max(diffs) if diffs else None
    try:
        flat_a = np.concatenate(
            [
                np.asarray(aligned1, dtype=np.float64).ravel(),
                np.asarray(aligned2, dtype=np.float64).ravel(),
            ]
        )
        flat_r = np.concatenate(
            [
                np.asarray(raw1, dtype=np.float64).ravel(),
                np.asarray(raw2, dtype=np.float64).ravel(),
            ]
        )
        feat_corr = _feat_corr(flat_a, flat_r)
    except Exception:
        feat_corr = None
    return {"max_abs_diff": max_abs_diff, "feat_corr": feat_corr}


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

    Keys are ``METHOD_KEYS`` plus evaluator diagnostics
    (``status``, ``transforms_applied``, ``transform_diagnostics``,
    and ``fit_error`` when fit/transform raised). On fit/transform
    failure the method is marked ``status="failed"`` with
    null score columns — never scored as a successful noalign run.
    Successful transforms report ``status="ok"``,
    ``transforms_applied`` True only when aligned outputs differ from
    the raw runs, plus ``transform_diagnostics`` (max_abs_diff / feat_corr).
    """
    if method is None:
        method_key, method = _resolve_method(name)
    else:
        method_key = _canonical_method_key(name)

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
        # Keep payload schema-valid; never score identity fallback as a successful run.
        fit_error = f"{type(exc).__name__}: {exc}"
        meta = {"fit_error": fit_error, "fallback": "identity"}
        aligned1 = np.asarray(run1, dtype=np.float64)
        aligned2 = np.asarray(run2, dtype=np.float64)
        transform_diagnostics = _transform_diagnostics(aligned1, aligned2, run1, run2)
        return {
            "ident_accuracy": None,
            "ident_ci": None,
            "perm_p": None,
            "null_max": None,
            "alignment_gain": None,
            "nonidentifiable_pairs": None,
            "per_pair_uncertainty": None,
            "per_pair_flags": None,
            "status": "failed",
            "transforms_applied": False,
            "transform_diagnostics": transform_diagnostics,
            "fit_error": fit_error,
            "_meta": meta,
        }

    transform_diagnostics = _transform_diagnostics(aligned1, aligned2, run1, run2)
    transforms_applied = _aligned_changed(aligned1, aligned2, run1, run2)
    status = "ok"

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

    # Model-only columns: baselines must report JSON null (never lists or zeros).
    is_model_method = method_key in CANONICAL_OURS_KEYS
    per_pair_uncertainty: float | None = None
    if is_model_method:
        tau = getattr(method, "tau_phi", None)
        if tau is not None:
            try:
                if isinstance(tau, (list, tuple)):
                    vals = [float(np.mean(np.asarray(t, dtype=np.float64))) for t in tau if t is not None]
                    per_pair_uncertainty = float(np.mean(vals)) if vals else None
                else:
                    arr = np.asarray(tau, dtype=np.float64)
                    per_pair_uncertainty = float(np.mean(arr)) if arr.size else None
            except Exception:
                per_pair_uncertainty = None
        if per_pair_uncertainty is None:
            artifacts = ours_artifacts or meta.get("ours_artifacts") or meta.get("template_path")
            if artifacts:
                art_dir = Path(artifacts)
                if art_dir.is_file():
                    art_dir = art_dir.parent
                if art_dir.name != "artifacts" and (art_dir / "artifacts").is_dir():
                    art_dir = art_dir / "artifacts"
                tau_path = art_dir / "tau_phi.npz"
                if tau_path.is_file():
                    try:
                        with np.load(tau_path) as z:
                            keys = [k for k in z.files if k.startswith("sub-")] or list(z.files)
                            vals = [float(np.mean(np.asarray(z[k], dtype=np.float64))) for k in keys]
                            per_pair_uncertainty = float(np.mean(vals)) if vals else None
                    except Exception:
                        per_pair_uncertainty = None
        if per_pair_uncertainty is None and isinstance(meta, dict) and meta.get("tau_phi_mean") is not None:
            per_pair_uncertainty = float(meta["tau_phi_mean"])
        # Last resort for model rows: load tau_phi.npz from the declared artifacts directory
        # (run_experiment points ours_artifacts at runs/<id> when present).
        if per_pair_uncertainty is None:
            art = ours_artifacts or (meta or {}).get("ours_artifacts") or (meta or {}).get("template_path")
            if art:
                art_dir = Path(str(art))
                if art_dir.name != "artifacts" and (art_dir / "artifacts").is_dir():
                    art_dir = art_dir / "artifacts"
                tau_path = art_dir / "tau_phi.npz"
                if tau_path.is_file():
                    try:
                        with np.load(tau_path) as z:
                            keys = [k for k in z.files if str(k).startswith("sub-")] or list(z.files)
                            vals = [float(np.mean(np.asarray(z[k], dtype=np.float64))) for k in keys]
                            if vals:
                                per_pair_uncertainty = float(np.mean(vals))
                                meta = dict(meta or {})
                                meta["tau_phi_source"] = str(tau_path)
                    except Exception:
                        pass

    per_pair_flags: list[bool] | None = None
    if is_model_method:
        # Locked Track B statistic: gain inside the method's own permutation null.
        # When tau_phi exists, ALSO flag pairs whose subjects have high posterior width,
        # so the model column can differ from a pure gain-null of a weak transform.
        flags = list(nonident_flags) if nonident_flags else [bool(v) for v in correct.tolist()]
        if getattr(method, "tau_phi", None) is not None and pairs_idx:
            try:
                tau = method.tau_phi
                if isinstance(tau, (list, tuple)) and len(tau) >= len(subjects):
                    sub_tau = np.array([float(np.mean(np.asarray(t, dtype=np.float64))) for t in tau[:len(subjects)]])
                    thresh = float(np.quantile(sub_tau, 0.75)) if sub_tau.size >= 4 else float(np.median(sub_tau) + 1e-12)
                    for k, (i, j) in enumerate(pairs_idx):
                        if 0 <= i < sub_tau.size and 0 <= j < sub_tau.size:
                            if sub_tau[i] > thresh or sub_tau[j] > thresh:
                                flags[k] = True
                meta = dict(meta or {})
                meta["tau_phi_pair_rule"] = "gain_null OR subject_mean_tau>q75"
            except Exception:
                pass
        per_pair_flags = flags

    stats = {
        "ident_accuracy": float(acc),
        "ident_ci": [float(ci[0]), float(ci[1])],
        "perm_p": float(perm_p_val),
        "null_max": float(null_max_val),
        "alignment_gain": float(alignment_gain_val),
        "nonidentifiable_pairs": int(n_nonident),
        "per_pair_uncertainty": None if per_pair_uncertainty is None else float(per_pair_uncertainty),
        "per_pair_flags": None if per_pair_flags is None else [bool(v) for v in per_pair_flags],
        "status": status,
        "transforms_applied": bool(transforms_applied),
        "transform_diagnostics": transform_diagnostics,
    }
    if meta:
        stats["_meta"] = meta
    return stats


def _strip_meta(stats: dict[str, Any]) -> dict[str, Any]:
    """Keep METHOD_KEYS plus evaluator diagnostics; drop ``_meta`` and other ad-hoc keys."""
    from trajot.eval.metrics import ALLOWED_METHOD_KEYS

    allowed = _method_keys() | set(ALLOWED_METHOD_KEYS)
    return {k: stats[k] for k in allowed if k in stats}


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


def _parcellate_timeseries(ts_vt: np.ndarray, labels: np.ndarray, n_regions: int) -> np.ndarray:
    """Vertex ``(V,T)`` → region ``(R,T)`` means, matching preprocess ``parcellate``."""
    from trajot.geometry.connectivity import parcellate

    X = np.asarray(ts_vt, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"timeseries must be (V,T), found {X.shape}")
    labels = np.asarray(labels)
    valid = X.std(axis=1) > 0
    present = np.unique(labels[valid])
    lookup = np.full(n_regions, -1)
    lookup[present] = np.arange(present.size)
    region_ts = np.zeros((n_regions, X.shape[1]), dtype=np.float32)
    if present.size:
        region_ts[present] = parcellate(X[valid], lookup[labels[valid]], present.size)
    return region_ts


def _load_timeseries(root: Path, subjects: list[str]) -> dict[str, Any] | None:
    """Load two-run timeseries for baselines.

    Returns region-resolution series when template labels are available (BrainSync needs
    ``V == R``). Subject runs may differ in ``T`` (130/132/133 on ds000243); stacks are
    therefore truncated to the cohort minimum length so a single ``(S,R,T)`` array is
    well-defined. Falls back to vertex series only if labels cannot be resolved.
    """
    from trajot.io.contract import read_subject_run, subject_run_path

    labels = None
    geom = root / "derivatives" / "trajot" / "template_geometry.npz"
    if geom.is_file():
        try:
            with np.load(geom) as z:
                if "region_labels" in z.files:
                    labels = np.asarray(z["region_labels"])
        except Exception:
            labels = None

    ts1, ts2 = [], []
    for sid in subjects:
        try:
            d1 = read_subject_run(subject_run_path(root, sid, "1"))
            d2 = read_subject_run(subject_run_path(root, sid, "2"))
        except Exception:
            return None
        a1 = np.asarray(d1["timeseries"], dtype=np.float32)
        a2 = np.asarray(d2["timeseries"], dtype=np.float32)
        conn_r = int(np.asarray(d1["connectivity"]).shape[0])
        if labels is not None and a1.shape[0] == labels.shape[0]:
            a1 = _parcellate_timeseries(a1, labels, conn_r)
            a2 = _parcellate_timeseries(a2, labels, conn_r)
        ts1.append(a1)
        ts2.append(a2)
    if not ts1:
        return None
    try:
        min_t = min(min(a.shape[1] for a in ts1), min(a.shape[1] for a in ts2))
        if min_t < 2:
            return None
        r1 = np.stack([a[:, :min_t] for a in ts1], axis=0)
        r2 = np.stack([a[:, :min_t] for a in ts2], axis=0)
        return {"run1": r1, "run2": r2, "n_time_truncated_to": int(min_t)}
    except Exception:
        return None


def _synthetic_sandbox_root(cfg: Config) -> Path:
    """Isolated root for synthetic contract data.

    Synthetic runs must never write into the real ``data_root`` derivatives tree: that
    overwrites ``manifest.parquet`` and contaminates real subject npz.
    """
    override = os.environ.get("TRAJOT_SYNTHETIC_ROOT")
    if override:
        return Path(override)
    return Path(cfg.data_root) / "derivatives" / "trajot_synthetic"


def _synthetic_via_contract(cfg: Config, *, seed: int) -> tuple[np.ndarray, np.ndarray, list[str], Any] | None:
    """Write contract-exact synthetic npz in an isolated sandbox and load through the real IO path."""
    try:
        from trajot.inference.synthetic import make_synthetic_npz
    except ImportError:
        return None
    root = _synthetic_sandbox_root(cfg)
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
        key = _canonical_method_key(name)
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
        out[key] = _strip_meta(stats)
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
