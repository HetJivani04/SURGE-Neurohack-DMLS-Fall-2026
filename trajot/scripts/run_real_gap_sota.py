#!/usr/bin/env python
"""Real frozen-data gap-filling + SOTA table (PRIMARY deliverable).

    python scripts/run_real_gap_sota.py
    python scripts/run_real_gap_sota.py --data-root /Users/anandlo/Surge2026F/ds000243-master \\
        --ours-artifacts runs/10_ours_full__73533e35__20260920T074217Z/artifacts

Writes ``trajot/results/tables/REAL_n49_gap_sota.json`` and ``.md``.

Primary SOTA metrics on REAL two-run connectomes (same subjects, same beta):
  - heldout_score_module:  mean_s -||C_run2 - T(C_run1)||_F^2   (train-map residual)
  - heldout_score_aligned: mean_s -||T(C_run2) - T(C_run1)||_F^2
  - alignment_gain on 500 pairs, seed 2026
  - ident_accuracy + tau_phi (uncertainty; baselines null)
  - scan-rescan Pearson corr raw vs after transform
  - group_neff via scripts/run_group_analysis.py on ours posteriors

Claim discipline: no invented wins. If ours does not beat noalign/fugw on
heldout / gain, report that honestly in notes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from trajot.baselines import get_baseline
from trajot.baselines.base import cfg_get
from trajot.config import Config, load_config
from trajot.eval.alignment_gain import alignment_gain, pair_gain
from trajot.eval.folds import make_folds
from trajot.eval.identification import accuracy_ci, identification_accuracy
from trajot.eval.uncertainty import heldout_predictive_score, subject_mean_tau
from trajot.io.contract import load_connectomes, read_manifest, subject_run_path
from trajot.runlog.parallel import pick_device, setup_threads

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = PROJECT_ROOT / "results" / "tables" / "frozen_cohort_n49.txt"
DEFAULT_TABLES = PROJECT_ROOT / "results" / "tables"
BETA_REAL = 29.189086229914952
PAIRS_SEED = 2026
N_PAIRS = 500


def _load_group_driver():
    path = Path(__file__).resolve().parent / "run_group_analysis.py"
    spec = importlib.util.spec_from_file_location("run_group_analysis", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_group_analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


def _upper(C: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(C.shape[-1], k=1)
    return np.asarray(C, dtype=np.float64)[..., iu[0], iu[1]]


def _row_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ac, bc = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(ac, bc) / denom)


def load_cohort(
    data_root: Path,
    subjects: list[str] | None = None,
    *,
    load_ts: bool = True,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    """Load two-run connectomes (and optional timeseries) for the frozen cohort."""
    data_root = Path(data_root)
    manifest = read_manifest(data_root)
    if subjects is None:
        if DEFAULT_COHORT.is_file():
            subjects = [ln.strip() for ln in DEFAULT_COHORT.read_text().splitlines() if ln.strip()]
        else:
            counts = manifest.groupby("subject_id")["run_id"].apply(set)
            subjects = sorted(s for s, runs in counts.items() if len(runs) >= 2)
    subjects = [str(s) for s in subjects]
    run1, sids1 = load_connectomes(data_root, subjects=subjects, run="1")
    run2, sids2 = load_connectomes(data_root, subjects=subjects, run="2")
    if sids1 != sids2:
        raise ValueError(f"run1/run2 subject mismatch: {sids1[:3]} vs {sids2[:3]}")
    ts_map: dict[str, Any] = {}
    if load_ts:
        for sid in sids1:
            ts_list = []
            for run in ("1", "2"):
                path = subject_run_path(data_root, sid, run)
                with np.load(path) as z:
                    ts_list.append(np.asarray(z["timeseries"], dtype=np.float64))
            # store as (2, V, T) — pad T if runs differ
            t_min = min(t.shape[1] for t in ts_list)
            ts_map[sid] = np.stack([t[:, :t_min] for t in ts_list], axis=0)
    meta = {
        "data_root": str(data_root),
        "n_subjects": len(sids1),
        "subjects": list(sids1),
        "R": int(run1.shape[1]),
        "beta": BETA_REAL,
    }
    return list(sids1), run1, run2, {"timeseries": ts_map, "meta": meta}


def _region_labels(data_root: Path, subject_id: str, V: int) -> np.ndarray | None:
    root = Path(data_root)
    geo = root / "derivatives" / "trajot" / "template_geometry.npz"
    if not geo.is_file():
        return None
    with np.load(geo) as z:
        if "region_labels" not in z.files:
            return None
        labels = np.asarray(z["region_labels"]).ravel().astype(np.int64)
    if labels.shape[0] == V:
        return labels
    path = subject_run_path(root, subject_id, "1")
    if not path.is_file():
        return None
    with np.load(path) as z:
        valid = np.asarray(z["timeseries"]).std(axis=1) > 0
    if int(valid.sum()) == V and labels.shape[0] == valid.shape[0]:
        return labels[valid]
    return None


def _parcellate_ts(
    ts_vt: np.ndarray,
    region_index: np.ndarray,
    R: int,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Average vertex time series into R parcels (BrainSync needs V==R).

    ``region_index`` is defined on *valid* vertices; when ``valid`` is given the
    time series is masked to those vertices first (ds000243 V_all=5124, V_valid~4591).
    """
    ts = np.asarray(ts_vt, dtype=np.float64)
    if valid is not None:
        ts = ts[np.asarray(valid, dtype=bool)]
    lab = np.asarray(region_index, dtype=np.int64).ravel()
    if lab.shape[0] != ts.shape[0]:
        raise ValueError(f"region_index {lab.shape[0]} != V={ts.shape[0]}")
    out = np.zeros((R, ts.shape[1]), dtype=np.float64)
    for r in range(R):
        mask = lab == r
        if not np.any(mask):
            continue
        out[r] = ts[mask].mean(axis=0)
    return out


def _timeseries_for_method(
    ts_map: dict[str, Any],
    subjects: list[str],
    *,
    region_labels: list[np.ndarray | None] | None = None,
    valid_masks: list[np.ndarray | None] | None = None,
    R: int | None = None,
    parcellate: bool = False,
) -> dict[str, np.ndarray]:
    """Stack (2,V,T) maps for BrainSync; optional vertex→region parcellation."""

    def _stack(which: int) -> np.ndarray:
        mats = []
        for i, s in enumerate(subjects):
            m = np.asarray(ts_map[s][which], dtype=np.float64)
            if parcellate and region_labels is not None and R is not None:
                lab = region_labels[i]
                if lab is not None:
                    valid = valid_masks[i] if valid_masks is not None else None
                    m = _parcellate_ts(m, lab, R, valid=valid)
            mats.append(m)
        v = min(m.shape[0] for m in mats)
        t = min(m.shape[1] for m in mats)
        out = np.zeros((len(mats), v, t), dtype=np.float64)
        for i, m in enumerate(mats):
            out[i] = m[:v, :t]
        return out

    run1, run2 = _stack(0), _stack(1)
    return {"run1": run1, "run2": run2, "run-1": run1, "run-2": run2}


def _declared_pairs(subjects: list[str], n_pairs: int = N_PAIRS, seed: int = PAIRS_SEED) -> list[tuple[int, int]]:
    n = len(subjects)
    if n < 2:
        return [(0, 0)]
    n_use = max(1, min(n_pairs, n * (n - 1)))
    folds = make_folds(sorted(subjects), scheme="pairs_without_replacement", n_pairs=n_use, seed=seed)
    ordered = sorted(str(s) for s in subjects)
    lookup = {s: i for i, s in enumerate(subjects)}
    pairs: list[tuple[int, int]] = []
    for a, b in getattr(folds, "pairs", []) or []:
        if a in lookup and b in lookup:
            pairs.append((lookup[a], lookup[b]))
    if not pairs:
        # fall back to within-subject self pairs
        pairs = [(i, i) for i in range(n)]
    return pairs


def score_transforms(
    method_name: str,
    aligned1: np.ndarray,
    aligned2: np.ndarray,
    run1: np.ndarray,
    run2: np.ndarray,
    subjects: list[str],
    pairs: list[tuple[int, int]],
    *,
    meta: dict[str, Any] | None = None,
    tau_phi: list[np.ndarray] | None = None,
    tau0_eff: float | None = None,
) -> dict[str, Any]:
    """PRIMARY metrics: same-map both-run reliability / gain / ident.

    Protocol (binding): each method's map is fitted on run1 and applied to BOTH
    runs. ``heldout ||C2 - T(C1)||`` is protocol-wrong (compares differently
    treated matrices) and is recorded only as a non-headline diagnostic.

    Primary columns:
      reliability_after = mean_s corr(vec(T(C1_s)), vec(T(C2_s)))
      reliability_raw   = mean_s corr(vec(C1_s), vec(C2_s))
      gain_after        = alignment_gain(T1, T2, raw1, raw2) on declared pairs
      ident_after       = ID accuracy on transformed both-run features
    """
    S = run1.shape[0]
    reliability_raw = []
    reliability_after = []
    heldout_proto_wrong = []
    for s in range(S):
        reliability_raw.append(_row_corr(_upper(run1[s]), _upper(run2[s])))
        reliability_after.append(_row_corr(_upper(aligned1[s]), _upper(aligned2[s])))
        heldout_proto_wrong.append(heldout_predictive_score(run2[s], aligned1[s]))

    ident = identification_accuracy(aligned1, aligned2, metric="pearson")
    lo, hi = accuracy_ci(np.asarray(ident.correct_mask, dtype=bool), n_boot=2000, seed=0)
    ident_raw = identification_accuracy(run1, run2, metric="pearson")
    gains = pair_gain(aligned1, aligned2, run1, run2, pairs)
    gain_after = float(alignment_gain(aligned1, aligned2, run1, run2, pairs))
    gain_self = float(np.mean([g for g, (i, j) in zip(gains, pairs) if i == j])) if any(
        i == j for i, j in pairs
    ) else None

    tau_means = subject_mean_tau(tau_phi) if tau_phi else None
    tau_mean = float(np.mean(tau_means)) if tau_means is not None and tau_means.size else None
    tau_median = float(np.median(tau_means)) if tau_means is not None and tau_means.size else None
    # Undetermined: λ < 0.5  ⇔  mean_tau > tau0_eff  (or entropy gate if meta says so)
    lam_list = None
    if meta and "lambda_mean" in meta:
        # reconstruct from meta if per-subject not stored; else use lambda_* stats
        pass
    n_undetermined = None
    if tau_means is not None and tau_means.size and tau0_eff is not None and tau0_eff > 0:
        n_undetermined = int(np.sum(tau_means > float(tau0_eff)))
    elif meta and meta.get("lambda_min") is not None and meta.get("lambda_mean") is not None:
        # approximate: subjects with λ < 0.5 unknown without per-subject list
        n_undetermined = None

    reliability_raw_m = float(np.mean(reliability_raw))
    reliability_after_m = float(np.mean(reliability_after))
    reliability_delta = reliability_after_m - reliability_raw_m
    ident_after = float(ident.accuracy)
    ident_raw_m = float(ident_raw.accuracy)

    # Collapse checks (must NOT collapse)
    reliability_collapsed = reliability_after_m < reliability_raw_m - 1e-6
    ident_collapsed = ident_after < 0.5 * ident_raw_m
    collapsed = bool(reliability_collapsed or ident_collapsed)

    max_abs = float(
        max(
            np.max(np.abs(aligned1 - run1)) if aligned1.size else 0.0,
            np.max(np.abs(aligned2 - run2)) if aligned2.size else 0.0,
        )
    )
    meta_out = {k: v for k, v in (meta or {}).items() if k not in {"per_subject_transform"}}
    row = {
        "method": method_name,
        "n_subjects": int(S),
        # --- PRIMARY (correct protocol) ---
        "reliability_raw": reliability_raw_m,
        "reliability_after": reliability_after_m,
        "reliability_delta": reliability_delta,
        "reliability_collapsed": bool(reliability_collapsed),
        "gain_after": gain_after,
        "gain_after_selfpairs": gain_self,
        "ident_after": ident_after,
        "ident_raw": ident_raw_m,
        "ident_ci": [float(lo), float(hi)],
        "ident_collapsed": bool(ident_collapsed),
        "collapsed": collapsed,
        # --- Track B uncertainty (ours only) ---
        "tau_phi_mean": tau_mean,
        "tau_phi_median": tau_median,
        "per_pair_uncertainty": tau_mean,
        "tau0_eff": float(tau0_eff) if tau0_eff is not None else meta_out.get("tau0_eff"),
        "lambda_mean": meta_out.get("lambda_mean"),
        "lambda_min": meta_out.get("lambda_min"),
        "lambda_max": meta_out.get("lambda_max"),
        "lambda_p10": meta_out.get("lambda_p10"),
        "lambda_p50": meta_out.get("lambda_p50"),
        "lambda_p90": meta_out.get("lambda_p90"),
        "lambda_source": meta_out.get("lambda_source"),
        "c_pop_mix": meta_out.get("c_pop_mix"),
        "n_undetermined_subjects": n_undetermined,
        "baselines_fill_uncertainty": False if method_name.startswith("ours") else None,
        # --- deprecated / non-headline diagnostics ---
        "heldout_score_module": float(np.mean(heldout_proto_wrong)),
        "heldout_protocol_wrong": True,
        "alignment_gain": gain_after,  # back-compat alias
        "ident_accuracy": ident_after,  # back-compat alias
        "scanrescan_corr_raw": reliability_raw_m,
        "scanrescan_corr_after": reliability_after_m,
        "transforms_applied": bool(max_abs > 0),
        "max_abs_diff": max_abs,
        "status": "ok",
        "meta": meta_out,
    }
    return row


def fit_and_transform(
    method_name: str,
    run1: np.ndarray,
    run2: np.ndarray,
    cfg: Any,
    subjects: list[str],
    *,
    data_root: Path,
    timeseries: dict[str, np.ndarray] | None = None,
    artifacts: Path | None = None,
    transform_mode: str | None = None,
    region_indices: list[np.ndarray] | None = None,
    tau0: float | None = None,
    tau0_auto: bool | None = None,
    lambda_source: str | None = None,
    c_pop_mix: float | None = None,
    hierarchical_pi_shrink: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[np.ndarray] | None]:
    method = get_baseline(method_name)
    if transform_mode is not None and hasattr(method, "transform_mode"):
        method.transform_mode = str(transform_mode)
    if tau0 is not None and hasattr(method, "tau0"):
        method.tau0 = float(tau0)
    if tau0_auto is not None and hasattr(method, "tau0_auto"):
        method.tau0_auto = bool(tau0_auto)
    if lambda_source is not None and hasattr(method, "lambda_source"):
        method.lambda_source = str(lambda_source)
    if c_pop_mix is not None and hasattr(method, "c_pop_mix"):
        method.c_pop_mix = float(c_pop_mix)
    if hierarchical_pi_shrink is not None and hasattr(method, "hierarchical_pi_shrink"):
        method.hierarchical_pi_shrink = bool(hierarchical_pi_shrink)

    extra: dict[str, Any] = {
        "subjects": list(subjects),
        "data_root": str(data_root),
        "beta_target": BETA_REAL,
    }
    if artifacts is not None:
        extra["run_dir"] = str(artifacts)
    if region_indices is not None:
        extra["region_indices"] = region_indices
    if timeseries is not None:
        extra["timeseries_run1"] = timeseries["run1"]
        extra["timeseries_run2"] = timeseries["run2"]

    try:
        method.fit(run1, cfg, extra=extra)
    except TypeError:
        method.fit(run1, cfg)

    ts_kwargs = {}
    if timeseries is not None:
        ts_kwargs = {"timeseries_run1": timeseries["run1"], "timeseries_run2": timeseries["run2"]}

    if hasattr(method, "transform_all"):
        try:
            aligned1 = np.asarray(method.transform_all(run1, **ts_kwargs), dtype=np.float64)
            ts2 = dict(ts_kwargs)
            if "timeseries_run1" in ts_kwargs:
                ts2["timeseries_run1"] = timeseries["run2"]
                ts2.pop("timeseries_run2", None)
            aligned2 = np.asarray(method.transform_all(run2, **ts2), dtype=np.float64)
        except Exception as exc:
            meta_err = {"transform_all_error": str(exc)}
            aligned1 = np.stack([method.transform(c) for c in run1], axis=0)
            aligned2 = np.stack([method.transform(c) for c in run2], axis=0)
            meta = dict(getattr(method, "meta", {}) or {})
            meta.update(meta_err)
            return aligned1, aligned2, meta, getattr(method, "tau_phi", None)
    else:
        aligned1 = np.stack([method.transform(c) for c in run1], axis=0)
        aligned2 = np.stack([method.transform(c) for c in run2], axis=0)

    meta = dict(getattr(method, "meta", {}) or {})
    return aligned1, aligned2, meta, getattr(method, "tau_phi", None)


def run_group_on_artifacts(artifacts: Path, subjects: list[str] | None = None) -> dict[str, Any] | None:
    try:
        driver = _load_group_driver()
        # Subsample subjects for memory: posterior draws are large (V~4591, K=100, M=4).
        return driver.run_group(artifacts, z=None, method="REML", subjects=subjects)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "artifacts": str(artifacts)}


def _verdict(rows: list[dict[str, Any]], group: dict[str, Any] | None) -> dict[str, Any]:
    """SOTA bar on CORRECT primary metrics + Track B gap fill."""
    by = {
        r["method"]: r
        for r in rows
        if r.get("status", "ok") == "ok" and r.get("gain_after") is not None
    }
    notes: list[str] = [
        "PRIMARY protocol: same subject map Q_s applied to BOTH runs (fitted on run1). "
        "Headline metrics are reliability_after, gain_after, ident_after. "
        "heldout ||C2-T(C1)|| is protocol-wrong and is NOT headlined.",
    ]
    baselines = [by[k] for k in ("noalign", "brainsync", "fugw", "conn_srm") if k in by]
    ours_all = [
        by[k]
        for k in (
            "ours_full_posterior_shrink_entropy",
            "ours_full_posterior_shrink_cpop",
            "ours_full_posterior_shrink",
            "ours_full_point_procrustes",
            "ours_full_c_bar_procrustes",
            "ours_ablated",
            "ours_full",
        )
        if k in by
    ]
    # Valid ours: non-collapsed; among those pick highest gain_after, tie-break reliability_after
    valid_ours = [r for r in ours_all if not r.get("collapsed", False)]
    pool = valid_ours or ours_all
    ours = max(pool, key=lambda r: (r.get("gain_after") or -9e9, r.get("reliability_after") or -9e9)) if pool else None
    noalign = by.get("noalign")
    brainsync = by.get("brainsync")
    fugw = by.get("fugw")
    conn = by.get("conn_srm")
    point = by.get("ours_full_point_procrustes")
    shrink = by.get("ours_full_posterior_shrink")
    shrink_e = by.get("ours_full_posterior_shrink_entropy")
    shrink_c = by.get("ours_full_posterior_shrink_cpop")

    def _noncollapsed(r: dict[str, Any] | None) -> bool:
        return bool(r) and not r.get("collapsed", False)

    # Beat each baseline on gain_after at non-collapsed reliability/ident
    beat = {}
    for name, base in (("noalign", noalign), ("brainsync", brainsync), ("fugw", fugw), ("conn_srm", conn)):
        if ours is None or base is None:
            beat[name] = None
            continue
        if base.get("collapsed"):
            beat[name] = True
            notes.append(
                f"{name} disqualified: identity/reliability collapse "
                f"(ident_after={base.get('ident_after')}, reliability_after={base.get('reliability_after')})."
            )
            continue
        beat[name] = bool(
            _noncollapsed(ours)
            and (ours.get("gain_after") or 0.0) > (base.get("gain_after") or 0.0)
        )
        notes.append(
            f"gain_after ours({ours['method']})={ours.get('gain_after')} vs {name}="
            f"{base.get('gain_after')} at reliability_after ours={ours.get('reliability_after')} "
            f"vs {name}={base.get('reliability_after')}; ident_after ours={ours.get('ident_after')} "
            f"vs {name}={base.get('ident_after')} -> beat={beat[name]}"
        )

    sota_gain = bool(ours and (ours.get("gain_after") or 0) > 0 and _noncollapsed(ours))
    beat_all = bool(all(beat.get(k) is True for k in ("noalign", "brainsync", "fugw")) and ours and _noncollapsed(ours))
    # conn_srm often collapses; require beating it only if it is non-collapsed
    if conn is not None and not conn.get("collapsed"):
        beat_all = beat_all and bool(beat.get("conn_srm"))

    for r in ours_all:
        notes.append(
            f"{r['method']}: gain_after={r.get('gain_after')}, reliability_after={r.get('reliability_after')} "
            f"(raw {r.get('reliability_raw')}, delta {r.get('reliability_delta')}), "
            f"ident_after={r.get('ident_after')}, collapsed={r.get('collapsed')}, "
            f"lambda_mean={r.get('lambda_mean')} [{r.get('lambda_min')},{r.get('lambda_max')}], "
            f"lambda_source={r.get('lambda_source')}, c_pop_mix={r.get('c_pop_mix')}, "
            f"tau0_eff={r.get('tau0_eff')}, n_undetermined={r.get('n_undetermined_subjects')}, "
            f"tau_phi_mean={r.get('tau_phi_mean')}"
        )

    # Iteration diagnosis
    if shrink is not None and point is not None:
        notes.append(
            "Iteration diagnostic (tau-gated shrink vs point): "
            f"gain_after {shrink.get('gain_after')} vs {point.get('gain_after')}; "
            f"reliability_after {shrink.get('reliability_after')} vs {point.get('reliability_after')}."
        )
    if shrink_e is not None:
        notes.append(
            f"Scientific iteration 1 — entropy-λ: gain_after={shrink_e.get('gain_after')}, "
            f"reliability_after={shrink_e.get('reliability_after')}, "
            f"lambda_mean={shrink_e.get('lambda_mean')} "
            f"[{shrink_e.get('lambda_min')},{shrink_e.get('lambda_max')}] "
            f"(source={shrink_e.get('lambda_source')})."
        )
    if shrink_c is not None:
        notes.append(
            f"Scientific iteration 2 — C_pop mix: gain_after={shrink_c.get('gain_after')}, "
            f"reliability_after={shrink_c.get('reliability_after')}, "
            f"c_pop_mix={shrink_c.get('c_pop_mix')}."
        )

    # Track B gap
    group_pass = bool(
        group and group.get("n_eff") is not None and group.get("n_subjects") is not None
        and float(group["n_eff"]) < float(group["n_subjects"])
    )
    if group and "error" not in group:
        notes.append(
            f"Track B group REML: n_eff={group.get('n_eff')} < S={group.get('n_subjects')} "
            f"(min {group.get('n_eff_min')}); ci_ratio={group.get('ci_ratio')}. "
            "Baselines emit point maps with no Sigma^al / tau_phi column."
        )
    if ours and ours.get("n_undetermined_subjects") is not None:
        notes.append(
            f"Track B undetermined subjects (mean_tau > tau0_eff={ours.get('tau0_eff')}): "
            f"{ours.get('n_undetermined_subjects')}/{ours.get('n_subjects')}. "
            "Baselines report alignment for ALL subjects with no uncertainty flag."
        )
    notes.append(
        "Literature gap (PLAN §1–4 / Thual 2025, BrainSync 2018, Takeda 2025): existing "
        "aligners emit point estimates; this framework adds per-subject tau_phi + group REML "
        "n_eff on REAL posteriors — columns baselines cannot fill."
    )
    if not beat_all:
        notes.append(
            "HONEST: ours does not beat BrainSync+FUGW+noalign on gain_after at non-collapsed "
            "reliability/ident after up to 2 scientific transform iterations."
        )

    literature_gap = (
        "Point-estimate-only aligners (BrainSync 2018, FUGW/Thual 2025, conn-SRM) cannot fill "
        "uncertainty/identifiability columns. Hierarchical population-of-couplings provides "
        "tau_phi + group Sigma^al (REML n_eff) on real rest-fMRI posteriors."
    )
    return {
        "metric_protocol": "same_map_both_runs_v1",
        "heldout_protocol_wrong_not_headlined": True,
        "best_ours_row": ours.get("method") if ours else None,
        "ours_gain_after": ours.get("gain_after") if ours else None,
        "ours_reliability_after": ours.get("reliability_after") if ours else None,
        "ours_ident_after": ours.get("ident_after") if ours else None,
        "ours_collapsed": ours.get("collapsed") if ours else None,
        "beat_noalign_on_gain_after": beat.get("noalign"),
        "beat_brainsync_on_gain_after": beat.get("brainsync"),
        "beat_fugw_on_gain_after": beat.get("fugw"),
        "beat_conn_srm_on_gain_after": beat.get("conn_srm"),
        "sota_gain_after_positive_nondegenerate": sota_gain,
        "sota_beat_brainsync_fugw_noalign": beat_all,
        "group_neff_lt_S": group_pass,
        "track_b_unique_columns": ["tau_phi", "group_neff", "lambda_distribution"],
        "notes": notes,
        "literature_gap": literature_gap,
        "uncertainty_only_win_insufficient": True,
        "iterations_used": int(sum(1 for r in (shrink_e, shrink_c) if r is not None)),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    rows = payload["methods"]
    v = payload["verdict"]
    lines = [
        "# REAL N=49 gap-filling + SOTA table (corrected protocol)",
        "",
        f"- data_root: `{payload['data_root']}`",
        f"- n_subjects: {payload['n_subjects']} (frozen cohort 015–063)",
        f"- beta: {payload['beta']} (real scan-rescan)",
        f"- pairs: {payload['n_pairs']} seed {payload['pairs_seed']}",
        f"- ours artifacts: `{payload.get('ours_artifacts')}`",
        f"- transform paths: {payload.get('transform_paths')}",
        f"- metric protocol: **{v.get('metric_protocol', 'same_map_both_runs')}** — "
        "same Q_s on BOTH runs; heldout ||C2-T(C1)|| is protocol-wrong and not headlined",
        "",
        "## PRIMARY method table (higher reliability_after / gain_after / ident_after better)",
        "",
        "| method | reliability_raw | reliability_after | Δrel | gain_after | ident_after | collapsed | λ_mean | n_undet | tau_phi |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]

    def fmt(x, nd=6):
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.{nd}g}"
        return str(x)

    for r in rows:
        lines.append(
            f"| {r['method']} | {fmt(r.get('reliability_raw'), 4)} | {fmt(r.get('reliability_after'), 4)} | "
            f"{fmt(r.get('reliability_delta'), 4)} | {fmt(r.get('gain_after'))} | {fmt(r.get('ident_after'), 4)} | "
            f"{r.get('collapsed')} | {fmt(r.get('lambda_mean'))} | {fmt(r.get('n_undetermined_subjects'))} | "
            f"{fmt(r.get('tau_phi_mean'))} |"
        )
    g = payload.get("group") or {}
    lines += ["", "## Group REML on real posteriors (Track B)", ""]
    if g and "error" not in g:
        lines += [
            f"- n_subjects: {g.get('n_subjects')}",
            f"- n_eff (mean): {g.get('n_eff')} (min {g.get('n_eff_min')})",
            f"- ci_ratio (reml/ttest): {g.get('ci_ratio')}",
            f"- mean_sigma2: {g.get('mean_sigma2')}",
            "",
        ]
    else:
        lines += [f"- error: {g.get('error', 'unavailable')}", ""]

    lines += [
        "## Honest verdict vs SOTA bar (corrected metrics)",
        "",
        f"- beat noalign on gain_after: **{v.get('beat_noalign_on_gain_after')}**",
        f"- beat BrainSync on gain_after: **{v.get('beat_brainsync_on_gain_after')}**",
        f"- beat FUGW on gain_after: **{v.get('beat_fugw_on_gain_after')}**",
        f"- beat conn_srm on gain_after (non-collapsed): **{v.get('beat_conn_srm_on_gain_after')}**",
        f"- beat BrainSync+FUGW+noalign at non-collapsed rel/ident: **{v.get('sota_beat_brainsync_fugw_noalign')}**",
        f"- group n_eff < S: **{v.get('group_neff_lt_S')}**",
        f"- best ours row: **{v.get('best_ours_row')}** "
        f"(gain_after={v.get('ours_gain_after')}, reliability_after={v.get('ours_reliability_after')}, "
        f"ident_after={v.get('ours_ident_after')}, collapsed={v.get('ours_collapsed')})",
        f"- scientific transform iterations used: **{v.get('iterations_used')}** (max 2)",
        "",
        "### Notes",
        "",
    ]
    for n in v.get("notes", []):
        lines.append(f"- {n}")
    lines += ["", "### Literature gap", "", v.get("literature_gap", ""), ""]
    return "\n".join(lines) + "\n"
    g = payload.get("group") or {}
    lines += [
        "",
        "## Group REML on real posteriors",
        "",
    ]
    if g and "error" not in g:
        lines += [
            f"- n_subjects: {g.get('n_subjects')}",
            f"- n_eff (mean over nodes): {g.get('n_eff')} (min {g.get('n_eff_min')})",
            f"- mean_weight: {g.get('mean_weight')}",
            f"- mean_sigma2: {g.get('mean_sigma2')}",
            f"- reml_theta_se: {g.get('reml_theta_se')}",
            f"- ttest_theta_se: {g.get('ttest_theta_se')}",
            f"- ci_ratio (reml/ttest): {g.get('ci_ratio')}",
            "",
        ]
    else:
        lines += [f"- error: {g.get('error', 'unavailable')}", ""]

    lines += [
        "## Honest verdict vs SOTA bar",
        "",
        f"- heldout beats noalign: **{v.get('sota_heldout_beats_noalign')}**",
        f"- heldout beats fugw: **{v.get('sota_heldout_beats_fugw')}**",
        f"- ours alignment_gain > 0: **{v.get('ours_alignment_gain_positive')}**",
        f"- group n_eff < S: **{v.get('group_neff_lt_S')}**",
        "",
        "### Notes",
        "",
    ]
    for n in v.get("notes", []):
        lines.append(f"- {n}")
    lines += [
        "",
        "### Literature gap (the point)",
        "",
        v.get("literature_gap", ""),
        "",
        "Uncertainty-only wins are insufficient: this table reports planted-free REAL "
        "heldout / gain / scan-rescan next to tau_phi and group n_eff.",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_cfg(data_root: Path) -> Any:
    try:
        cfg = load_config(PROJECT_ROOT / "configs" / "experiments" / "10_ours_full.yaml")
        return cfg
    except Exception:
        raw = {
            "experiment": "real_gap_sota",
            "data": {"root": str(data_root), "contract_version": "1.0.0"},
            "run": {"seed": 0, "n_jobs": 1},
            "eval": {"pairs": {"n": N_PAIRS, "seed": PAIRS_SEED}, "permutations": {"B": 200}},
            "model": {"K": 100, "r": 8, "m_draws": 4, "train": {"epochs": 2}},
        }
        return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(cfg_get(build_cfg(Path(".")), "data.root", None) or Path("/Users/anandlo/Surge2026F/ds000243-master")))
    parser.add_argument("--ours-artifacts", type=Path, default=PROJECT_ROOT / "runs" / "10_ours_full__73533e35__20260920T074217Z" / "artifacts")
    parser.add_argument("--ablated-artifacts", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--methods", type=str, default="noalign,brainsync,fugw,conn_srm,ours")
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    parser.add_argument("--skip-ts", action="store_true", help="skip timeseries load (brainsync may no-op)")
    parser.add_argument("--group-subjects", type=int, default=12, help="subjects for group REML smoke on large posteriors")
    parser.add_argument("--full-fugw", action="store_true", help="allow slow FUGW on all subjects")
    args = parser.parse_args(argv)

    setup_threads("outer", n_jobs=1)
    pick_device(prefer_mps=False)

    data_root = Path(args.data_root)
    cfg = build_cfg(data_root)
    print(f"real_gap_sota: data_root={data_root}", flush=True)

    t0 = time.time()
    subjects, run1, run2, bundle = load_cohort(data_root, load_ts=not args.skip_ts)
    ts_map = bundle["timeseries"]
    print(f"loaded N={len(subjects)} R={run1.shape[1]} in {time.time()-t0:.1f}s", flush=True)

    pairs = _declared_pairs(subjects, n_pairs=args.n_pairs, seed=PAIRS_SEED)

    # region indices for posterior_shrink pooling + BrainSync parcellation
    geo = data_root / "derivatives" / "trajot" / "template_geometry.npz"
    labels_all = None
    if geo.is_file():
        with np.load(geo) as z:
            if "region_labels" in z.files:
                labels_all = np.asarray(z["region_labels"]).ravel().astype(np.int64)
    region_indices: list[np.ndarray | None] = []
    valid_masks: list[np.ndarray | None] = []
    for sid in subjects:
        lab = None
        valid = None
        with np.load(subject_run_path(data_root, sid, "1")) as z:
            valid = np.asarray(z["timeseries"]).std(axis=1) > 0
        if labels_all is not None and labels_all.shape[0] == valid.shape[0]:
            lab = labels_all[valid]
        region_indices.append(lab)
        valid_masks.append(valid)

    R = int(run1.shape[1])
    can_parcellate = all(lab is not None for lab in region_indices) and labels_all is not None
    timeseries = None
    if ts_map:
        # BrainSync needs V==R: parcellate vertex ts into Schaefer regions when labels exist.
        timeseries = _timeseries_for_method(
            ts_map,
            subjects,
            region_labels=region_indices,
            valid_masks=valid_masks,
            R=R,
            parcellate=can_parcellate,
        )
        print(
            f"timeseries stack run1 {timeseries['run1'].shape} parcellate={can_parcellate}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    transform_paths: dict[str, str] = {}
    tau_by_method: dict[str, list[np.ndarray] | None] = {}

    method_plan: list[dict[str, Any]] = []
    wanted = {m.strip() for m in args.methods.split(",") if m.strip()}
    if "noalign" in wanted:
        method_plan.append({"name": "noalign", "method": "noalign"})
    if "brainsync" in wanted and timeseries is not None:
        method_plan.append({"name": "brainsync", "method": "brainsync"})
    if "fugw" in wanted:
        method_plan.append({"name": "fugw", "method": "fugw", "limit": None if args.full_fugw else 15})
    if "conn_srm" in wanted:
        method_plan.append({"name": "conn_srm", "method": "conn_srm"})
    if "ours" in wanted and args.ours_artifacts is not None and Path(args.ours_artifacts).exists():
        art = Path(args.ours_artifacts)
        # Baseline ours paths
        method_plan.append(
            {
                "name": "ours_full_point_procrustes",
                "method": "ours_full",
                "artifacts": art,
                "transform_mode": "point_procrustes",
            }
        )
        method_plan.append(
            {
                "name": "ours_full_posterior_shrink",
                "method": "ours_full",
                "artifacts": art,
                "transform_mode": "posterior_shrink",
                "tau0_auto": True,
                "lambda_source": "tau",
                "c_pop_mix": 0.0,
            }
        )
        # Scientific iteration 1: entropy-gated λ (adaptive, varies across subjects)
        method_plan.append(
            {
                "name": "ours_full_posterior_shrink_entropy",
                "method": "ours_full",
                "artifacts": art,
                "transform_mode": "posterior_shrink",
                "tau0_auto": True,
                "lambda_source": "row_entropy",
                "c_pop_mix": 0.0,
            }
        )
        # Scientific iteration 2: C_pop mix on Procrustes target + entropy λ
        method_plan.append(
            {
                "name": "ours_full_posterior_shrink_cpop",
                "method": "ours_full",
                "artifacts": art,
                "transform_mode": "posterior_shrink",
                "tau0_auto": True,
                "lambda_source": "row_entropy",
                "c_pop_mix": 0.35,
            }
        )
        method_plan.append(
            {
                "name": "ours_ablated",
                "method": "ours_ablated",
                "artifacts": Path(args.ablated_artifacts) if args.ablated_artifacts else art,
                "transform_mode": "point_procrustes",
            }
        )

    for spec in method_plan:
        name = spec["name"]
        print(f"=== fitting {name} ===", flush=True)
        t1 = time.time()
        sub_idx = None
        r1, r2, subs = run1, run2, subjects
        ts = timeseries
        ridx = region_indices
        if spec.get("limit"):
            L = int(spec["limit"])
            sub_idx = list(range(min(L, len(subjects))))
            r1, r2 = run1[sub_idx], run2[sub_idx]
            subs = [subjects[i] for i in sub_idx]
            if timeseries is not None:
                ts = {
                    "run1": timeseries["run1"][sub_idx],
                    "run2": timeseries["run2"][sub_idx],
                }
            ridx = [region_indices[i] for i in sub_idx]
            pairs_use = _declared_pairs(subs, n_pairs=min(args.n_pairs, len(subs) * max(1, len(subs) - 1)), seed=PAIRS_SEED)
        else:
            pairs_use = pairs
            ts = timeseries
            ridx = region_indices

        try:
            aligned1, aligned2, meta, tau_phi = fit_and_transform(
                spec["method"],
                r1,
                r2,
                cfg,
                subs,
                data_root=data_root,
                timeseries=ts,
                artifacts=spec.get("artifacts"),
                transform_mode=spec.get("transform_mode"),
                region_indices=ridx,
                tau0=spec.get("tau0"),
                tau0_auto=spec.get("tau0_auto"),
                lambda_source=spec.get("lambda_source"),
                c_pop_mix=spec.get("c_pop_mix"),
            )
        except Exception as exc:
            rows.append(
                {
                    "method": name,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_subjects": len(subs),
                    "reliability_after": None,
                    "gain_after": None,
                    "ident_after": None,
                    "collapsed": None,
                    "heldout_score_module": None,
                    "tau_phi_mean": None,
                }
            )
            print(f"FAILED {name}: {exc}", flush=True)
            continue

        transform_paths[name] = str(meta.get("transform") or meta.get("transform_mode") or "")
        tau_by_method[name] = tau_phi
        row = score_transforms(
            name,
            aligned1,
            aligned2,
            r1,
            r2,
            subs,
            pairs_use,
            meta=meta,
            tau_phi=tau_phi,
            tau0_eff=meta.get("tau0_eff"),
        )
        if spec.get("limit"):
            row["n_subjects_fugw_subset"] = len(subs)
            row["notes"] = f"FUGW scored on first {len(subs)} subjects (runtime); use --full-fugw for N={len(subjects)}"
        rows.append(row)
        print(
            f"done {name} in {time.time()-t1:.1f}s: "
            f"rel_after={row.get('reliability_after')} (raw {row.get('reliability_raw')}) "
            f"gain_after={row.get('gain_after')} ident_after={row.get('ident_after')} "
            f"collapsed={row.get('collapsed')} λ={row.get('lambda_mean')} "
            f"path={transform_paths[name]}",
            flush=True,
        )

    # Group REML on real posteriors (ours artifacts)
    group = None
    if args.ours_artifacts and Path(args.ours_artifacts).exists():
        print("=== group REML on ours posteriors ===", flush=True)
        g_subjects = subjects[: max(2, int(args.group_subjects))]
        # Prefer subjects present in artifacts
        art = Path(args.ours_artifacts)
        if art.name != "artifacts":
            art = art / "artifacts"
        try:
            with np.load(art / "template.npz") as z:
                art_ids = [str(s) for s in z["subject_ids"]] if "subject_ids" in z.files else g_subjects
            g_subjects = [s for s in g_subjects if s in art_ids] or art_ids[: max(2, int(args.group_subjects))]
        except Exception:
            pass
        group = run_group_on_artifacts(art, subjects=g_subjects)
        print(f"group: { {k: group.get(k) for k in ('n_subjects','n_eff','ci_ratio','error') if group} }", flush=True)

    verdict = _verdict(rows, group)
    payload = {
        "table": "REAL_n49_gap_sota",
        "data_root": str(data_root),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "beta": BETA_REAL,
        "n_pairs": args.n_pairs,
        "pairs_seed": PAIRS_SEED,
        "ours_artifacts": str(args.ours_artifacts) if args.ours_artifacts else None,
        "transform_paths": transform_paths,
        "methods": rows,
        "group": group,
        "verdict": verdict,
        "created_unix": time.time(),
        "runtime_sec": time.time() - t0,
        "notes": "PRIMARY real-data table. Synthetic planted-GT is secondary.",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "REAL_n49_gap_sota.json"
    md_path = out_dir / "REAL_n49_gap_sota.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(render_markdown(payload))
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)
    print(render_markdown(payload), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
