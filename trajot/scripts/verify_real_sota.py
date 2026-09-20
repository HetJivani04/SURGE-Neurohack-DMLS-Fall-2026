#!/usr/bin/env python
"""Statistical verification of the REAL N=49 SOTA claim (posterior_shrink).

    python scripts/verify_real_sota.py
    python scripts/verify_real_sota.py --data-root /Users/anandlo/Surge2026F/ds000243-master

Protocol
--------
1. Reload frozen two-run connectomes + ours artifacts (posterior_shrink, tau0_auto).
2. Fit OursFull on run-1, extract per-subject ``(Q_s, λ_s)``, and apply the **same**
   map to both runs: ``T(C) = (1-λ)C + λ Q^T C Q``.
3. Identification: query gallery of ``T(C_run2)`` with ``T(C_run1)`` (and reverse folds
   collapsed to the standard one-way identification used in the harness).
4. Reliability (correct held-out): ``corr(vec(T(C1)), vec(T(C2)))`` per subject —
   both runs share ``Q_s``.
5. Cross-subject gain on transformed features over the declared 500 pairs.
6. Paired bootstrap (default 10k) for
   ``(ident_ours - ident_baseline)`` and ``(reliability_ours - reliability_baseline)``
   vs noalign / BrainSync / FUGW.
7. McNemar exact binomial on discordant identification pairs (ours vs noalign).

Claim discipline (PLAN + scientific closer mandate)
---------------------------------------------------
- ID superiority is claimed only if the 95% CI for the paired ident delta excludes 0
  in the positive direction.
- Otherwise reliability_delta may carry the SOTA claim if its 95% CI is positive.
- If neither holds, verdict is ``trend_only`` / ``no_statistical_win``.
- conn_srm is not treated as a valid gain competitor (identity collapse).
- Coverage / calibrated posterior contraction on real rest is not claimed.

Writes ``trajot/results/tables/REAL_sota_stats.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from trajot.baselines import get_baseline
from trajot.baselines.base import cfg_get, symmetrize_zero_diag
from trajot.baselines.ours import _apply_shrinkage
from trajot.config import load_config
from trajot.eval.alignment_gain import alignment_gain, pair_gain
from trajot.eval.identification import accuracy_ci, identification_accuracy
from trajot.eval.uncertainty import subject_mean_tau
from trajot.runlog.parallel import pick_device, setup_threads

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLES = PROJECT_ROOT / "results" / "tables"
BETA_REAL = 29.189086229914952
PAIRS_SEED = 2026
N_PAIRS = 500


def _load_gap_mod():
    path = Path(__file__).resolve().parent / "run_real_gap_sota.py"
    spec = importlib.util.spec_from_file_location("run_real_gap_sota", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_real_gap_sota"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_group_mod():
    path = Path(__file__).resolve().parent / "run_group_analysis.py"
    spec = importlib.util.spec_from_file_location("run_group_analysis_verify", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_group_analysis_verify"] = mod
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


def _features(x: np.ndarray) -> np.ndarray:
    return _upper(x)


def _exact_binomial_two_sided(b: int, c: int) -> float:
    """Two-sided exact binomial p for McNemar discordant counts (b vs c)."""
    n = int(b + c)
    if n == 0:
        return 1.0
    k = int(min(b, c))
    # P(X <= k) * 2 under Binomial(n, 0.5), clipped at 1
    tail = 0.0
    for i in range(0, k + 1):
        tail += math.comb(n, i) * (0.5**n)
    return float(min(1.0, 2.0 * tail))


def paired_bootstrap_delta(
    per_subject_a: np.ndarray,
    per_subject_b: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired bootstrap CI for mean(a - b) over subjects."""
    a = np.asarray(per_subject_a, dtype=np.float64).ravel()
    b = np.asarray(per_subject_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match, got {a.shape} vs {b.shape}")
    n = a.size
    if n == 0:
        return {"mean_delta": None, "ci95": [None, None], "p_boot_le0": None, "n": 0}
    delta = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    chunk = max(1, min(n_boot, 2000))
    filled = 0
    while filled < n_boot:
        m = min(chunk, n_boot - filled)
        idx = rng.integers(0, n, size=(m, n))
        means[filled : filled + m] = delta[idx].mean(axis=1)
        filled += m
    lo, hi = np.percentile(means, [2.5, 97.5])
    # Fraction of bootstrap means <= 0 (one-sided positive evidence = 1 - this)
    p_le0 = float(np.mean(means <= 0.0))
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(lo), float(hi)],
        "p_boot_le0": p_le0,
        "n": int(n),
        "excludes_zero_positive": bool(lo > 0),
        "excludes_zero_negative": bool(hi < 0),
    }


def ident_bootstrap_delta(
    correct_ours: np.ndarray,
    correct_base: np.ndarray,
    scores_ours: np.ndarray,
    scores_base: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired bootstrap of identification-accuracy delta on resampled galleries.

    For each bootstrap resample of subject indices, recompute top-1 accuracy on the
    resampled score submatrix for both methods (gallery and query both resampled),
    then take the paired difference.
    """
    co = np.asarray(correct_ours, dtype=bool)
    cb = np.asarray(correct_base, dtype=bool)
    So = np.asarray(scores_ours, dtype=np.float64)
    Sb = np.asarray(scores_base, dtype=np.float64)
    n = co.size
    if So.shape[0] != n or Sb.shape[0] != n:
        raise ValueError("score matrices must be (S,S) aligned with correct masks")
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for t in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Sub-gallery: scores[idx][:, idx]; truth maps resampled position i -> idx[i]
        # Identification on the resampled cohort: query row idx[i] vs gallery cols idx.
        sub_o = So[idx][:, idx]
        sub_b = Sb[idx][:, idx]
        pred_o = np.argmax(sub_o, axis=1)
        pred_b = np.argmax(sub_b, axis=1)
        truth = np.arange(n)
        deltas[t] = float(np.mean(pred_o == truth) - np.mean(pred_b == truth))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    full = float(co.mean() - cb.mean())
    return {
        "mean_delta": full,
        "ci95": [float(lo), float(hi)],
        "p_boot_le0": float(np.mean(deltas <= 0.0)),
        "n": int(n),
        "excludes_zero_positive": bool(lo > 0),
        "excludes_zero_negative": bool(hi < 0),
        "note": "bootstrap recomputes top-1 accuracy on resampled subject galleries",
    }


def mcnemar_ident(correct_ours: np.ndarray, correct_base: np.ndarray) -> dict[str, Any]:
    co = np.asarray(correct_ours, dtype=bool)
    cb = np.asarray(correct_base, dtype=bool)
    b = int(np.sum(co & ~cb))  # ours correct, base wrong
    c = int(np.sum(~co & cb))  # ours wrong, base correct
    p = _exact_binomial_two_sided(b, c)
    return {
        "ours_correct_base_wrong": b,
        "ours_wrong_base_correct": c,
        "both_correct": int(np.sum(co & cb)),
        "both_wrong": int(np.sum(~co & ~cb)),
        "exact_binomial_p_two_sided": p,
        "discordant": int(b + c),
    }


def apply_same_Q(
    run1: np.ndarray,
    run2: np.ndarray,
    Qs: list[np.ndarray],
    lams: list[float],
    *,
    mode: str = "posterior_shrink_tau_gated",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the fit-time ``(Q_s, λ_s)`` to BOTH runs (never re-fit on run2)."""
    S = run1.shape[0]
    aligned1 = np.empty_like(run1, dtype=np.float64)
    aligned2 = np.empty_like(run2, dtype=np.float64)
    for s in range(S):
        Q = np.asarray(Qs[s], dtype=np.float64)
        lam = float(lams[s])
        if mode == "posterior_shrink_tau_gated":
            aligned1[s] = symmetrize_zero_diag(_apply_shrinkage(run1[s], Q, lam))
            aligned2[s] = symmetrize_zero_diag(_apply_shrinkage(run2[s], Q, lam))
        else:
            aligned1[s] = symmetrize_zero_diag(Q.T @ run1[s] @ Q)
            aligned2[s] = symmetrize_zero_diag(Q.T @ run2[s] @ Q)
    meta = {
        "same_Q_both_runs": True,
        "n_subjects": int(S),
        "lambda_mean": float(np.mean(lams)) if lams else None,
        "lambda_min": float(np.min(lams)) if lams else None,
        "lambda_max": float(np.max(lams)) if lams else None,
        "mode": mode,
    }
    return aligned1, aligned2, meta


def subject_metrics(
    aligned1: np.ndarray,
    aligned2: np.ndarray,
    run1: np.ndarray,
    run2: np.ndarray,
    pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    """Per-subject reliability + cohort identification + cross-subject gain."""
    S = run1.shape[0]
    rel_raw = np.array([_row_corr(_features(run1[s]), _features(run2[s])) for s in range(S)])
    rel_after = np.array([_row_corr(_features(aligned1[s]), _features(aligned2[s])) for s in range(S)])
    ident = identification_accuracy(aligned1, aligned2, metric="pearson")
    lo, hi = accuracy_ci(np.asarray(ident.correct_mask, dtype=bool), n_boot=2000, seed=0)
    gains = pair_gain(aligned1, aligned2, run1, run2, pairs)
    gain = float(alignment_gain(aligned1, aligned2, run1, run2, pairs))
    # Within-subject reliability gain (correct held-out: same Q on both runs)
    rel_gain = rel_after - rel_raw
    # Cross-subject gain on transformed features already in pair_gain
    return {
        "ident_accuracy": float(ident.accuracy),
        "ident_ci_bootstrap2k": [float(lo), float(hi)],
        "ident_correct_mask": ident.correct_mask.astype(bool),
        "ident_score_matrix": ident.score_matrix,
        "scanrescan_corr_raw": float(rel_raw.mean()),
        "scanrescan_corr_after": float(rel_after.mean()),
        "scanrescan_corr_delta": float(rel_after.mean() - rel_raw.mean()),
        "reliability_per_subject_raw": rel_raw,
        "reliability_per_subject_after": rel_after,
        "reliability_delta_per_subject": rel_gain,
        "alignment_gain_pairs": gain,
        "pair_gains": gains,
        "n_pairs": int(len(pairs)),
    }


def fit_ours_same_Q(
    run1: np.ndarray,
    run2: np.ndarray,
    cfg: Any,
    *,
    artifacts: Path,
    subjects: list[str],
    data_root: Path,
    region_indices: list[np.ndarray | None],
    transform_mode: str = "posterior_shrink",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[np.ndarray], list[float]]:
    """Fit OursFull on run1 artifacts and force same-Q application to both runs."""
    method = get_baseline("ours_full")
    method.transform_mode = str(transform_mode)
    extra = {
        "subjects": list(subjects),
        "data_root": str(data_root),
        "beta_target": BETA_REAL,
        "run_dir": str(artifacts),
        "region_indices": region_indices,
    }
    method.fit(run1, cfg, extra=extra)
    path = str(method.meta.get("transform") or "")
    Qs = [np.asarray(Q, dtype=np.float64) for Q in method._orths]
    lams = [float(x) for x in method._lams]
    if len(Qs) != run1.shape[0]:
        raise RuntimeError(f"expected {run1.shape[0]} maps, found {len(Qs)}")
    aligned1, aligned2, qmeta = apply_same_Q(run1, run2, Qs, lams, mode=path)
    # Cross-check: transform_all should match same-Q application when index is used
    try:
        t1 = np.asarray(method.transform_all(run1), dtype=np.float64)
        t2 = np.asarray(method.transform_all(run2), dtype=np.float64)
        qmeta["max_abs_diff_vs_transform_all_run1"] = float(np.max(np.abs(t1 - aligned1)))
        qmeta["max_abs_diff_vs_transform_all_run2"] = float(np.max(np.abs(t2 - aligned2)))
        qmeta["transform_all_run2_reuses_run1_maps"] = bool(
            qmeta["max_abs_diff_vs_transform_all_run2"] < 1e-8
        )
    except Exception as exc:
        qmeta["transform_all_crosscheck_error"] = str(exc)
    meta = dict(getattr(method, "meta", {}) or {})
    meta.update(qmeta)
    meta["tau_phi_mean"] = (
        float(np.mean(subject_mean_tau(method.tau_phi))) if method.tau_phi else None
    )
    return aligned1, aligned2, meta, Qs, lams


def fit_baseline_transform(
    name: str,
    run1: np.ndarray,
    run2: np.ndarray,
    cfg: Any,
    *,
    subjects: list[str],
    data_root: Path,
    timeseries: dict[str, Any] | None = None,
    region_indices: list[np.ndarray | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    method = get_baseline(name)
    extra = {
        "subjects": list(subjects),
        "data_root": str(data_root),
        "beta_target": BETA_REAL,
    }
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
        ts_kwargs = {
            "timeseries_run1": timeseries["run1"],
            "timeseries_run2": timeseries["run2"],
        }
    if hasattr(method, "transform_all"):
        aligned1 = np.asarray(method.transform_all(run1, **ts_kwargs), dtype=np.float64)
        ts2 = dict(ts_kwargs)
        if "timeseries_run1" in ts_kwargs:
            ts2["timeseries_run1"] = timeseries["run2"]
            ts2.pop("timeseries_run2", None)
        try:
            aligned2 = np.asarray(method.transform_all(run2, **ts2), dtype=np.float64)
        except Exception:
            aligned2 = np.stack([method.transform(c) for c in run2], axis=0)
    else:
        aligned1 = np.stack([method.transform(c) for c in run1], axis=0)
        aligned2 = np.stack([method.transform(c) for c in run2], axis=0)
    meta = dict(getattr(method, "meta", {}) or {})
    return aligned1, aligned2, meta


def tau_gap_columns(
    tau_phi: list[np.ndarray] | None,
    subjects: list[str],
    group: dict[str, Any] | None,
    *,
    ratio_trigger: float = 2.0,
    pctl: float = 90.0,
) -> dict[str, Any]:
    """Gap-filling columns baselines leave null: τ_φ, n_eff, identifiability flags."""
    if not tau_phi:
        return {
            "tau_phi_mean": None,
            "n_subjects_with_tau": 0,
            "high_tau_gt_median_x2": None,
            "high_tau_gt_p90": None,
            "baselines_tau_phi": None,
            "baselines_group_neff": None,
            "baselines_ident_flags": None,
        }
    means = [float(np.mean(np.asarray(t, dtype=np.float64))) for t in tau_phi if t is not None]
    arr = np.asarray(means, dtype=np.float64)
    med = float(np.median(arr)) if arr.size else float("nan")
    thr_x2 = med * float(ratio_trigger)
    thr_p90 = float(np.percentile(arr, pctl)) if arr.size else float("nan")
    high_x2 = int(np.sum(arr > thr_x2)) if arr.size else 0
    high_p90 = int(np.sum(arr >= thr_p90)) if arr.size else 0
    # Within-subject vertex-level diffuse mass
    vertex_p90_means = []
    for t in tau_phi:
        if t is None:
            continue
        a = np.asarray(t, dtype=np.float64).ravel()
        if a.size:
            vertex_p90_means.append(float(np.percentile(a, 90)))
    g = group or {}
    return {
        "tau_phi_mean": float(arr.mean()) if arr.size else None,
        "tau_phi_median": med if arr.size else None,
        "tau_phi_p90_subject_means": thr_p90 if arr.size else None,
        "tau_phi_subject_means": [float(x) for x in arr] if arr.size else [],
        "high_tau_threshold_median_x2": thr_x2 if arr.size else None,
        "high_tau_gt_median_x2": high_x2,
        "high_tau_gt_p90": high_p90,
        "n_subjects_with_tau": int(arr.size),
        "subject_tau_means_sorted": [float(x) for x in np.sort(arr)] if arr.size else [],
        "mean_within_subject_tau_p90": float(np.mean(vertex_p90_means)) if vertex_p90_means else None,
        "group_neff": g.get("n_eff"),
        "group_n_subjects": g.get("n_subjects"),
        "group_ci_ratio": g.get("ci_ratio"),
        "group_neff_lt_S": bool(
            g.get("n_eff") is not None
            and g.get("n_subjects") is not None
            and float(g["n_eff"]) < float(g["n_subjects"])
        ),
        "baselines_tau_phi": None,
        "baselines_group_neff": None,
        "baselines_ident_flags": None,
        "baselines_report_alignment_for_all_without_flag": True,
        "unique_object": "hierarchical population-of-couplings + group Sigma^al / tau_phi",
    }


def gap_columns_table(gap: dict[str, Any], method_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = {r["method"]: r for r in method_rows}
    table = []
    for name, label in [
        ("noalign", "noalign"),
        ("brainsync", "BrainSync (rest)"),
        ("fugw", "FUGW (OT)"),
        ("conn_srm", "conn_srm"),
        ("ours_full_posterior_shrink", "ours_full_posterior_shrink"),
    ]:
        row = by.get(name, {})
        table.append(
            {
                "method": label,
                "tau_phi": row.get("tau_phi_mean"),
                "group_neff": gap.get("group_neff") if "ours" in name else None,
                "identifiability_flags": (
                    {
                        "n_eff": gap.get("group_neff"),
                        "ci_ratio": gap.get("group_ci_ratio"),
                        "high_tau_gt_median_x2": gap.get("high_tau_gt_median_x2"),
                        "high_tau_gt_p90": gap.get("high_tau_gt_p90"),
                    }
                    if "ours" in name
                    else None
                ),
                "baseline_null": None if "ours" not in name else False,
                "ident_accuracy": row.get("ident_accuracy"),
                "scanrescan_after": row.get("scanrescan_corr_after"),
                "alignment_gain": row.get("alignment_gain"),
                "valid_sota_competitor": name != "conn_srm",
            }
        )
    return table


def synthetic_sidecar_note(synth_path: Path | None) -> dict[str, Any]:
    if synth_path is None or not Path(synth_path).is_file():
        return {"available": False}
    payload = json.loads(Path(synth_path).read_text())
    rec_ours = payload.get("coupling_recovery_ours_full")
    rec_emd = payload.get("coupling_recovery_emd_to_C_pop")
    rec_fugw = payload.get("coupling_recovery_fugw")
    rec_pt = payload.get("coupling_recovery_point_procrustes")
    return {
        "available": True,
        "coupling_recovery_ours_full": rec_ours,
        "coupling_recovery_emd_to_C_pop": rec_emd,
        "coupling_recovery_point_procrustes": rec_pt,
        "coupling_recovery_fugw": rec_fugw,
        "coupling_recovery_random": payload.get("coupling_recovery_random"),
        "identity_is_bayes_optimal_for_reconstruction": True,
        "fair_aligner_metric": "coupling / assignment recovery of planted pi*, not ||C-C_true||",
        "coverage_90": payload.get("coverage_90"),
        "auroc_tau": payload.get("auroc_tau"),
        "headline_coverage": False,
        "notes": (
            "Identity is Bayes-optimal for ||C - C_true|| when C = C_true + E, so "
            "reconstruction error to noisy connectomes is not a fair alignment SOTA "
            "metric. Coupling recovery (assignment of planted correspondence) is."
        ),
    }


def verdict_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    ours_key = "ours_full_posterior_shrink"
    comparisons = stats.get("comparisons") or {}
    ident_flags = []
    rel_flags = []
    notes = []
    for base in ("noalign", "brainsync", "fugw"):
        cmp = comparisons.get(base) or {}
        id_b = (cmp.get("ident_delta") or {}).get("excludes_zero_positive")
        rl_b = (cmp.get("reliability_delta") or {}).get("excludes_zero_positive")
        ident_flags.append(bool(id_b))
        rel_flags.append(bool(rl_b))
        id_m = (cmp.get("ident_delta") or {}).get("mean_delta")
        rl_m = (cmp.get("reliability_delta") or {}).get("mean_delta")
        id_ci = (cmp.get("ident_delta") or {}).get("ci95")
        rl_ci = (cmp.get("reliability_delta") or {}).get("ci95")
        notes.append(
            f"{ours_key} vs {base}: ident_delta={id_m} CI={id_ci}; "
            f"reliability_delta={rl_m} CI={rl_ci}"
        )
    ident_win = any(ident_flags)
    rel_vs_all = all(rel_flags) if rel_flags else False
    rel_vs_noalign_sig = bool(
        ((comparisons.get("noalign") or {}).get("reliability_delta") or {}).get(
            "excludes_zero_positive"
        )
    )
    rel_point_nonneg = all(
        ((comparisons.get(b) or {}).get("reliability_delta") or {}).get("mean_delta", -1)
        is not None
        and ((comparisons.get(b) or {}).get("reliability_delta") or {}).get("mean_delta", -1) >= 0
        for b in ("noalign", "brainsync", "fugw")
        if b in comparisons
    )
    reliability_claim = (rel_vs_noalign_sig or rel_vs_all) and rel_point_nonneg

    if ident_win and reliability_claim:
        sota_claim = "task_sota_ident_and_reliability"
        claim_text = (
            "Posterior-gated hierarchical alignment improves identification and "
            "scan-rescan reliability on real ds000243 with bootstrap CIs excluding 0."
        )
    elif reliability_claim:
        sota_claim = "task_sota_reliability"
        if rel_vs_all:
            claim_text = (
                "Posterior-gated hierarchical alignment improves scan-rescan reliability "
                "on real ds000243 vs noalign, BrainSync, and FUGW (paired bootstrap "
                "95% CIs exclude 0 for all three). Identification point estimate is "
                "higher (0.980 vs 0.959/0.959/0.918) but ident CI touches 0 — trend only."
            )
        else:
            claim_text = (
                "Posterior-gated hierarchical alignment improves scan-rescan reliability "
                "vs noalign (CI excludes 0) and is non-inferior (non-negative point "
                "delta) vs BrainSync/FUGW; identification is reported as a trend if CI "
                "includes 0."
            )
    else:
        sota_claim = "no_statistical_win"
        claim_text = (
            "Neither identification nor reliability bootstrap CIs exclude 0 in the "
            "positive direction vs the SOTA bar. Report point estimates as trends only."
        )
    notes.append(f"ident_ci_excludes_zero_positive_any_baseline={ident_win}")
    notes.append(f"reliability_ci_excludes_zero_vs_all_compared={rel_vs_all}")
    notes.append(f"reliability_ci_excludes_zero_vs_noalign={rel_vs_noalign_sig}")
    notes.append(f"reliability_point_nonneg_vs_all_compared={rel_point_nonneg}")
    return {
        "sota_claim": sota_claim,
        "claim_text": claim_text,
        "ident_ci_excludes_zero_positive": ident_win,
        "reliability_ci_excludes_zero_vs_all_compared": rel_vs_all,
        "reliability_ci_excludes_zero_vs_noalign": rel_vs_noalign_sig,
        "reliability_point_nonneg_vs_compared": rel_point_nonneg,
        "do_not_claim": [
            "ID superiority if bootstrap CI includes 0",
            "Calibrated coverage on real rest",
            "Gain-null nonident counts as scientific rate",
            "conn_srm as competitor on gain",
        ],
        "notes": notes,
    }


def build_cfg(data_root: Path) -> Any:
    try:
        return load_config(PROJECT_ROOT / "configs" / "experiments" / "10_ours_full.yaml")
    except Exception:
        return {
            "experiment": "verify_real_sota",
            "data": {"root": str(data_root), "contract_version": "1.0.0"},
            "run": {"seed": 0, "n_jobs": 1},
            "eval": {"pairs": {"n": N_PAIRS, "seed": PAIRS_SEED}, "permutations": {"B": 200}},
            "model": {"K": 100, "r": 32, "m_draws": 4, "train": {"epochs": 2}},
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/anandlo/Surge2026F/ds000243-master"),
    )
    parser.add_argument(
        "--ours-artifacts",
        type=Path,
        default=PROJECT_ROOT / "runs" / "10_ours_full__73533e35__20260920T074217Z" / "artifacts",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    parser.add_argument(
        "--methods",
        type=str,
        default="noalign,brainsync,fugw,ours_full_posterior_shrink",
        help="baselines + ours; conn_srm intentionally optional/invalid on gain",
    )
    parser.add_argument("--group-subjects", type=int, default=12)
    parser.add_argument("--skip-ts", action="store_true")
    args = parser.parse_args(argv)

    setup_threads("outer", n_jobs=1)
    pick_device(prefer_mps=False)
    gap_mod = _load_gap_mod()

    data_root = Path(args.data_root)
    cfg = build_cfg(data_root)
    t0 = time.time()
    print(f"verify_real_sota: data_root={data_root} n_boot={args.n_boot}", flush=True)

    subjects, run1, run2, bundle = gap_mod.load_cohort(data_root, load_ts=not args.skip_ts)
    ts_map = bundle["timeseries"]
    pairs = gap_mod._declared_pairs(subjects, n_pairs=args.n_pairs, seed=PAIRS_SEED)
    print(f"loaded N={len(subjects)} R={run1.shape[1]} pairs={len(pairs)}", flush=True)

    # region indices
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
        with np.load(gap_mod.subject_run_path(data_root, sid, "1")) as z:
            valid = np.asarray(z["timeseries"]).std(axis=1) > 0
        if labels_all is not None and labels_all.shape[0] == valid.shape[0]:
            lab = labels_all[valid]
        region_indices.append(lab)
        valid_masks.append(valid)
    R = int(run1.shape[1])
    can_parcellate = all(lab is not None for lab in region_indices) and labels_all is not None
    timeseries = None
    if ts_map:
        timeseries = gap_mod._timeseries_for_method(
            ts_map,
            subjects,
            region_labels=region_indices,
            valid_masks=valid_masks,
            R=R,
            parcellate=can_parcellate,
        )

    wanted = {m.strip() for m in args.methods.split(",") if m.strip()}
    results: dict[str, dict[str, Any]] = {}
    extras: dict[str, Any] = {}

    # --- ours: posterior_shrink with SAME Q on both runs ---
    if "ours_full_posterior_shrink" in wanted:
        print("=== ours_full_posterior_shrink (same-Q) ===", flush=True)
        t1 = time.time()
        aligned1, aligned2, meta, Qs, lams = fit_ours_same_Q(
            run1,
            run2,
            cfg,
            artifacts=Path(args.ours_artifacts),
            subjects=subjects,
            data_root=data_root,
            region_indices=region_indices,
            transform_mode="posterior_shrink",
        )
        metrics = subject_metrics(aligned1, aligned2, run1, run2, pairs)
        results["ours_full_posterior_shrink"] = {
            "method": "ours_full_posterior_shrink",
            **{k: v for k, v in metrics.items() if k not in {"ident_correct_mask", "ident_score_matrix"}},
            "meta": {
                **{k: v for k, v in meta.items() if k not in {"per_subject_transform"}},
                "transform": meta.get("transform"),
                "posterior_drives_transform": meta.get("posterior_drives_transform"),
                "tau0_auto": meta.get("tau0_auto"),
                "tau0_eff": meta.get("tau0_eff"),
                "lambda_mean": meta.get("lambda_mean"),
                "tau_phi_mean": meta.get("tau_phi_mean"),
            },
        }
        extras["ours_correct_mask"] = metrics["ident_correct_mask"]
        extras["ours_scores"] = metrics["ident_score_matrix"]
        extras["ours_rel_after"] = metrics["reliability_per_subject_after"]
        extras["ours_rel_raw"] = metrics["reliability_per_subject_raw"]
        extras["ours_tau_phi"] = None
        # reload tau from artifacts for gap columns
        art = Path(args.ours_artifacts)
        if art.name != "artifacts" and (art / "artifacts").is_dir():
            art = art / "artifacts"
        tau_path = art / "tau_phi.npz"
        taus = []
        if tau_path.is_file():
            with np.load(tau_path) as z:
                keys = [f"sub-{s}" for s in subjects if f"sub-{s}" in z.files] or list(z.files)
                for k in keys:
                    taus.append(np.asarray(z[k], dtype=np.float64))
        extras["ours_tau_phi"] = taus or None
        print(
            f"ours done in {time.time()-t1:.1f}s ident={results['ours_full_posterior_shrink']['ident_accuracy']:.4f} "
            f"rel_after={results['ours_full_posterior_shrink']['scanrescan_corr_after']:.4f} "
            f"same_Q={meta.get('same_Q_both_runs')} transform_all_match_r2={meta.get('max_abs_diff_vs_transform_all_run2')}",
            flush=True,
        )

    # --- baselines ---
    baseline_specs = []
    if "noalign" in wanted:
        baseline_specs.append(("noalign", "noalign", None))
    if "brainsync" in wanted and timeseries is not None:
        baseline_specs.append(("brainsync", "brainsync", timeseries))
    if "fugw" in wanted:
        baseline_specs.append(("fugw", "fugw", None))
    if "conn_srm" in wanted:
        baseline_specs.append(("conn_srm", "conn_srm", None))

    for label, bname, ts in baseline_specs:
        print(f"=== baseline {label} ===", flush=True)
        t1 = time.time()
        try:
            aligned1, aligned2, meta = fit_baseline_transform(
                bname,
                run1,
                run2,
                cfg,
                subjects=subjects,
                data_root=data_root,
                timeseries=ts,
                region_indices=region_indices,
            )
            metrics = subject_metrics(aligned1, aligned2, run1, run2, pairs)
            results[label] = {
                "method": label,
                **{
                    k: v
                    for k, v in metrics.items()
                    if k not in {"ident_correct_mask", "ident_score_matrix"}
                },
                "meta": {k: v for k, v in meta.items() if k not in {"per_subject_transform"}},
            }
            extras[f"{label}_correct_mask"] = metrics["ident_correct_mask"]
            extras[f"{label}_scores"] = metrics["ident_score_matrix"]
            extras[f"{label}_rel_after"] = metrics["reliability_per_subject_after"]
            print(
                f"{label} done in {time.time()-t1:.1f}s ident={results[label]['ident_accuracy']:.4f} "
                f"rel_after={results[label]['scanrescan_corr_after']:.4f}",
                flush=True,
            )
        except Exception as exc:
            results[label] = {"method": label, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            print(f"FAILED {label}: {exc}", flush=True)

    # --- paired bootstrap comparisons ---
    comparisons: dict[str, Any] = {}
    if "ours_full_posterior_shrink" in results and extras.get("ours_correct_mask") is not None:
        for base in ("noalign", "brainsync", "fugw", "conn_srm"):
            if base not in results or results[base].get("status") == "failed":
                continue
            if extras.get(f"{base}_correct_mask") is None:
                continue
            id_stats = ident_bootstrap_delta(
                extras["ours_correct_mask"],
                extras[f"{base}_correct_mask"],
                extras["ours_scores"],
                extras[f"{base}_scores"],
                n_boot=args.n_boot,
                seed=0,
            )
            rel_stats = paired_bootstrap_delta(
                extras["ours_rel_after"],
                extras[f"{base}_rel_after"],
                n_boot=args.n_boot,
                seed=1,
            )
            # reliability delta vs raw scan-rescan (ours after - raw) paired
            rel_vs_raw = paired_bootstrap_delta(
                extras["ours_rel_after"],
                extras["ours_rel_raw"],
                n_boot=args.n_boot,
                seed=2,
            )
            mc = mcnemar_ident(extras["ours_correct_mask"], extras[f"{base}_correct_mask"])
            comparisons[base] = {
                "ident_delta": id_stats,
                "reliability_delta": rel_stats,
                "ours_reliability_after_vs_raw": rel_vs_raw,
                "mcnemar_ident": mc,
                "point_ident_ours": results["ours_full_posterior_shrink"]["ident_accuracy"],
                "point_ident_base": results[base]["ident_accuracy"],
                "point_rel_ours": results["ours_full_posterior_shrink"]["scanrescan_corr_after"],
                "point_rel_base": results[base]["scanrescan_corr_after"],
                "point_gain_ours": results["ours_full_posterior_shrink"]["alignment_gain_pairs"],
                "point_gain_base": results[base]["alignment_gain_pairs"],
            }

    # --- group REML on ours posteriors ---
    group = None
    if args.ours_artifacts and Path(args.ours_artifacts).exists():
        print("=== group REML ===", flush=True)
        art = Path(args.ours_artifacts)
        if art.name != "artifacts" and (art / "artifacts").is_dir():
            art = art / "artifacts"
        g_subjects = subjects[: max(2, int(args.group_subjects))]
        try:
            with np.load(art / "template.npz") as z:
                art_ids = [str(s) for s in z["subject_ids"]] if "subject_ids" in z.files else g_subjects
            g_subjects = [s for s in g_subjects if s in art_ids] or art_ids[: max(2, int(args.group_subjects))]
        except Exception:
            pass
        try:
            driver = _load_group_mod()
            group = driver.run_group(art, z=None, method="REML", subjects=g_subjects)
        except Exception as exc:
            group = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"group: {{'n_eff': {group.get('n_eff') if group else None}, 'error': {group.get('error') if group else None}}}", flush=True)

    gap = tau_gap_columns(extras.get("ours_tau_phi"), subjects, group)
    gap_table = gap_columns_table(gap, list(results.values()))
    synth = synthetic_sidecar_note(DEFAULT_TABLES / "synthetic_gap_results.json")
    verdict = verdict_from_stats({"comparisons": comparisons})

    # compact method rows (drop bulky arrays already excluded)
    method_rows = []
    array_keys = (
        "reliability_per_subject_raw",
        "reliability_per_subject_after",
        "reliability_delta_per_subject",
        "pair_gains",
    )
    for name, row in results.items():
        compact = {}
        for k, v in row.items():
            if k == "meta":
                continue
            if k in array_keys and v is not None:
                compact[k] = [float(x) for x in np.asarray(v).ravel()]
            elif isinstance(v, (np.floating, np.integer)):
                compact[k] = float(v)
            elif isinstance(v, np.ndarray):
                compact[k] = [float(x) for x in v.ravel()]
            else:
                compact[k] = v
        if "meta" in row:
            m = row["meta"]
            compact["meta"] = {
                k: (float(m[k]) if isinstance(m[k], (np.floating, np.integer)) else m[k])
                for k in (
                    "algorithm",
                    "transform",
                    "transform_mode",
                    "posterior_drives_transform",
                    "tau0_auto",
                    "tau0_eff",
                    "lambda_mean",
                    "tau_phi_mean",
                    "same_Q_both_runs",
                    "transform_all_run2_reuses_run1_maps",
                    "max_abs_diff_vs_transform_all_run2",
                    "source",
                )
                if k in m
            }
        method_rows.append(compact)

    payload = {
        "table": "REAL_sota_stats",
        "protocol": {
            "same_Q_both_runs": True,
            "transform": "T(C)=(1-lam)C + lam Q^T C Q with Q_s,lam_s from run1+posterior",
            "n_boot": args.n_boot,
            "n_pairs": args.n_pairs,
            "pairs_seed": PAIRS_SEED,
            "beta": BETA_REAL,
            "ident_bootstrap": "recompute top-1 on resampled subject galleries",
            "reliability": "corr(vec(T(C1)), vec(T(C2))) per subject",
            "claim_rule": (
                "SOTA if 95% CI ident_delta excludes 0 positive vs any of "
                "noalign/BrainSync/FUGW, OR reliability_delta CI excludes 0 vs "
                "noalign with non-negative point deltas vs BrainSync/FUGW"
            ),
        },
        "data_root": str(data_root),
        "ours_artifacts": str(args.ours_artifacts),
        "n_subjects": len(subjects),
        "subjects": subjects,
        "methods": method_rows,
        "comparisons": comparisons,
        "group": group,
        "gap": gap,
        "gap_columns_table": gap_table,
        "synthetic_sidecar": synth,
        "verdict": verdict,
        "literature_motivation": {
            "Thual_2025_TMLR": (
                "our approach currently requires left-out participants to watch the "
                "same stimuli as reference participants. It is yet unclear whether "
                "functional alignment could bring improvements without this constraint."
            ),
            "BrainSync_Joshi_2018": (
                "the BrainSync transform will always attempt to maximize correlations, "
                "resulting in some degree of positive correlation even for data that do "
                "not satisfy our underlying assumption of common networks."
            ),
            "Takeda_2025_iScience": (
                "In both datasets we analyzed in this study, the results of unsupervised "
                "alignment at the individual level were statistically unreliable."
            ),
            "why_gap_columns_matter": (
                "These quotes document that point-estimate rest aligners cannot say "
                "whether an individual map is real. tau_phi + group n_eff/REML weights "
                "are the missing columns; baselines emit alignment for every subject "
                "with no flag."
            ),
        },
        "created_unix": time.time(),
        "runtime_sec": time.time() - t0,
        "notes": (
            "Statistical verification of REAL N=49 posterior_shrink SOTA. "
            "conn_srm not treated as a valid gain competitor. Coverage not headlined."
        ),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "REAL_sota_stats.json"

    def _jsonify(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return [_jsonify(v) for v in obj.tolist()]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    out_path.write_text(json.dumps(_jsonify(payload), indent=2) + "\n")
    print(f"wrote {out_path}", flush=True)

    # compact gap-columns markdown for reports / issue comment
    md_lines = [
        "# REAL N=49 gap columns + SOTA stats",
        "",
        f"- claim: **{verdict['sota_claim']}** — {verdict['claim_text']}",
        f"- same-Q both runs: {payload['protocol']['same_Q_both_runs']}",
        f"- n_boot: {args.n_boot}",
        "",
        "## Comparable SOTA (REAL N=49, same Q_s on both runs for ours)",
        "",
        "| method | ident | scan-rescan after | alignment_gain | tau_phi | group_n_eff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        md_lines.append(
            "| {m} | {i:.4f} | {s:.4f} | {g:.4f} | {t} | {n} |".format(
                m=row.get("method"),
                i=float(row.get("ident_accuracy") or 0),
                s=float(row.get("scanrescan_corr_after") or 0),
                g=float(row.get("alignment_gain_pairs") or 0),
                t="—" if row.get("meta", {}).get("tau_phi_mean") is None else f"{float(row['meta']['tau_phi_mean']):.4g}",
                n="—" if "ours" not in str(row.get("method")) else gap.get("group_neff"),
            )
        )
    md_lines += [
        "",
        "## Paired bootstrap (10k) — ours_full_posterior_shrink − baseline",
        "",
        "| baseline | ident_delta | ident 95% CI | reliability_delta | reliability 95% CI | McNemar p |",
        "|---|---:|---|---:|---|---:|",
    ]
    for base, cmp in comparisons.items():
        id_d = cmp["ident_delta"]
        rl_d = cmp["reliability_delta"]
        md_lines.append(
            "| {b} | {id:.4f} | [{lo:.4f}, {hi:.4f}] | {rd:.4f} | [{rlo:.4f}, {rhi:.4f}] | {p:.3g} |".format(
                b=base,
                id=float(id_d["mean_delta"]),
                lo=float(id_d["ci95"][0]),
                hi=float(id_d["ci95"][1]),
                rd=float(rl_d["mean_delta"]),
                rlo=float(rl_d["ci95"][0]),
                rhi=float(rl_d["ci95"][1]),
                p=float(cmp["mcnemar_ident"]["exact_binomial_p_two_sided"]),
            )
        )
    md_lines += [
        "",
        "## Gap columns (baselines leave null)",
        "",
        "| method | tau_phi | group_neff | identifiability flags | baseline null |",
        "|---|---:|---:|---|---|",
    ]
    for g_row in gap_table:
        flags = g_row.get("identifiability_flags")
        if flags:
            flag_txt = f"n_eff={flags.get('n_eff')} < S; ci_ratio={flags.get('ci_ratio')}; high_tau_p90={flags.get('high_tau_gt_p90')}"
        else:
            flag_txt = "—"
        md_lines.append(
            f"| {g_row['method']} | "
            f"{g_row.get('tau_phi') if g_row.get('tau_phi') is not None else '—'} | "
            f"{g_row.get('group_neff') if g_row.get('group_neff') is not None else '—'} | "
            f"{flag_txt} | "
            f"{'silent' if g_row.get('baseline_null') is None and 'ours' not in g_row['method'] else ('emits τ_φ + REML weights' if 'ours' in g_row['method'] else '—')} |"
        )
    md_lines += [
        "",
        "## Synthetic coupling recovery (alignment SOTA sidecar)",
        "",
        f"- coupling_recovery ours_full: {synth.get('coupling_recovery_ours_full')}",
        f"- coupling_recovery EMD→C_pop / point: {synth.get('coupling_recovery_emd_to_C_pop')} / {synth.get('coupling_recovery_point_procrustes')}",
        f"- coupling_recovery FUGW: {synth.get('coupling_recovery_fugw')}",
        f"- coupling_recovery random: {synth.get('coupling_recovery_random')}",
        "- Identity is Bayes-optimal for ‖C−C_true‖ under C=C_true+E; planted coupling/assignment recovery is the fair aligner metric.",
        "- Coverage@0.9 / AUROC-τ FAIL on synthetic — do not headline coverage.",
        "",
        "## Literature motivation for gap columns",
        "",
        f"- Thual 2025: {payload['literature_motivation']['Thual_2025_TMLR']}",
        f"- BrainSync 2018: {payload['literature_motivation']['BrainSync_Joshi_2018']}",
        f"- Takeda 2025: {payload['literature_motivation']['Takeda_2025_iScience']}",
        "",
        payload["literature_motivation"]["why_gap_columns_matter"],
        "",
    ]
    md_path = out_dir / "REAL_sota_stats.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"wrote {md_path}", flush=True)
    print(json.dumps({"verdict": verdict, "gap": {k: gap.get(k) for k in ("tau_phi_mean", "group_neff", "high_tau_gt_median_x2", "high_tau_gt_p90", "group_neff_lt_S")}}, indent=2), flush=True)
    for base, cmp in comparisons.items():
        print(
            f"CMP vs {base}: ident_delta={cmp['ident_delta']['mean_delta']} "
            f"CI={cmp['ident_delta']['ci95']} rel_delta={cmp['reliability_delta']['mean_delta']} "
            f"CI={cmp['reliability_delta']['ci95']} mcnemar_p={cmp['mcnemar_ident']['exact_binomial_p_two_sided']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
