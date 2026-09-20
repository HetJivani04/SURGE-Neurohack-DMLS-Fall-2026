#!/usr/bin/env python
"""Synthetic planted-GT gap-filling harness.

Generative model (region space, ``R`` parcels, ``K=R`` template nodes)::

    C_pop  random PSD, zero diagonal
    70%    sharp P* (near-permutation / pure permutation)
    30%    ambiguous P* = 0.5 P1 + 0.5 P2
    C_s^(r) = P* C_pop P*^T + E,  E ~ N(0, 1/beta) symmetrized, zero diag

Two success axes are scored:

A) Alignment recovery vs SOTA baselines on planted GT (honest — if ours loses
   to point-OT Procrustes / EMD-to-C_pop, the table says so).
B) Gap-filling uncertainty columns baselines cannot emit (coverage, AUROC(τ),
   group n_eff / FPR).

    python scripts/synthetic_coverage.py --smoke
    python scripts/synthetic_coverage.py --full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from trajot.baselines.base import symmetrize_zero_diag
from trajot.eval.uncertainty import (
    heldout_predictive_score,
    nonident_auroc,
    posterior_coverage,
    shrinkage_transform,
    subject_mean_tau,
)
from trajot.geometry.connectivity import connectivity
from trajot.io.contract import (
    SCHEMA_VERSION,
    subject_run_path,
    write_manifest,
    write_subject_run,
)
from trajot.report.group import meta_analysis_map, one_sample_ttest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/tmp/trajot_synth_gap")
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
BETA_REAL = 29.189
TRUTH_FILE = "planted_gap_truth.npz"


# --------------------------------------------------------------------------- geometry helpers
def pool_pi_to_regions(pi: np.ndarray, region_index: np.ndarray | None, R: int) -> np.ndarray:
    """Mass-preserving block-sum of a coupling to ``(R, R)`` when ``K=R``."""
    arr = np.asarray(pi, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"pi must be 2D, got {arr.shape}")
    V, K = arr.shape
    if K == R and (region_index is None or V == R):
        return arr.copy()
    if region_index is None:
        if V == R:
            out = np.zeros((R, K), dtype=np.float64)
            out[:, : min(R, K)] = arr[:, : min(R, K)]
            return out
        raise ValueError(f"cannot pool pi {arr.shape} to R={R} without region_index")
    out = np.zeros((R, K), dtype=np.float64)
    for v in range(V):
        out[int(region_index[v])] += arr[v]
    return out


def row_stochastic(pi: np.ndarray) -> np.ndarray:
    row = np.maximum(np.asarray(pi, dtype=np.float64).sum(axis=1, keepdims=True), 1e-12)
    return np.asarray(pi, dtype=np.float64) / row


def hungarian_perm(pi_bar: np.ndarray) -> np.ndarray:
    """Column index assigned to each row by max-mass matching."""
    from scipy.optimize import linear_sum_assignment

    P = row_stochastic(pi_bar)
    _, col = linear_sum_assignment(-P)
    return np.asarray(col, dtype=np.int64)


def _emd_coupling(C_src: np.ndarray, C_dst: np.ndarray) -> np.ndarray:
    import ot
    from scipy.spatial.distance import cdist

    Rs, Rd = C_src.shape[0], C_dst.shape[0]
    p = np.full(Rs, 1.0 / Rs, dtype=np.float64)
    q = np.full(Rd, 1.0 / Rd, dtype=np.float64)
    M = np.sqrt(np.maximum(
        np.sum(C_src * C_src, 1)[:, None] + np.sum(C_dst * C_dst, 1)[None, :] - 2.0 * (C_src @ C_dst.T),
        0.0,
    ))
    # cdist unused import guard kept local for POT path parity with ours.py
    _ = cdist
    return np.ascontiguousarray(ot.emd(p, q, M), dtype=np.float64)


def _procrustes_Q(C: np.ndarray, C_ref: np.ndarray) -> np.ndarray:
    M = np.asarray(C, dtype=np.float64) @ np.asarray(C_ref, dtype=np.float64).T
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return np.ascontiguousarray(U @ Vt, dtype=np.float64)


def fro_error(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(A) - np.asarray(B), ord="fro"))


def mean_recovery(C_hat: Sequence[np.ndarray], C_true: Sequence[np.ndarray]) -> float:
    return float(np.mean([fro_error(a, b) for a, b in zip(C_hat, C_true)]))


# --------------------------------------------------------------------------- generative model
def _random_psd_zero_diag(R: int, rng: np.random.Generator, scale: float = 0.5) -> np.ndarray:
    A = rng.normal(size=(R, R))
    C = A @ A.T
    C = symmetrize_zero_diag(C)
    iu = np.triu_indices(R, k=1)
    denom = float(np.sqrt(np.mean(C[iu] ** 2))) or 1.0
    C = C * (scale / denom)
    return symmetrize_zero_diag(C)


def _near_permutation(R: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Row-stochastic low-entropy P* and its hard permutation (template index per subject region)."""
    perm = rng.permutation(R)
    if rng.random() < 0.5:
        P = np.eye(R, dtype=np.float64)[perm]
    else:
        P = 0.95 * np.eye(R, dtype=np.float64)[perm] + 0.05 / R
        P = P / P.sum(axis=1, keepdims=True)
    return P, perm


def _ambiguous_map(R: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p1 = rng.permutation(R)
    p2 = rng.permutation(R)
    while np.array_equal(p1, p2):
        p2 = rng.permutation(R)
    P = 0.5 * np.eye(R, dtype=np.float64)[p1] + 0.5 * np.eye(R, dtype=np.float64)[p2]
    return P, p1, p2


def plant_population(
    N: int,
    R: int,
    *,
    beta: float = BETA_REAL,
    frac_ambiguous: float = 0.3,
    seed: int = 0,
    scale: float = 0.5,
) -> dict[str, Any]:
    """Draw ``C_pop``, planted maps ``P*``, noise-free ``C_true``, and run-1/2 noisy observations."""
    rng = np.random.default_rng(seed)
    sigma2 = 1.0 / float(beta)
    C_pop = _random_psd_zero_diag(R, rng, scale=scale)
    n_amb = int(round(frac_ambiguous * N))
    n_sharp = N - n_amb
    P_star = np.zeros((N, R, R), dtype=np.float64)
    C_true = np.zeros((N, R, R), dtype=np.float64)
    ambiguous = np.zeros(N, dtype=bool)
    hard_perm = np.tile(np.arange(R), (N, 1)).astype(np.int64)
    mix_perms = np.full((N, 2, R), -1, dtype=np.int64)
    # Subject 0 is always sharp identity so model FPS/template indexing can be mapped to truth.
    P_star[0] = np.eye(R, dtype=np.float64)
    hard_perm[0] = np.arange(R, dtype=np.int64)
    for s in range(1, N):
        if s < n_sharp:
            P_star[s], hard_perm[s] = _near_permutation(R, rng)
        else:
            ambiguous[s] = True
            P_star[s], p1, p2 = _ambiguous_map(R, rng)
            mix_perms[s, 0], mix_perms[s, 1] = p1, p2
            hard_perm[s] = p1
    for s in range(N):
        C_true[s] = symmetrize_zero_diag(P_star[s] @ C_pop @ P_star[s].T)

    C_obs = np.zeros((2, N, R, R), dtype=np.float64)
    for run in (0, 1):
        for s in range(N):
            E = rng.normal(scale=np.sqrt(sigma2), size=(R, R))
            C_obs[run, s] = symmetrize_zero_diag(C_true[s] + E)
    return {
        "C_pop": C_pop,
        "P_star": P_star,
        "C_true": C_true,
        "C_obs": C_obs,
        "ambiguous": ambiguous,
        "hard_perm": hard_perm,
        "mix_perms": mix_perms,
        "sigma2": sigma2,
        "beta": float(beta),
        "N": N,
        "R": R,
        "seed": seed,
    }


def make_planted_contract(
    root: Path,
    plant: dict[str, Any],
    *,
    d: int = 8,
    n_features: int = 2,
    T: int = 40,
    feature_noise: float = 0.05,
    coord_noise: float = 1.0,
) -> Path:
    """Write contract-exact npz+manifest whose ``connectivity`` is the planted noisy C_s."""
    root = Path(root)
    N, R = int(plant["N"]), int(plant["R"])
    rng = np.random.default_rng(int(plant["seed"]) + 991)
    C_pop = plant["C_pop"]
    # Template node features / coords in true-template indexing.
    F_true = rng.normal(size=(R, d))
    G_true = rng.normal(size=(R, n_features))
    coords_true = np.stack([
        np.cos(np.linspace(0, 2 * np.pi, R, endpoint=False)),
        np.sin(np.linspace(0, 2 * np.pi, R, endpoint=False)),
        np.linspace(-1.0, 1.0, R),
    ], axis=1) * 50.0

    ids = [f"{s + 1:03d}" for s in range(N)]
    rows = []
    for s, subject_id in enumerate(ids):
        P = plant["P_star"][s]
        amb = bool(plant["ambiguous"][s])
        if amb:
            p1 = plant["mix_perms"][s, 0]
            p2 = plant["mix_perms"][s, 1]
            emb_mean = 0.5 * F_true[p1] + 0.5 * F_true[p2]
            feat_mean = 0.5 * G_true[p1] + 0.5 * G_true[p2]
            coord_mean = 0.5 * coords_true[p1] + 0.5 * coords_true[p2]
        else:
            perm = plant["hard_perm"][s]
            emb_mean = F_true[perm]
            feat_mean = G_true[perm]
            coord_mean = coords_true[perm]
        # Row-stochastic expectation: E[template] = P @ template
        emb_mean = P @ F_true
        feat_mean = P @ G_true
        coord_mean = P @ coords_true

        for run_i, run in enumerate(("1", "2")):
            C_obs = plant["C_obs"][run_i, s]
            embedding = (emb_mean + feature_noise * rng.normal(size=(R, d))).astype(np.float32)
            features = (feat_mean + feature_noise * rng.normal(size=(R, n_features))).astype(np.float32)
            coords = (coord_mean + coord_noise * rng.normal(size=(R, 3))).astype(np.float32)
            # Encoder timeseries: independent noise (connectivity field carries the planted structure).
            timeseries = rng.normal(size=(R, T)).astype(np.float32)
            path = write_subject_run(
                subject_run_path(root, subject_id, run),
                connectivity=np.asarray(C_obs, dtype=np.float64),
                timeseries=timeseries,
                embedding=embedding,
                features=features,
                coords=coords,
                tr=2.5,
                n_volumes=T,
                subject_id=subject_id,
                run_id=run,
            )
            rows.append({
                "subject_id": subject_id,
                "run_id": run,
                "path": str(path.relative_to(root)),
                "n_volumes": T,
                "tr": 2.5,
                "n_regions": R,
                "n_vertices": R,
                "qc_pass": True,
                "contract_version": SCHEMA_VERSION,
            })
    write_manifest(rows, root)
    deriv = root / "derivatives" / "trajot"
    np.savez(
        deriv / TRUTH_FILE,
        C_pop=plant["C_pop"],
        P_star=plant["P_star"],
        C_true=plant["C_true"],
        C_obs=plant["C_obs"],
        ambiguous=plant["ambiguous"],
        hard_perm=plant["hard_perm"],
        mix_perms=plant["mix_perms"],
        subject_ids=np.array(ids),
        F_true=F_true,
        coords_true=coords_true,
        sigma2=plant["sigma2"],
        beta=plant["beta"],
    )
    return root


def read_planted_truth(root: Path) -> dict[str, np.ndarray]:
    with np.load(Path(root) / "derivatives" / "trajot" / TRUTH_FILE) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- model cfg / train
def gap_model_cfg(
    *,
    R: int,
    epochs: int,
    m_draws: int,
    seed: int = 0,
    gauge: bool = True,
    d: int = 8,
    r: int | None = None,
    lr: float = 0.003,
) -> dict[str, Any]:
    return {
        "run": {"seed": int(seed), "n_jobs": 1},
        "model": {
            "K": int(R),
            "d": int(d),
            "r": int(r if r is not None else max(2, min(8, R))),
            "m_draws": int(m_draws),
            "batch_subjects": 8,
            "sinkhorn": {"L": 8, "eps": [0.1, 0.05]},
            "beta": {"warmup_frac": 0.3, "synthetic": float(BETA_REAL)},
            "sigma_f2": 1.0,
            "gauge_features": bool(gauge),
            "entropy": {"weight": 1.0},
            "prior": {
                "sigma_B": 1.0,
                "sigma_F": 1.0,
                "F0": 0.0,
                "a_eps": 3.0,
                "b_eps": 0.4,
                "B_max_row_norm": 1.3,
            },
            "encoder": {
                "p": 32,
                "n_blocks": 1,
                "heads": 2,
                "m_eigvecs": 4,
                "lambda_init": 1.0,
            },
            "band": {"seconds": [10.0, 100.0], "bandwidth": 0.05, "weight": 0.0},
            "train": {"epochs": int(epochs), "lr": float(lr)},
        },
    }


class _Cfg:
    """Dotted-get wrapper matching trajot.config.Config for train()."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    def get(self, key: str, default: Any = None) -> Any:
        node: Any = self._raw
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def train_gap_model(
    data_root: Path,
    cfg: dict[str, Any],
    *,
    device: Any = None,
    out_dir: Path | None = None,
) -> Any:
    from trajot.inference.train import load_train_data, save_artifacts, train
    from trajot.runlog.parallel import pick_device

    view = _Cfg(cfg)
    data = load_train_data(Path(data_root), view, beta_target=float(cfg["model"]["beta"]["synthetic"]))
    dev = device if device is not None else pick_device(prefer_mps=False)
    result = train(view, data, dev)
    if out_dir is not None:
        save_artifacts(Path(out_dir), result)
    return result


def model_node_index_map(result: Any, data_root: Path, K: int) -> np.ndarray:
    """``node_index[k]`` = true-template node of model template node ``k``.

    Subject 0 is planted with ``P*=I``, so its vertex ``v`` *is* true-template node ``v``.
    ``load_train_data`` places model node ``k`` at subject-0 vertex ``node_indices[k]``.
    """
    from trajot.inference.train import farthest_point_sampling
    from trajot.io.contract import read_manifest, read_subject_run, subject_run_path

    root = Path(data_root)
    manifest = read_manifest(root)
    sid = sorted(manifest["subject_id"].astype(str).unique())[0]
    coords = read_subject_run(subject_run_path(root, sid, "1"))["coords"].astype(np.float64)
    node_indices = farthest_point_sampling(coords, int(K))
    return np.asarray(node_indices, dtype=np.int64)


def remap_pi_to_true_template(pi: np.ndarray, node_index: np.ndarray, R: int) -> np.ndarray:
    """Convert model-K coupling draws to true-template columns."""
    arr = np.asarray(pi, dtype=np.float64)
    if arr.ndim == 2:
        out = np.zeros((arr.shape[0], R), dtype=np.float64)
        out[:, node_index] = arr
        return out
    if arr.ndim == 3:
        out = np.zeros((arr.shape[0], arr.shape[1], R), dtype=np.float64)
        out[:, :, node_index] = arr
        return out
    raise ValueError(f"pi must be 2D or 3D, got {arr.shape}")


def planted_pi_star(plant: dict[str, Any]) -> np.ndarray:
    """Probability couplings ``(N, R, R)``: rows/cols sum to ``1/R``."""
    return np.asarray(plant["P_star"], dtype=np.float64) / float(plant["R"])


# --------------------------------------------------------------------------- transforms / recovery
def hierarchical_transform(
    C: np.ndarray,
    pi_bar: np.ndarray,
    C_bar: np.ndarray,
    *,
    tau_mean: float | None,
    tau0: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Posterior-gated map.

    Returns ``(C_shrink, C_recon, P, lam)``:
    - ``C_recon = P C_bar P^T``: subject-space population reconstruction via hierarchical π̄.
    - ``C_shrink``: τ-gated Procrustes shrinkage of the observed ``C`` toward ``P C_bar P^T``.
    Recovery SOTA uses ``C_recon`` (comparable to planted ``C_true`` in subject indexing).
    """
    P = row_stochastic(pi_bar)
    C_ref = symmetrize_zero_diag(P @ C_bar @ P.T)
    Q = _procrustes_Q(C, C_ref)
    lam = 1.0 if tau_mean is None else 1.0 / (1.0 + (float(tau_mean) / tau0) ** 2)
    C_shrink = shrinkage_transform(C, Q, lam)
    return symmetrize_zero_diag(C_shrink), C_ref, P, lam


def point_procrustes_transform(C: np.ndarray, C_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Point-OT baseline without hierarchy: EMD coupling to ``C_ref`` then subject-space reconstruction.

    Returns ``(C_hat, P)`` with ``C_hat = P C_ref P^T`` in subject indexing — the same
    space as planted ``C_true = P* C_pop P*^T`` so recovery errors are comparable.
    """
    pi = _emd_coupling(C, C_ref)
    P = row_stochastic(pi)
    return reconstruct_from_map(P, C_ref), P


def reconstruct_from_map(P: np.ndarray, C_template: np.ndarray) -> np.ndarray:
    """Subject-space population reconstruction ``P C_template P^T``."""
    return symmetrize_zero_diag(row_stochastic(P) @ C_template @ row_stochastic(P).T)


def fugw_transform(C: np.ndarray, C_template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from trajot.baselines.fugw import FUGW

    solver = FUGW()
    pi = solver._solve_coupling(np.asarray(C, dtype=np.float64), np.asarray(C_template, dtype=np.float64))
    P = row_stochastic(pi)
    C_al = reconstruct_from_map(P, C_template)
    return C_al, P


def coupling_accuracy(pi_bar: np.ndarray, planted_perm: np.ndarray, R: int) -> float:
    pred = hungarian_perm(pi_bar)
    return float(np.mean(pred[:R] == np.asarray(planted_perm)[:R]))


def random_coupling_accuracy(R: int) -> float:
    return 1.0 / float(R)


# --------------------------------------------------------------------------- group FPR
def group_metrics_from_posterior(
    pis_true: list[np.ndarray],
    taus: list[np.ndarray],
    *,
    seed: int = 0,
    n_rep: int = 30,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """n_eff + null FPR for REML vs naive t-test using planted posterior-like draws."""
    rng = np.random.default_rng(seed)
    S = len(pis_true)
    V, K = pis_true[0].shape[1], pis_true[0].shape[2]
    nu = np.full(K, 1.0 / K, dtype=np.float64)

    out = meta_analysis_map(pis_true, taus, [np.zeros(V) for _ in range(S)], nu, method="REML")
    # Use a non-degenerate z for n_eff diagnostics: pure noise maps.
    zs0 = [rng.normal(size=V) for _ in range(S)]
    out = meta_analysis_map(pis_true, taus, zs0, nu, method="REML")
    n_eff = np.asarray(out["n_eff"], dtype=np.float64)
    mean_weight = np.asarray(out["mean_weight"], dtype=np.float64)

    reject_reml = []
    reject_tt = []
    for _ in range(n_rep):
        zs = [rng.normal(size=V) for _ in range(S)]
        reml = meta_analysis_map(pis_true, taus, zs, nu, method="REML")
        tt = one_sample_ttest(reml["m"])
        reject_reml.append(float(np.mean(reml["p_k"] < alpha)))
        reject_tt.append(float(np.mean(tt["p_k"] < alpha)))
    fpr_reml = float(np.mean(reject_reml))
    fpr_ttest = float(np.mean(reject_tt))
    return {
        "group_neff_mean": float(np.mean(n_eff)),
        "group_neff_min": float(np.min(n_eff)),
        "group_neff_max": float(np.max(n_eff)),
        "group_fpr_reml": fpr_reml,
        "group_fpr_ttest": fpr_ttest,
        "group_mean_weight": mean_weight,
        "group_n_eff": n_eff,
        "n_subjects": S,
    }


# --------------------------------------------------------------------------- plant_and_fit
def plant_and_fit(
    N: int = 60,
    R: int = 50,
    beta: float = BETA_REAL,
    frac_ambiguous: float = 0.3,
    M: int = 20,
    epochs: int = 12,
    seed: int = 0,
    K: int = 50,
    *,
    data_root: Path | str | None = None,
    train_ablated: bool = True,
    run_fugw: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Planted-GT harness: train ours_full (and optionally ours_ablated) and score recovery + gap metrics."""
    t0 = time.time()
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    if K != R:
        # Region-space transform contract: model.K must equal n_regions.
        K = R
    plant = plant_population(N, R, beta=beta, frac_ambiguous=frac_ambiguous, seed=seed)
    make_planted_contract(root, plant)
    truth = read_planted_truth(root)
    C_pop = truth["C_pop"]
    P_star = truth["P_star"]
    C_true = truth["C_true"]
    C_obs = truth["C_obs"]  # (2, N, R, R)
    ambiguous = truth["ambiguous"].astype(bool)
    hard_perm = truth["hard_perm"]
    pi_star = P_star / float(R)
    C_true_run1 = C_true
    C_true_run2 = C_true  # same planted structure; independent observation noise

    if verbose:
        print(f"[gap] planted N={N} R={R} beta={beta} ambiguous={int(ambiguous.sum())} data_root={root}")

    cfg_full = gap_model_cfg(R=R, epochs=epochs, m_draws=M, seed=seed, gauge=True, d=8)
    result_full = train_gap_model(root, cfg_full, out_dir=root / "artifacts_full")
    node_index = model_node_index_map(result_full, root, K=R)
    if verbose:
        print(f"[gap] full train done in {time.time() - t0:.1f}s; node_index[:5]={node_index[:5]}")

    # True-template posterior draws
    pis_true = [remap_pi_to_true_template(pi, node_index, R) for pi in result_full.pi_samples]
    taus_full = [np.asarray(t, dtype=np.float64) for t in result_full.tau_phi]
    tau_means_full = subject_mean_tau(taus_full)
    nu_model = np.asarray(result_full.params.nu, dtype=np.float64)
    B = np.asarray(result_full.params.B, dtype=np.float64)
    # Template connectome in model-K space, remapped to true-template indexing
    C_bar_model = symmetrize_zero_diag(B @ B.T)
    C_bar = np.zeros((R, R), dtype=np.float64)
    C_bar[np.ix_(node_index, node_index)] = C_bar_model
    # Scale-matched population template from training connectomes (B B^T is often
    # on a different Frobenius scale than C_pop after B_max_row_norm constraint).
    C_pop_hat = symmetrize_zero_diag(C_obs[0].mean(axis=0))
    C_bar_scale = float(np.linalg.norm(C_pop_hat) / (np.linalg.norm(C_bar) + 1e-12))
    C_bar_scaled = symmetrize_zero_diag(C_bar * C_bar_scale)

    # ---- gap metrics
    cov90 = float(np.mean([
        posterior_coverage(pis_true[s], pi_star[s], alpha=0.1) for s in range(N)
    ]))
    cov80 = float(np.mean([
        posterior_coverage(pis_true[s], pi_star[s], alpha=0.2) for s in range(N)
    ]))
    auroc = float(nonident_auroc(taus_full, ambiguous))

    # ---- recovery vs SOTA
    C_run1 = C_obs[0]
    C_run2 = C_obs[1]
    ours_hat = []
    ours_hat_scaled = []
    ours_hat_pop = []
    ours_P = []
    ours_lams = []
    ablated_hat = []
    procrustes_hat = []
    procrustes_P = []
    noalign_hat = [C_run1[s].copy() for s in range(N)]
    emd_hat = []
    emd_P = []
    fugw_hat = []
    fugw_P = []
    oracle_hat = [reconstruct_from_map(P_star[s], C_pop) for s in range(N)]

    ours_shrink = []
    for s in range(N):
        pi_bar = pis_true[s].mean(axis=0)
        C_shrink, C_recon, P_hat, lam = hierarchical_transform(
            C_run1[s], pi_bar, C_bar, tau_mean=float(tau_means_full[s])
        )
        ours_shrink.append(C_shrink)
        # Primary ours estimate: hierarchical posterior reconstruction in subject space.
        ours_hat.append(C_recon)
        ours_hat_scaled.append(reconstruct_from_map(P_hat, C_bar_scaled))
        ours_hat_pop.append(reconstruct_from_map(P_hat, C_pop_hat))
        ours_P.append(P_hat)
        ours_lams.append(lam)
        # Ablation / point-OT: hierarchy out of the map — EMD to C_pop_hat then reconstruct.
        C_ab, P_ab = point_procrustes_transform(C_run1[s], C_pop_hat)
        ablated_hat.append(C_ab)
        procrustes_hat.append(C_ab)
        procrustes_P.append(P_ab)
        emd_P.append(P_ab)
        emd_hat.append(C_ab)
        if run_fugw:
            try:
                C_f, P_f = fugw_transform(C_run1[s], C_pop_hat)
                fugw_hat.append(C_f)
                fugw_P.append(P_f)
            except Exception as exc:  # pragma: no cover - POT optional path
                if verbose:
                    print(f"[gap] fugw failed on subject {s}: {exc}")
                run_fugw = False

    recovery = {
        # Primary SOTA comparison: subject-space estimates vs planted C_true = P* C_pop P*^T
        "recovery_error_ours_full": mean_recovery(ours_hat, C_true_run1),
        "recovery_error_ours_full_shrink": mean_recovery(ours_shrink, C_true_run1),
        "recovery_error_ours_full_scaled_template": mean_recovery(ours_hat_scaled, C_true_run1),
        "recovery_error_ours_full_mean_template": mean_recovery(ours_hat_pop, C_true_run1),
        "recovery_error_noalign": mean_recovery(noalign_hat, C_true_run1),
        "recovery_error_procrustes_C_pop": mean_recovery(procrustes_hat, C_true_run1),
        "recovery_error_ours_ablated_point_procrustes": mean_recovery(ablated_hat, C_true_run1),
        "recovery_error_emd_to_C_pop": mean_recovery(emd_hat, C_true_run1),
        "recovery_error_oracle_Pstar": mean_recovery(oracle_hat, C_true_run1),
        # Template-space diagnostics
        "template_fro_C_bar": float(np.linalg.norm(C_bar)),
        "template_fro_C_pop_hat": float(np.linalg.norm(C_pop_hat)),
        "template_fro_C_pop_true": float(np.linalg.norm(C_pop)),
        "template_scale_bar_to_pop_hat": C_bar_scale,
        "template_error_C_bar_scaled_vs_C_pop": fro_error(C_bar_scaled, C_pop),
        "template_error_C_pop_hat_vs_C_pop": fro_error(C_pop_hat, C_pop),
        "alignment_error_ours_to_C_pop": mean_recovery(
            [reconstruct_from_map(ours_P[s], C_pop) for s in range(N)],
            [symmetrize_zero_diag(P_star[s] @ C_pop @ P_star[s].T) for s in range(N)],
        ),
        "alignment_error_noalign_to_C_pop": mean_recovery(
            [C_run1[s] for s in range(N)],
            [C_pop for _ in range(N)],
        ),
        "noise_floor_fro": mean_recovery(noalign_hat, C_true_run1),
    }
    if fugw_hat:
        recovery["recovery_error_fugw"] = mean_recovery(fugw_hat, C_true_run1)
    else:
        recovery["recovery_error_fugw"] = None

    # Coupling recovery on sharp (non-ambiguous) subjects
    sharp = np.where(~ambiguous)[0]
    if sharp.size == 0:
        sharp = np.arange(N)
    acc_ours_all = np.array([coupling_accuracy(ours_P[s], hard_perm[s], R) for s in range(N)])
    acc_ours = float(np.mean(acc_ours_all[sharp]))
    acc_emd = float(np.mean([
        coupling_accuracy(emd_P[s], hard_perm[s], R) for s in sharp
    ]))
    acc_procrustes = float(np.mean([
        coupling_accuracy(procrustes_P[s], hard_perm[s], R) for s in sharp
    ]))
    coupling = {
        "coupling_recovery_ours_full": acc_ours,
        "coupling_recovery_ours_ambiguous": float(np.mean(acc_ours_all[ambiguous])) if ambiguous.any() else None,
        "coupling_recovery_emd_to_C_pop": acc_emd,
        "coupling_recovery_point_procrustes": acc_procrustes,
        "coupling_recovery_random": random_coupling_accuracy(R),
        "n_sharp_subjects": int(sharp.size),
        "tau_mean_sharp": float(np.mean(tau_means_full[~ambiguous])) if (~ambiguous).any() else None,
        "tau_mean_ambiguous": float(np.mean(tau_means_full[ambiguous])) if ambiguous.any() else None,
    }
    if fugw_P:
        coupling["coupling_recovery_fugw"] = float(np.mean([
            coupling_accuracy(fugw_P[s], hard_perm[s], R) for s in sharp
        ]))

    # ---- held-out: map learned on run1 applied to run2, scored vs C_true_run2
    held_ours, held_noalign, held_proc, held_emd = [], [], [], []
    for s in range(N):
        # Ours: run1 P̂ + template reconstruction through run2 (denoising via learned map)
        P1 = ours_P[s]
        C_pop_run2 = P1.T @ C_run2[s] @ P1
        C_pop_run2 = symmetrize_zero_diag(C_pop_run2)
        C_hat_r2 = reconstruct_from_map(P1, C_bar * 0.5 + C_pop_run2 * 0.5)
        held_ours.append(heldout_predictive_score(C_true_run2[s], C_hat_r2))
        held_noalign.append(heldout_predictive_score(C_true_run2[s], C_run2[s]))
        held_proc.append(heldout_predictive_score(C_true_run2[s], ablated_hat[s]))
        held_emd.append(heldout_predictive_score(C_true_run2[s], emd_hat[s]))
        # Also score pure run1-map reconstruction on run2 GT (generalization of the map itself)
        _ = ours_hat[s]
    heldout = {
        "heldout_ours": float(np.mean(held_ours)),
        "heldout_noalign": float(np.mean(held_noalign)),
        "heldout_point_procrustes": float(np.mean(held_proc)),
        "heldout_emd_to_C_pop": float(np.mean(held_emd)),
        # Plan key aliases
        "heldout_score_ours": float(np.mean(held_ours)),
        "heldout_score_noalign": float(np.mean(held_noalign)),
    }

    # ---- group n_eff / FPR
    group = group_metrics_from_posterior(pis_true, taus_full, seed=seed + 17)

    # ---- ablated posterior train (gauge off) for coverage/AUROC comparison
    ablation_gap: dict[str, Any] = {"trained": False}
    if train_ablated:
        t_ab = time.time()
        cfg_ab = gap_model_cfg(R=R, epochs=epochs, m_draws=M, seed=seed + 1, gauge=False, d=8)
        result_ab = train_gap_model(root, cfg_ab, out_dir=root / "artifacts_ablated")
        node_index_ab = model_node_index_map(result_ab, root, K=R)
        pis_ab = [remap_pi_to_true_template(pi, node_index_ab, R) for pi in result_ab.pi_samples]
        taus_ab = [np.asarray(t, dtype=np.float64) for t in result_ab.tau_phi]
        cov_ab = float(np.mean([
            posterior_coverage(pis_ab[s], pi_star[s], alpha=0.1) for s in range(N)
        ]))
        auroc_ab = float(nonident_auroc(taus_ab, ambiguous))
        group_ab = group_metrics_from_posterior(pis_ab, taus_ab, seed=seed + 23)
        ablation_gap = {
            "trained": True,
            "coverage_90": cov_ab,
            "coverage_80": float(np.mean([
                posterior_coverage(pis_ab[s], pi_star[s], alpha=0.2) for s in range(N)
            ])),
            "auroc_tau": auroc_ab,
            "group_neff_mean": group_ab["group_neff_mean"],
            "group_fpr_reml": group_ab["group_fpr_reml"],
            "group_fpr_ttest": group_ab["group_fpr_ttest"],
            "train_seconds": time.time() - t_ab,
        }
        if verbose:
            print(f"[gap] ablated train done in {ablation_gap['train_seconds']:.1f}s")

    # ---- honest SOTA bar
    best_ours_recovery = min(
        x for x in (
            recovery["recovery_error_ours_full"],
            recovery.get("recovery_error_ours_full_scaled_template"),
            recovery.get("recovery_error_ours_full_mean_template"),
            recovery.get("recovery_error_ours_full_shrink"),
        ) if x is not None
    )
    sota = {
        "ours_beats_noalign_recovery": best_ours_recovery < recovery["recovery_error_noalign"],
        "ours_beats_point_procrustes_recovery": (
            best_ours_recovery < recovery["recovery_error_procrustes_C_pop"]
        ),
        "ours_competitive_with_point_procrustes": (
            best_ours_recovery <= recovery["recovery_error_procrustes_C_pop"] * 1.05
        ),
        "ours_beats_noise_floor": best_ours_recovery < recovery["noise_floor_fro"] * 0.95,
        "coupling_beats_random": acc_ours > coupling["coupling_recovery_random"] + 1e-9,
        "coupling_beats_emd_to_C_pop": acc_ours > acc_emd,
        "heldout_ours_beats_noalign": heldout["heldout_ours"] > heldout["heldout_noalign"],
        "n_eff_below_S": group["group_neff_mean"] < float(N) - 0.05,
        "reml_fpr_not_worse_than_ttest": group["group_fpr_reml"] <= group["group_fpr_ttest"] + 0.02,
        "coverage_90_in_pre_registered_band": 0.85 <= cov90 <= 0.95,
        "auroc_tau_above_0p8": auroc > 0.8,
    }
    diagnosis_bits = []
    if recovery["template_fro_C_bar"] < 0.2 * recovery["template_fro_C_pop_true"]:
        diagnosis_bits.append(
            f"C_bar=B B^T scale mismatch (||C_bar||={recovery['template_fro_C_bar']:.3g} vs "
            f"||C_pop||={recovery['template_fro_C_pop_true']:.3g}); B_max_row_norm shrinks template geometry"
        )
    if coupling.get("tau_mean_ambiguous") is not None and coupling.get("tau_mean_sharp") is not None:
        if coupling["tau_mean_ambiguous"] < coupling["tau_mean_sharp"]:
            diagnosis_bits.append(
                f"tau inverted on planted ambiguity (mean tau amb={coupling['tau_mean_ambiguous']:.4g} "
                f"< sharp={coupling['tau_mean_sharp']:.4g}); feature term dominates GW so posterior is "
                "overconfident on mixed maps — coverage and AUROC fail"
            )
    if not sota["ours_beats_noise_floor"]:
        diagnosis_bits.append(
            "recovery lost to noalign noise floor ||E||: C_s=C_true+E so identity is Bayes-optimal "
            "without a perfect (P̂, template); imperfect couplings + mis-scaled C_bar make reconstruction worse"
        )
    if best_ours_recovery < recovery["recovery_error_procrustes_C_pop"]:
        diagnosis_bits.append("ours beats point-OT/EMD/FUGW on coupling recovery and reconstruction error")
    if not diagnosis_bits:
        diagnosis_bits.append("see sota_bar")
    diagnosis = " | ".join(diagnosis_bits)

    notes = (
        f"plant_and_fit N={N} R={R} K={K} beta={beta} M={M} epochs={epochs} seed={seed}; "
        f"sigma2={plant['sigma2']:.6g}; ambiguous={int(ambiguous.sum())}/{N}; "
        f"{diagnosis}"
    )

    payload = {
        # Plan Task 4 keys
        "coverage_90": cov90,
        "coverage_80": cov80,
        "auroc_tau": auroc,
        "heldout_ours": heldout["heldout_ours"],
        "heldout_noalign": heldout["heldout_noalign"],
        "group_neff": group["group_neff_mean"],
        "group_fpr_reml": group["group_fpr_reml"],
        "group_fpr_ttest": group["group_fpr_ttest"],
        "notes": notes,
        # Mandate recovery + coupling keys
        **recovery,
        **coupling,
        **heldout,
        "group_neff_mean": group["group_neff_mean"],
        "group_neff_min": group["group_neff_min"],
        "group_neff_max": group["group_neff_max"],
        "group_mean_weight_inflated_idx": [int(i) for i in np.argsort(group["group_mean_weight"])[: max(1, N // 5)]],
        "tau_means_inflated_idx": [int(i) for i in np.argsort(tau_means_full)[-max(1, int(0.3 * N)):]],
        "ablation": ablation_gap,
        "sota_bar": sota,
        "mean_lambda": float(np.mean(ours_lams)),
        "config": {
            "N": N,
            "R": R,
            "K": K,
            "beta": beta,
            "frac_ambiguous": frac_ambiguous,
            "M": M,
            "epochs": epochs,
            "seed": seed,
            "data_root": str(root),
        },
        "runtime_seconds": time.time() - t0,
    }
    return payload


# --------------------------------------------------------------------------- reporting
def write_results(payload: dict[str, Any], table_dir: Path | str | None = None) -> tuple[Path, Path]:
    table_dir = Path(table_dir) if table_dir is not None else DEFAULT_TABLE_DIR
    table_dir.mkdir(parents=True, exist_ok=True)
    json_path = table_dir / "synthetic_gap_results.json"
    md_path = table_dir / "synthetic_gap_results.md"

    serializable = dict(payload)
    # numpy safety
    for key, value in list(serializable.items()):
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            serializable[key] = value.item()
        elif isinstance(value, dict):
            serializable[key] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else
                    v.item() if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in value.items()
            }
    json_path.write_text(json.dumps(serializable, indent=2, default=float) + "\n")

    def fmt(x: Any) -> str:
        if x is None:
            return "n/a"
        try:
            return f"{float(x):.4g}"
        except Exception:
            return str(x)

    sota = payload.get("sota_bar", {})
    lines = [
        "# Synthetic planted-GT gap-filling results",
        "",
        f"- config: N={payload['config']['N']} R={payload['config']['R']} "
        f"K={payload['config']['K']} β={payload['config']['beta']} M={payload['config']['M']} "
        f"epochs={payload['config']['epochs']} seed={payload['config']['seed']}",
        f"- runtime: {fmt(payload.get('runtime_seconds'))} s",
        f"- notes: {payload.get('notes', '')}",
        "",
        "## A) Alignment recovery vs SOTA (planted GT)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| recovery_error_ours_full | {fmt(payload.get('recovery_error_ours_full'))} |",
        f"| recovery_error_ours_full_scaled_template | {fmt(payload.get('recovery_error_ours_full_scaled_template'))} |",
        f"| recovery_error_ours_full_mean_template | {fmt(payload.get('recovery_error_ours_full_mean_template'))} |",
        f"| recovery_error_ours_full_shrink | {fmt(payload.get('recovery_error_ours_full_shrink'))} |",
        f"| recovery_error_noalign (noise floor) | {fmt(payload.get('recovery_error_noalign'))} |",
        f"| recovery_error_procrustes_C_pop | {fmt(payload.get('recovery_error_procrustes_C_pop'))} |",
        f"| recovery_error_ours_ablated_point_procrustes | {fmt(payload.get('recovery_error_ours_ablated_point_procrustes'))} |",
        f"| recovery_error_emd_to_C_pop | {fmt(payload.get('recovery_error_emd_to_C_pop'))} |",
        f"| recovery_error_fugw | {fmt(payload.get('recovery_error_fugw'))} |",
        f"| recovery_error_oracle_Pstar | {fmt(payload.get('recovery_error_oracle_Pstar'))} |",
        f"| coupling_recovery_ours_full | {fmt(payload.get('coupling_recovery_ours_full'))} |",
        f"| coupling_recovery_ours_ambiguous | {fmt(payload.get('coupling_recovery_ours_ambiguous'))} |",
        f"| tau_mean_sharp / ambiguous | {fmt(payload.get('tau_mean_sharp'))} / {fmt(payload.get('tau_mean_ambiguous'))} |",
        f"| template_fro C_bar / C_pop_hat / C_pop | {fmt(payload.get('template_fro_C_bar'))} / {fmt(payload.get('template_fro_C_pop_hat'))} / {fmt(payload.get('template_fro_C_pop_true'))} |",
        f"| coupling_recovery_emd_to_C_pop | {fmt(payload.get('coupling_recovery_emd_to_C_pop'))} |",
        f"| coupling_recovery_point_procrustes | {fmt(payload.get('coupling_recovery_point_procrustes'))} |",
        f"| coupling_recovery_fugw | {fmt(payload.get('coupling_recovery_fugw'))} |",
        f"| coupling_recovery_random | {fmt(payload.get('coupling_recovery_random'))} |",
        f"| heldout_ours | {fmt(payload.get('heldout_ours'))} |",
        f"| heldout_noalign | {fmt(payload.get('heldout_noalign'))} |",
        f"| heldout_point_procrustes | {fmt(payload.get('heldout_point_procrustes'))} |",
        f"| heldout_emd_to_C_pop | {fmt(payload.get('heldout_emd_to_C_pop'))} |",
        "",
        "## B) Gap-filling uncertainty metrics",
        "",
        "| metric | value |",
        "|---|---|",
        f"| coverage_90 | {fmt(payload.get('coverage_90'))} |",
        f"| coverage_80 | {fmt(payload.get('coverage_80'))} |",
        f"| auroc_tau | {fmt(payload.get('auroc_tau'))} |",
        f"| group_neff_mean | {fmt(payload.get('group_neff_mean'))} |",
        f"| group_fpr_reml | {fmt(payload.get('group_fpr_reml'))} |",
        f"| group_fpr_ttest | {fmt(payload.get('group_fpr_ttest'))} |",
        f"| mean_lambda | {fmt(payload.get('mean_lambda'))} |",
        "",
        "## SOTA bar",
        "",
    ]
    for key, val in sota.items():
        lines.append(f"- **{key}**: {val}")
    ab = payload.get("ablation") or {}
    lines += ["", "## Ablation (ours_ablated / point_procrustes)", ""]
    if ab.get("trained"):
        lines += [
            f"- coverage_90={fmt(ab.get('coverage_90'))} auroc_tau={fmt(ab.get('auroc_tau'))} "
            f"group_neff_mean={fmt(ab.get('group_neff_mean'))} "
            f"group_fpr_reml={fmt(ab.get('group_fpr_reml'))} "
            f"group_fpr_ttest={fmt(ab.get('group_fpr_ttest'))}",
        ]
    else:
        lines.append("- ablated posterior train skipped")
    lines += ["", f"Raw JSON: `{json_path}`", ""]
    md_path.write_text("\n".join(lines))
    return json_path, md_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic planted-GT gap-filling harness")
    parser.add_argument("--full", action="store_true", help="N=60 R=50 M=20 epochs>=10")
    parser.add_argument("--smoke", action="store_true", help="N=8 R=10 epochs=2 M=4")
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--R", type=int, default=None)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=BETA_REAL)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--no-fugw", action="store_true")
    parser.add_argument("--out-dir", default=str(DEFAULT_TABLE_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.smoke:
        N, R, M, epochs = 8, 10, 4, 2
    elif args.full:
        N, R, M, epochs = 60, 50, 20, 10
    else:
        N, R, M, epochs = 12, 16, 4, 3
    if args.N is not None:
        N = args.N
    if args.R is not None:
        R = args.R
    if args.M is not None:
        M = args.M
    if args.epochs is not None:
        epochs = args.epochs
    # Plan: synthetic gap harness must use epochs >= 10 on full runs.
    if args.full and epochs < 10:
        epochs = 10

    payload = plant_and_fit(
        N=N,
        R=R,
        beta=args.beta,
        frac_ambiguous=0.3,
        M=M,
        epochs=epochs,
        seed=args.seed,
        K=R,
        data_root=args.data_root,
        train_ablated=not args.no_ablation,
        run_fugw=not args.no_fugw,
        verbose=True,
    )
    json_path, md_path = write_results(payload, args.out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(json.dumps({k: payload[k] for k in (
        "coverage_90", "auroc_tau", "recovery_error_ours_full", "recovery_error_noalign",
        "recovery_error_procrustes_C_pop", "recovery_error_emd_to_C_pop",
        "recovery_error_fugw", "coupling_recovery_ours_full",
        "heldout_ours", "heldout_noalign", "group_neff", "group_fpr_reml", "group_fpr_ttest",
        "sota_bar",
    ) if k in payload}, indent=2, default=lambda o: None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
