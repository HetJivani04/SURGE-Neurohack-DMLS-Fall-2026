from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .base import Baseline, CfgView, cfg_get, register_baseline, symmetrize_zero_diag


def _spectral_rows(A: np.ndarray, d: int) -> np.ndarray:
    """Row features in a shared d-dim spectral space (top-|λ| eigensystem or SVD)."""
    A = np.asarray(A, dtype=np.float64)
    if A.shape[1] <= d:
        return A
    if A.shape[0] == A.shape[1]:
        vals, vecs = np.linalg.eigh(0.5 * (A + A.T))
        order = np.argsort(np.abs(vals))[::-1][:d]
        return np.ascontiguousarray(vecs[:, order] * np.sqrt(np.abs(vals[order])))
    u, s, _ = np.linalg.svd(A, full_matrices=False)
    return np.ascontiguousarray(u[:, :d] * s[:d])


def _cdist_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Euclidean distances between row profiles of ``A`` and ``B`` → ``(n_A, n_B)``.

    When profile widths differ (subject ``C`` is ``R×R``, template ``C_bar`` is ``K×K``),
    both sides are projected into a shared spectral feature space so the cost matrix is
    naturally rectangular ``(R, K)`` rather than requiring ``R == K``.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[1] != B.shape[1]:
        d = min(A.shape[0], B.shape[0], A.shape[1], B.shape[1])
        A = _spectral_rows(A, d)
        B = _spectral_rows(B, d)
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    return np.sqrt(np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0))


def _template_connectome(B: np.ndarray) -> np.ndarray:
    """Template geometry ``C_bar = B B^T`` from learned loadings ``(K, r)``."""
    Bf = np.asarray(B, dtype=np.float64)
    return symmetrize_zero_diag(Bf @ Bf.T)


def _soft_coupling(C: np.ndarray, C_bar: np.ndarray) -> np.ndarray:
    """Region-to-template soft coupling from connectivity profiles against learned template."""
    import ot

    Rs, Rd = C.shape[0], C_bar.shape[0]
    p = np.full(Rs, 1.0 / Rs, dtype=np.float64)
    q = np.full(Rd, 1.0 / Rd, dtype=np.float64)
    M = _cdist_rows(C, C_bar)
    return np.ascontiguousarray(ot.emd(p, q, M), dtype=np.float64)


def _orthogonal_from_coupling(C: np.ndarray, C_bar: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Spectrum-preserving orthogonal map ``Q (R,R)`` derived from the template coupling.

    Barycentric projection of the template into subject index space is ``C_ref = P C_bar P^T``
    with ``P`` the row-stochastic coupling. ``Q`` is the orthogonal Procrustes solution of
    ``argmin ||Q C - C_ref||_F`` (equiv. ``UV^T`` from ``SVD(C C_ref^T)``). Applying ``Q^T C Q``
    reindexes the subject connectome in template coordinates **without collapsing subject
    eigenvalues** toward the group mean — unlike a mass-weighted barycentric push, which
    destroys the between-subject spectrum that identification depends on.

    ``pi`` may be rectangular ``(R, K)``; ``C_bar`` is then ``(K, K)`` and ``C_ref`` is ``(R, R)``.
    """
    C = np.asarray(C, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    C_bar = np.asarray(C_bar, dtype=np.float64)
    R, Rd = pi.shape
    if C_bar.shape[0] != Rd:
        raise ValueError(f"C_bar {C_bar.shape} incompatible with coupling {pi.shape}")
    row = np.maximum(pi.sum(axis=1, keepdims=True), 1e-12)
    P = pi / row  # (R, Rd) barycentric weights
    C_ref = P @ C_bar @ P.T
    C_ref = symmetrize_zero_diag(C_ref)
    M = C @ C_ref.T
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return np.ascontiguousarray(U @ Vt, dtype=np.float64)


def pool_pi_to_regions(
    pi: np.ndarray,
    region_index: np.ndarray | None,
    R: int,
) -> np.ndarray:
    """Mass-preserving vertex→region pooling of a coupling to ``(R, K)``.

    ``(R, K)`` input is returned as a copy. ``(V, K)`` input is block-summed over
    ``region_index[v]`` so each region row carries the total posterior mass of its
    vertices (mass-preserving, not an average).
    """
    pi = np.asarray(pi, dtype=np.float64)
    if pi.ndim != 2:
        raise ValueError(f"pi must be 2D (V,K) or (R,K), found shape {pi.shape}")
    R = int(R)
    if R <= 0:
        raise ValueError(f"R must be positive, got {R}")
    if pi.shape[0] == R:
        return np.ascontiguousarray(pi.copy(), dtype=np.float64)
    if region_index is None:
        raise ValueError(
            f"pi rows {pi.shape[0]} != R={R}; region_index is required to pool vertices to regions"
        )
    region_index = np.asarray(region_index).ravel()
    if region_index.shape[0] != pi.shape[0]:
        raise ValueError(
            f"region_index length {region_index.shape[0]} != pi rows {pi.shape[0]}"
        )
    if region_index.size and (region_index.min() < 0 or region_index.max() >= R):
        raise ValueError(
            f"region_index values must lie in [0, {R}), got [{region_index.min()}, {region_index.max()}]"
        )
    out = np.zeros((R, pi.shape[1]), dtype=np.float64)
    np.add.at(out, region_index.astype(np.int64), pi)
    return out


def mix_identity_orthogonal(Q: np.ndarray, lam: float) -> np.ndarray:
    """Re-orthogonalized interpolation on O(R) between ``I`` and ``Q``.

    Forms ``M = (1-λ) I + λ Q`` and returns ``UV^T`` from ``SVD(M)``.
    λ=0 → identity (up to SVD sign); λ=1 → ``Q`` when ``Q`` is orthogonal.
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be square, found {Q.shape}")
    lam = float(lam)
    R = Q.shape[0]
    M = (1.0 - lam) * np.eye(R, dtype=np.float64) + lam * Q
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return np.ascontiguousarray(U @ Vt, dtype=np.float64)


def _local_shrinkage(C: np.ndarray, Q: np.ndarray, lam: float) -> np.ndarray:
    """Local fallback for Task-1 ``shrinkage_transform``: ``(1-λ) C + λ Q^T C Q``."""
    C = np.asarray(C, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    return (1.0 - lam) * C + lam * (Q.T @ C @ Q)


def _apply_shrinkage(C: np.ndarray, Q: np.ndarray, lam: float) -> np.ndarray:
    """C̃ = (1-λ)C + λ Q^T C Q via Task-1 ``shrinkage_transform`` when importable."""
    try:
        from trajot.eval.uncertainty import shrinkage_transform

        return shrinkage_transform(C, Q, lam)
    except Exception:
        return _local_shrinkage(C, Q, lam)


def _lambda_from_tau(mean_tau: float | None, tau0: float) -> float:
    """τ-gated alignment strength: ``λ = 1 / (1 + (τ/τ0)²)``; missing τ → full map (λ=1)."""
    if mean_tau is None:
        return 1.0
    tau0 = float(tau0)
    if tau0 <= 0:
        raise ValueError(f"tau0 must be positive, got {tau0}")
    return 1.0 / (1.0 + (float(mean_tau) / tau0) ** 2)


def subject_tau_means(tau_phi: Sequence[Any] | None) -> list[float | None]:
    """Per-subject mean τ (or None when missing)."""
    if not tau_phi:
        return []
    out: list[float | None] = []
    for t in tau_phi:
        if t is None:
            out.append(None)
            continue
        arr = np.asarray(t, dtype=np.float64)
        out.append(float(np.mean(arr)) if arr.size else None)
    return out


def calibrated_tau0(
    tau_phi: Sequence[Any] | None,
    fixed_tau0: float,
    *,
    auto: bool = True,
    ratio_trigger: float = 10.0,
) -> float:
    """Scale τ0 to the empirical subject-τ median when the fixed value cannot discriminate.

    Adaptive λ(τ) is the hierarchy-in-the-map mechanism. If fixed ``tau0`` swamps
    (or is swamped by) trained τ, every subject shares λ and the posterior stops
    driving alignment strength. When ``auto`` and ``fixed_tau0 / median(τ)`` is
    outside ``[1/ratio_trigger, ratio_trigger]``, return ``max(median(τ), ε)``.
    """
    fixed = float(fixed_tau0)
    if not auto or not tau_phi:
        return fixed
    means = [m for m in subject_tau_means(tau_phi) if m is not None]
    if not means:
        return fixed
    med = float(np.median(means))
    if med <= 0:
        return fixed
    ratio = fixed / med
    if ratio > float(ratio_trigger) or ratio < 1.0 / float(ratio_trigger):
        return max(med, 1e-12)
    return fixed


def shrink_pi_hierarchical(
    pi_bars: list[np.ndarray],
    tau_means: Sequence[float | None],
    *,
    kappa0: float = 1.0,
) -> list[np.ndarray]:
    """Hierarchical shrinkage of subject couplings toward the population mean.

    ``π_s^hier = (w_s π̄_s + w_pop π̄_pop) / (w_s + w_pop)`` with
    ``w_s = 1/(τ_s²+ε)`` and ``w_pop = κ0/(τ̄²+ε)``.

    Low-τ (determined) subjects keep their own posterior coupling — the
    posterior drives their map. High-τ subjects borrow the population coupling.
    Point baselines (BrainSync / FUGW / ULOT) cannot perform this step.
    """
    if not pi_bars:
        return []
    stacked = np.stack([np.asarray(p, dtype=np.float64) for p in pi_bars], axis=0)
    pi_pop = stacked.mean(axis=0)
    taus = list(tau_means) if tau_means else [None] * len(pi_bars)
    if len(taus) < len(pi_bars):
        taus = taus + [None] * (len(pi_bars) - len(taus))
    valid = [t for t in taus if t is not None and np.isfinite(t)]
    tau_bar = float(np.mean(valid)) if valid else 0.0
    eps = 1e-12
    out: list[np.ndarray] = []
    for s, pi in enumerate(pi_bars):
        pi = np.asarray(pi, dtype=np.float64)
        t_s = taus[s]
        if t_s is None:
            out.append(pi.copy())
            continue
        w_s = 1.0 / (float(t_s) ** 2 + eps)
        w_pop = float(kappa0) / (tau_bar**2 + eps)
        out.append((w_s * pi + w_pop * pi_pop) / (w_s + w_pop))
    return out


def _discover_data_roots(extra: Mapping[str, Any], run_dir: Path | None) -> list[Path]:
    """Candidate data roots for Schaefer region labels when extra['data_root'] is missing."""
    roots: list[Path] = []
    for key in ("data_root", "data.root"):
        val = extra.get(key)
        if val:
            roots.append(Path(str(val)))
    if run_dir is not None:
        # metrics / beta sidecars sometimes sit beside artifacts
        for cand in (run_dir, Path(run_dir).parent):
            for name in ("paths.yaml", "metrics.json"):
                p = Path(cand) / name
                if not p.is_file():
                    continue
                try:
                    text = p.read_text()
                except Exception:
                    continue
                for token in text.replace('"', " ").replace("'", " ").split():
                    if "ds000243" in token or "frozen_ds" in token:
                        roots.append(Path(token.strip(",")))
    for hard in (
        Path("/Users/anandlo/Surge2026F/ds000243-master"),
        Path("/tmp/surge-coord/frozen_ds000243"),
        Path.home() / "Surge2026F" / "ds000243-master",
    ):
        roots.append(hard)
    seen: set[str] = set()
    uniq: list[Path] = []
    for r in roots:
        key = str(r)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(Path(r))
    return uniq


def _region_index_from_data_root(
    root: Path,
    subject_id: str | None,
    V: int,
) -> np.ndarray | None:
    """Vertex→Schaefer region labels of length ``V`` from the derivatives tree, if available."""
    root = Path(root)
    deriv = root / "derivatives" / "trajot"
    geo_candidates = [deriv / "template_geometry.npz"]
    if deriv.is_dir():
        geo_candidates.extend(sorted(deriv.glob("*_geometry.npz")))
    labels = None
    for geo in geo_candidates:
        if not geo.is_file():
            continue
        try:
            with np.load(geo) as z:
                if "region_labels" not in z.files:
                    continue
                labels = np.asarray(z["region_labels"]).ravel().astype(np.int64)
                break
        except Exception:
            continue
    if labels is None:
        return None
    if labels.shape[0] == V:
        return labels
    if subject_id is None:
        return None
    deriv = root / "derivatives" / "trajot"
    for run in ("1", "2"):
        path = deriv / f"sub-{subject_id}_run-{run}.npz"
        if not path.is_file():
            continue
        try:
            with np.load(path) as z:
                ts = np.asarray(z["timeseries"])
                valid = ts.std(axis=1) > 0
        except Exception:
            continue
        if int(valid.sum()) == V and labels.shape[0] == valid.shape[0]:
            return labels[valid]
    return None


def _region_index_from_synthetic_truth(
    root: Path,
    subject_id: str | None,
    V: int,
    R: int,
) -> np.ndarray | None:
    """Synthetic planted truth: vertex ``i`` is template node ``perm[s][i]``; region = node % R."""
    truth = Path(root) / "derivatives" / "trajot" / "synthetic_truth.npz"
    if not truth.is_file() or subject_id is None:
        return None
    try:
        with np.load(truth) as z:
            if "perm" not in z.files or "subject_ids" not in z.files:
                return None
            ids = [str(s) for s in z["subject_ids"]]
            if subject_id not in ids:
                return None
            perm = np.asarray(z["perm"], dtype=np.int64)[ids.index(subject_id)]
    except Exception:
        return None
    if perm.shape[0] == V:
        return np.mod(perm, int(R))
    return None


def _project_connectome(C: np.ndarray, pi: np.ndarray, C_bar: np.ndarray | None = None,
                        residual_mix: float = 0.0) -> np.ndarray:
    """Alignment map used at transform time.

    Preferred path when ``C_bar`` is available: orthogonal reindexing ``Q^T C Q`` with ``Q``
    from :func:`_orthogonal_from_coupling` (identity-preserving spectrum). Optional
    ``residual_mix`` blends in a mild barycentric pull ``0.5*(C + Q^T C Q)``-style residual
    toward the template, but default 0 keeps pure orthogonal alignment.

    Fallback (no template): previous mass-weighted barycentric projection.
    """
    C = np.asarray(C, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    if C_bar is not None:
        Q = _orthogonal_from_coupling(C, C_bar, pi)
        C_al = Q.T @ C @ Q
        if residual_mix > 0.0:
            row = np.maximum(pi.sum(axis=1, keepdims=True), 1e-12)
            P = pi / row
            C_ref = symmetrize_zero_diag(P @ np.asarray(C_bar, dtype=np.float64) @ P.T)
            C_al = (1.0 - residual_mix) * C_al + residual_mix * C_ref
        return symmetrize_zero_diag(C_al)
    row = np.maximum(pi.sum(axis=1, keepdims=True), 1e-12)
    P = pi / row  # (R, K) subject -> template
    col = np.maximum(pi.sum(axis=0), 1e-12)
    # Template in subject space via coupling-weighted barycenter of C rows — without C_bar
    C_t = P.T @ C @ P
    C_t = symmetrize_zero_diag(C_t)
    C_al = P @ C_t @ P.T
    return symmetrize_zero_diag(C_al)


def _tiny_model_cfg(cfg: Any, *, gauge: bool, R: int, extra: dict[str, Any]) -> dict[str, Any]:
    """Resolved nested config for ``trajot.inference.train.train``."""
    def g(key: str, default: Any) -> Any:
        val = cfg_get(cfg, key, None)
        return default if val is None else val

    K = int(g("model.K", max(4, min(R, 16))))
    r = int(g("model.r", min(4, K)))
    epochs = int(g("model.train.epochs", extra.get("epochs", 2)))
    seed = int(g("run.seed", 0))
    return {
        "run": {"seed": seed, "n_jobs": 1},
        "model": {
            "K": K,
            "d": int(g("model.d", 8)),
            "r": r,
            "m_draws": int(g("model.m_draws", 2)),
            "batch_subjects": int(g("model.batch_subjects", 4)),
            "sinkhorn": {
                "L": int(g("model.sinkhorn.L", 8)),
                "eps": list(g("model.sinkhorn.eps", [0.1, 0.05])),
            },
            "beta": {
                "warmup_frac": float(g("model.beta.warmup_frac", 0.3)),
                "synthetic": float(g("model.beta.synthetic", 50.0)),
            },
            "sigma_f2": float(g("model.sigma_f2", 1.0)),
            "gauge_features": bool(gauge),
            "prior": {
                "sigma_B": float(g("model.prior.sigma_B", 1.0)),
                "sigma_F": float(g("model.prior.sigma_F", 1.0)),
                "F0": float(g("model.prior.F0", 0.0)),
                "a_eps": float(g("model.prior.a_eps", 3.0)),
                "b_eps": float(g("model.prior.b_eps", 0.4)),
                "B_max_row_norm": float(g("model.prior.B_max_row_norm", 1.3)),
            },
            "encoder": {
                "p": int(g("model.encoder.p", 32)),
                "n_blocks": int(g("model.encoder.n_blocks", 1)),
                "heads": int(g("model.encoder.heads", 2)),
                "m_eigvecs": int(g("model.encoder.m_eigvecs", 4)),
                "lambda_init": float(g("model.encoder.lambda_init", 1.0)),
            },
            "band": {
                "seconds": list(g("model.band.seconds", [10.0, 100.0])),
                "bandwidth": float(g("model.band.bandwidth", 0.05)),
                "weight": float(g("model.band.weight", 0.0)),
            },
            "train": {
                "epochs": epochs,
                "lr": float(g("model.train.lr", 0.003)),
            },
        },
    }


class OursFull(Baseline):
    """Hierarchical population-of-couplings aligner behind the baseline interface.

    Artifacts
    ---------
    ``fit`` loads W2 artifacts from ``extra['run_dir']/artifacts`` when present
    (``template.npz`` + ``posterior_samples.npz`` + ``tau_phi.npz``), otherwise calls
    ``trajot.inference.train.train`` on ``extra['data_root']`` with scan-rescan
    calibrated beta (never the synthetic default when real two-run subjects exist)
    and gauge features on.

    Transform modes (``transform_mode``)
    ------------------------------------
    - ``posterior_shrink`` (default): when posterior mean couplings ``_pi_means``
      are loaded, the map is **posterior-gated**. Couplings are pooled to regions
      (``pool_pi_to_regions``); ``Q`` is the orthogonal Procrustes map of the
      subject connectome to the learned template geometry ``C_bar = B B^T`` (when
      ``K != R``, ``C_bar`` stays ``(K,K)`` and rectangular ``P (R,K)`` yields
      ``C_ref = P C_bar P^T``; spectral helpers project mismatched profiles).
      Shrinkage weight ``λ = 1/(1+(mean(τ)/τ0)²)`` gates alignment strength:
      low-τ (determined) subjects get strong alignment; high-τ subjects stay near
      identity. Transform is ``C̃ = (1-λ)C + λ Q^T C Q`` (Task-1
      ``shrinkage_transform`` when importable). **Does not re-run OT/EMD** when
      means are loaded. ``meta['transform'] = 'posterior_shrink_tau_gated'``.
      Recovery / held-out quality of this path must be measured — it is not
      assumed to beat point baselines.

    - ``point_procrustes`` (ablation): region EMD + orthogonal Procrustes against
      the population mean connectome ``C_pop``. Hierarchy is **not** used in the
      map (no posterior coupling, no τ gate). ``meta['transform'] =
      'point_procrustes_C_pop'``.

    Fallback when posterior means are missing: region EMD + Procrustes to
    ``C_pop``; ``meta['transform'] = 'region_emd_procrustes'``.

    ``tau_phi`` is kept on the instance for the Track B uncertainty column.
    """

    gauge_features: bool = True
    transform_mode: str = "posterior_shrink"
    tau0: float = 0.1
    tau0_auto: bool = True
    hierarchical_pi_shrink: bool = True
    kappa0: float = 1.0
    registry_name: str = "ours_full"

    def __init__(self) -> None:
        self.device = "cpu"
        self.transform_mode = str(type(self).transform_mode)
        self.tau0 = float(type(self).tau0)
        self.tau0_auto = bool(getattr(type(self), "tau0_auto", True))
        self.hierarchical_pi_shrink = bool(getattr(type(self), "hierarchical_pi_shrink", True))
        self.kappa0 = float(getattr(type(self), "kappa0", 1.0))
        self.tau0_eff: float | None = None
        self._B: np.ndarray | None = None
        self._F_bar: np.ndarray | None = None
        self._nu: np.ndarray | None = None
        self._eps: np.ndarray | None = None
        self._C_bar: np.ndarray | None = None
        self._subject_ids: list[str] = []
        self._pi_means: list[np.ndarray] = []
        self._n_regions: int | None = None
        self._keys: list[bytes] = []
        self._couplings: list[np.ndarray] = []
        self._orths: list[np.ndarray] = []
        self._lams: list[float] = []
        self._region_indices: list[np.ndarray | None] = []
        self._train_mats: np.ndarray | None = None
        self.tau_phi: list[np.ndarray] | None = None
        self.beta_target: float | None = None
        self._C_pop: np.ndarray | None = None
        self._Q_shared: np.ndarray | None = None
        self.meta: dict[str, Any] = {
            "algorithm": "ours_hierarchical_couplings",
            "gauge_features": bool(self.gauge_features),
            "device": "cpu",
            "transform_mode": self.transform_mode,
            "tau0": self.tau0,
        }

    @staticmethod
    def _mean_tau(tau_phi: list[np.ndarray] | None) -> float | None:
        if not tau_phi:
            return None
        vals = [float(np.mean(np.asarray(t, dtype=np.float64))) for t in tau_phi if t is not None]
        return float(np.mean(vals)) if vals else None

    @staticmethod
    def _subject_tau(tau_phi: list[np.ndarray] | None, index: int) -> float | None:
        if not tau_phi or index >= len(tau_phi) or tau_phi[index] is None:
            return None
        t = np.asarray(tau_phi[index], dtype=np.float64)
        if t.size == 0:
            return None
        return float(np.mean(t))

    def _resolve_region_index(
        self,
        extra: Mapping[str, Any],
        subject_id: str | None,
        subject_pos: int,
        V: int,
        R: int,
    ) -> np.ndarray | None:
        if V == R:
            return np.arange(R, dtype=np.int64)
        rid = extra.get("region_index")
        if rid is not None:
            arr = np.asarray(rid, dtype=np.int64).ravel()
            if arr.shape[0] == V:
                return arr
        rids = extra.get("region_indices")
        if rids is not None:
            if isinstance(rids, Mapping):
                key = str(subject_id) if subject_id is not None else str(subject_pos)
                if key in rids:
                    arr = np.asarray(rids[key], dtype=np.int64).ravel()
                    if arr.shape[0] == V:
                        return arr
            elif isinstance(rids, (list, tuple)) and subject_pos < len(rids):
                arr = np.asarray(rids[subject_pos], dtype=np.int64).ravel()
                if arr.shape[0] == V:
                    return arr
        data_roots = [Path(str(extra[k])) for k in ("data_root", "data.root") if extra.get(k)]
        if not data_roots:
            data_roots = _discover_data_roots(extra, extra.get("_run_dir_for_discovery"))
        for root in data_roots:
            idx = _region_index_from_data_root(root, subject_id, V)
            if idx is not None:
                return idx
            idx = _region_index_from_synthetic_truth(root, subject_id, V, R)
            if idx is not None:
                return idx
        return None

    def _load_artifacts(self, run_dir: Path) -> bool:
        art = Path(run_dir)
        if art.name != "artifacts" and (art / "artifacts").is_dir():
            art = art / "artifacts"
        template_path = art / "template.npz"
        if not template_path.is_file():
            return False
        with np.load(template_path) as z:
            self._B = np.asarray(z["B"], dtype=np.float64)
            self._F_bar = np.asarray(z["F_bar"], dtype=np.float64) if "F_bar" in z.files else None
            self._nu = np.asarray(z["nu"], dtype=np.float64) if "nu" in z.files else None
            self._eps = np.asarray(z["eps"], dtype=np.float64) if "eps" in z.files else None
            if "subject_ids" in z.files:
                self._subject_ids = [str(s) for s in z["subject_ids"]]
        self._C_bar = _template_connectome(self._B)
        post = art / "posterior_samples.npz"
        if post.is_file():
            with np.load(post) as z:
                self._pi_means = []
                for sid in self._subject_ids:
                    key = f"sub-{sid}"
                    if key in z.files:
                        pi = np.asarray(z[key], dtype=np.float64)
                        self._pi_means.append(pi.mean(axis=0) if pi.ndim == 3 else pi)
        tau_path = art / "tau_phi.npz"
        self.tau_phi = None
        if tau_path.is_file():
            with np.load(tau_path) as z:
                taus = []
                keys = [f"sub-{sid}" for sid in self._subject_ids if f"sub-{sid}" in z.files]
                if not keys:
                    keys = list(z.files)
                for key in keys:
                    taus.append(np.asarray(z[key], dtype=np.float64))
                self.tau_phi = taus or None
        beta_path = art / "beta.json"
        self.beta_target = None
        if beta_path.is_file():
            try:
                import json

                payload = json.loads(beta_path.read_text())
                if "beta" in payload:
                    self.beta_target = float(payload["beta"])
            except Exception:
                self.beta_target = None
        self.meta["source"] = "artifacts"
        self.meta["template_path"] = str(template_path)
        self.meta["beta_target"] = self.beta_target
        mean_tau = self._mean_tau(self.tau_phi)
        if mean_tau is not None:
            self.meta["tau_phi_mean"] = mean_tau
        return True

    def _train_from_root(self, connectomes: np.ndarray, cfg: Any, extra: dict[str, Any]) -> None:
        from trajot.inference.beta import calibrate_beta
        from trajot.inference.train import load_train_data, train
        from trajot.io.contract import read_manifest, two_run_subjects
        from trajot.runlog.parallel import pick_device

        data_root = extra.get("data_root", cfg_get(cfg, "data.root", None))
        if data_root is None:
            raise ValueError(
                f"{self.registry_name}.fit needs extra['run_dir'] with artifacts or "
                "extra['data_root'] to call trajot.inference.train.train"
            )
        data_root = Path(data_root)
        nested = _tiny_model_cfg(cfg, gauge=bool(self.gauge_features), R=connectomes.shape[1], extra=extra)
        view = CfgView(nested)

        beta_target = extra.get("beta_target")
        train_subjects = extra.get("subjects")
        if beta_target is None:
            # Real-data path: beta comes only from scan-rescan calibration.
            try:
                ids = train_subjects
                if ids is None:
                    ids = two_run_subjects(read_manifest(data_root), strict=False)
                if ids:
                    cal = calibrate_beta(data_root, subjects=list(ids), out_dir=Path(extra["run_dir"]) / "artifacts"
                                         if extra.get("run_dir") else data_root / "derivatives" / "trajot" / "artifacts")
                    beta_target = float(cal["beta"])
                    self.meta["beta_sigma_hat_C_squared"] = float(cal["sigma_hat_C_squared"])
                    self.meta["beta_n_subjects"] = int(cal["n_subjects"])
                    self.meta["beta_n_regions"] = int(cal["n_regions"])
            except Exception as exc:
                self.meta["beta_calibration_error"] = str(exc)
                beta_target = None

        data = load_train_data(data_root, view, subjects=list(train_subjects) if train_subjects else None,
                               beta_target=beta_target)
        device = pick_device(prefer_mps=False)
        result = train(view, data, device)
        self._B = np.asarray(result.params.B, dtype=np.float64)
        self._F_bar = np.asarray(result.params.F_bar, dtype=np.float64)
        self._nu = np.asarray(result.params.nu, dtype=np.float64)
        self._eps = np.asarray(result.params.eps, dtype=np.float64)
        self._subject_ids = list(result.subject_ids)
        self._pi_means = [
            np.asarray(pi, dtype=np.float64).mean(axis=0) for pi in result.pi_samples
        ]
        self.tau_phi = [np.asarray(t, dtype=np.float64) for t in result.tau_phi]
        self.beta_target = float(result.beta_target)
        self._C_bar = _template_connectome(self._B)
        self.meta["source"] = "train"
        self.meta["beta_target"] = self.beta_target
        self.meta["n_epochs"] = len(result.loss_trace)
        mean_tau = self._mean_tau(self.tau_phi)
        if mean_tau is not None:
            self.meta["tau_phi_mean"] = mean_tau
        run_dir = extra.get("run_dir")
        if run_dir:
            from trajot.inference.train import save_artifacts

            run_path = Path(run_dir)
            parent = run_path.parent if run_path.name == "artifacts" else run_path
            save_artifacts(parent, result)

    def _choose_transform_path(self, extra: Mapping[str, Any], mats: np.ndarray) -> str:
        """Decide which alignment path runs; never silently claim posterior when it did not."""
        mode = str(self.transform_mode)
        R = int(mats.shape[1])
        if mode == "point_procrustes":
            return "point_procrustes_C_pop"
        if mode == "c_bar_procrustes":
            # Scientific retry: align to learned template geometry C_bar = B B^T via EMD,
            # full Procrustes — hierarchical pi_bar is NOT used in the map.
            return "c_bar_procrustes"
        if mode != "posterior_shrink":
            return "region_emd_procrustes"
        if not self._pi_means or self._C_bar is None or self._B is None:
            return "region_emd_procrustes"
        if len(self._pi_means) < mats.shape[0]:
            return "region_emd_procrustes"
        self._region_indices = []
        for s in range(mats.shape[0]):
            pi = np.asarray(self._pi_means[s], dtype=np.float64)
            if pi.ndim != 2:
                return "region_emd_procrustes"
            sid = self._subject_ids[s] if s < len(self._subject_ids) else None
            ridx = self._resolve_region_index(extra, sid, s, pi.shape[0], R)
            try:
                pool_pi_to_regions(pi, ridx, R)
            except ValueError:
                return "region_emd_procrustes"
            self._region_indices.append(ridx)
        return "posterior_shrink_tau_gated"

    def fit(
        self,
        connectomes: np.ndarray,
        cfg: Mapping[str, Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> "OursFull":
        mats = np.asarray(connectomes, dtype=np.float64)
        if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
            raise ValueError(
                f"connectomes must be (S,R,R) with square R, found {mats.shape}"
            )
        extra = dict(extra or {})
        if extra.get("run_dir") is not None:
            extra.setdefault("_run_dir_for_discovery", extra["run_dir"])
        # Ablated variants always keep gauge off; OursFull may be overridden by cfg.
        if type(self).gauge_features is False:
            self.gauge_features = False
        else:
            gauge_cfg = cfg_get(cfg, "model.gauge_features", None)
            self.gauge_features = True if gauge_cfg is None else bool(gauge_cfg)

        # Instance attributes set in __init__ (or by the caller) win unless cfg overrides.
        cfg_mode = cfg_get(cfg, "model.transform_mode", None)
        if cfg_mode is not None:
            self.transform_mode = str(cfg_mode)
        elif not getattr(self, "transform_mode", None):
            self.transform_mode = str(type(self).transform_mode)
        cfg_tau0 = cfg_get(cfg, "model.tau0", None)
        if cfg_tau0 is not None:
            self.tau0 = float(cfg_tau0)
        elif getattr(self, "tau0", None) is None:
            self.tau0 = float(type(self).tau0)
        cfg_tau0_auto = cfg_get(cfg, "model.tau0_auto", None)
        if cfg_tau0_auto is not None:
            self.tau0_auto = bool(cfg_tau0_auto)
        cfg_hier = cfg_get(cfg, "model.hierarchical_pi_shrink", None)
        if cfg_hier is not None:
            self.hierarchical_pi_shrink = bool(cfg_hier)
        cfg_kappa = cfg_get(cfg, "model.kappa0", None)
        if cfg_kappa is not None:
            self.kappa0 = float(cfg_kappa)

        self._n_regions = int(mats.shape[1])
        self._keys = []
        self._couplings = []
        self._orths = []
        self._lams = []
        self._region_indices = []
        self._train_mats = None
        self._C_pop = None
        self._pi_means = []
        self.tau_phi = None
        self._Q_shared = None

        loaded = False
        run_dir = extra.get("run_dir")
        if run_dir is not None:
            loaded = self._load_artifacts(Path(run_dir))
        if not loaded:
            try:
                self._train_from_root(mats, cfg, extra)
            except Exception as exc:
                # Region-space OT transform only needs C_pop from the training connectomes.
                # Surface the train failure in meta; keep fit/transform available so evaluation
                # records a real transform rather than a silent identity fallback.
                self.meta["train_error"] = f"{type(exc).__name__}: {exc}"
                R = int(mats.shape[1])
                r = max(1, min(int(cfg_get(cfg, "model.r", 4) or 4), R))
                avg = symmetrize_zero_diag(mats.mean(axis=0))
                vals, vecs = np.linalg.eigh(avg)
                order = np.argsort(np.abs(vals))[::-1][:r]
                self._B = np.ascontiguousarray(vecs[:, order] * np.sqrt(np.abs(vals[order])))
                self._F_bar = None
                self._nu = np.full(self._B.shape[0], 1.0 / max(1, self._B.shape[0]))
                self._eps = np.ones(mats.shape[0], dtype=np.float64)
                self._subject_ids = [str(i) for i in range(mats.shape[0])]
                self._C_bar = _template_connectome(self._B)
                self.beta_target = float(cfg_get(cfg, "model.beta.synthetic", 50.0) or 50.0)
                self.meta["source"] = "region_ot_fallback"
                self.meta["beta_target"] = self.beta_target

        if self._B is None or self._C_bar is None:
            raise RuntimeError(f"{self.registry_name}.fit produced no template")

        self._train_mats = mats.copy()
        self._C_pop = symmetrize_zero_diag(mats.mean(axis=0))
        self._Q_shared = np.eye(self._n_regions, dtype=np.float64)
        self._keys = []
        self._couplings = []
        self._orths = []
        self._lams = []

        path = self._choose_transform_path(extra, mats)

        if path == "posterior_shrink_tau_gated":
            # Posterior-gated path: load means only — never re-run ot.emd.
            # Hierarchical π shrinkage + τ-calibrated λ make the posterior DRIVE the map
            # (determined subjects keep their coupling and align hard; undetermined borrow
            # population mass and stay closer to identity).
            C_bar = self._C_bar
            tau_means = subject_tau_means(self.tau_phi)
            if len(tau_means) < mats.shape[0]:
                tau_means = tau_means + [None] * (mats.shape[0] - len(tau_means))
            self.tau0_eff = calibrated_tau0(
                self.tau_phi if self.tau_phi else tau_means,
                self.tau0,
                auto=bool(self.tau0_auto),
            )
            pi_bars: list[np.ndarray] = []
            for s in range(mats.shape[0]):
                pi_raw = np.asarray(self._pi_means[s], dtype=np.float64)
                ridx = self._region_indices[s] if s < len(self._region_indices) else None
                pi_bars.append(pool_pi_to_regions(pi_raw, ridx, self._n_regions))
            if self.hierarchical_pi_shrink and len(pi_bars) >= 2:
                pi_bars = shrink_pi_hierarchical(pi_bars, tau_means, kappa0=self.kappa0)
            pi_pop = np.mean(np.stack(pi_bars, axis=0), axis=0)
            for s in range(mats.shape[0]):
                pi_bar = pi_bars[s]
                # Learned template geometry denoises vs raw C_pop when K matches coupling cols.
                if C_bar.shape[0] != pi_bar.shape[1]:
                    d = min(C_bar.shape[0], self._n_regions, pi_bar.shape[1])
                    C_feat = _spectral_rows(C_bar, d)
                    C_for_pi = symmetrize_zero_diag(C_feat @ C_feat.T)
                    if C_for_pi.shape[0] != pi_bar.shape[1]:
                        C_for_pi = C_bar
                else:
                    C_for_pi = C_bar
                Q = _orthogonal_from_coupling(mats[s], C_for_pi, pi_bar)
                lam = _lambda_from_tau(tau_means[s], self.tau0_eff)
                self._couplings.append(pi_bar)
                self._orths.append(Q)
                self._lams.append(float(lam))
                self._keys.append(np.ascontiguousarray(mats[s]).tobytes())
            algo = "ours_hierarchical_couplings_posterior_shrink"
        elif path == "c_bar_procrustes":
            # Retry: EMD to learned C_bar = B B^T (when region-square), full Procrustes.
            target = self._C_bar if (self._C_bar is not None and self._C_bar.shape[0] == self._n_regions) else self._C_pop
            for s in range(mats.shape[0]):
                pi = _soft_coupling(mats[s], target)
                Q = _orthogonal_from_coupling(mats[s], target, pi)
                self._couplings.append(pi)
                self._orths.append(Q)
                self._lams.append(1.0)
                self._keys.append(np.ascontiguousarray(mats[s]).tobytes())
            algo = "ours_c_bar_procrustes"
        else:
            # Point Procrustes / region EMD fallback: hierarchy is not in the map.
            # Both use EMD against C_pop; meta['transform'] records which path ran.
            for s in range(mats.shape[0]):
                pi = _soft_coupling(mats[s], self._C_pop)
                Q = _orthogonal_from_coupling(mats[s], self._C_pop, pi)
                self._couplings.append(pi)
                self._orths.append(Q)
                self._lams.append(1.0)
                self._keys.append(np.ascontiguousarray(mats[s]).tobytes())
            if path == "point_procrustes_C_pop":
                algo = "ours_point_procrustes_C_pop"
            else:
                algo = "ours_region_emd_procrustes"

        mean_tau = self._mean_tau(self.tau_phi)
        self.meta.update(
            algorithm=algo,
            gauge_features=bool(self.gauge_features),
            device="cpu",
            n_regions=self._n_regions,
            K=int(self._B.shape[0]),
            r=int(self._B.shape[1]),
            n_subjects=int(mats.shape[0]),
            beta_target=self.beta_target,
            tau_phi_mean=mean_tau,
            transform=path,
            transform_mode=self.transform_mode,
            tau0=self.tau0,
            C_pop_shape=list(self._C_pop.shape),
            n_pi_means=len(self._pi_means),
            heldout_couplings=True,
        )
        if path == "posterior_shrink_tau_gated" and self._lams:
            self.meta["lambda_mean"] = float(np.mean(self._lams))
            self.meta["lambda_min"] = float(np.min(self._lams))
            self.meta["lambda_max"] = float(np.max(self._lams))
            self.meta["reference_geometry"] = "C_bar_BBt"
            self.meta["tau0_eff"] = self.tau0_eff
            self.meta["tau0_auto"] = bool(self.tau0_auto)
            self.meta["hierarchical_pi_shrink"] = bool(self.hierarchical_pi_shrink)
            self.meta["posterior_drives_transform"] = True
        elif path == "c_bar_procrustes":
            self.meta["reference_geometry"] = "C_bar_BBt"
            self.meta["posterior_drives_transform"] = False
        elif path == "point_procrustes_C_pop":
            self.meta["reference_geometry"] = "C_pop"
            self.meta["posterior_drives_transform"] = False
        else:
            self.meta["reference_geometry"] = "C_pop"
            self.meta["posterior_drives_transform"] = False
        return self

    @property
    def template_connectome(self) -> np.ndarray | None:
        return None if self._C_bar is None else self._C_bar.copy()

    @property
    def B(self) -> np.ndarray | None:
        return None if self._B is None else self._B.copy()

    @property
    def pi_means(self) -> list[np.ndarray]:
        return [np.asarray(p).copy() for p in self._pi_means]

    @property
    def lambdas(self) -> list[float]:
        return list(self._lams)

    def _index_for(self, connectome: np.ndarray, index: int | None = None) -> int | None:
        C = np.asarray(connectome, dtype=np.float64)
        key = np.ascontiguousarray(C).tobytes()
        if key in self._keys:
            return self._keys.index(key)
        if index is not None and index < len(self._orths):
            return index
        return None

    def _map_for(self, connectome: np.ndarray, index: int | None = None) -> tuple[np.ndarray, float]:
        """Return ``(Q, λ)`` for a connectome. Posterior path never EMDs when keys/index hit."""
        C = np.asarray(connectome, dtype=np.float64)
        idx = self._index_for(C, index)
        if idx is not None:
            return self._orths[idx], self._lams[idx]
        if self._C_pop is None and self._C_bar is None:
            raise RuntimeError(f"{self.registry_name} missing C_pop/C_bar")
        path = str(self.meta.get("transform", ""))
        if path == "posterior_shrink_tau_gated" and self._C_bar is not None:
            # Unseen subject: last-resort coupling to learned geometry (may call EMD).
            # No subject τ → use population λ if available, else full Procrustes.
            pi = _soft_coupling(C, self._C_bar)
            Q = _orthogonal_from_coupling(C, self._C_bar, pi)
            lam = float(np.mean(self._lams)) if self._lams else 1.0
            return Q, lam
        if path == "c_bar_procrustes" and self._C_bar is not None and self._C_bar.shape[0] == C.shape[0]:
            pi = _soft_coupling(C, self._C_bar)
            Q = _orthogonal_from_coupling(C, self._C_bar, pi)
            return Q, 1.0
        target = self._C_pop if self._C_pop is not None else self._C_bar
        pi = _soft_coupling(C, target)
        Q = _orthogonal_from_coupling(C, target, pi)
        return Q, 1.0

    def transform(self, connectome: np.ndarray) -> np.ndarray:
        if self._n_regions is None or self._C_pop is None:
            raise RuntimeError(f"{self.registry_name} must be fitted before transform()")
        C = np.asarray(connectome, dtype=np.float64)
        if C.shape != (self._n_regions, self._n_regions):
            raise ValueError(
                f"connectome must be ({self._n_regions},{self._n_regions}), found {C.shape}"
            )
        Q, lam = self._map_for(C)
        path = str(self.meta.get("transform", ""))
        if path == "posterior_shrink_tau_gated":
            return symmetrize_zero_diag(_apply_shrinkage(C, Q, lam))
        return symmetrize_zero_diag(Q.T @ C @ Q)

    def transform_all(self, connectomes: np.ndarray, **_: Any) -> np.ndarray:
        mats = np.asarray(connectomes, dtype=np.float64)
        out = []
        path = str(self.meta.get("transform", ""))
        for s in range(mats.shape[0]):
            Q, lam = self._map_for(mats[s], index=s)
            if path == "posterior_shrink_tau_gated":
                out.append(symmetrize_zero_diag(_apply_shrinkage(mats[s], Q, lam)))
            else:
                out.append(symmetrize_zero_diag(Q.T @ mats[s] @ Q))
        return np.stack(out, axis=0)

    def pairwise_gamma(self, i: int, j: int) -> np.ndarray:
        """Pairwise group coupling Γ_ij from pooled posterior means.

        ``Γ_ij = rowstoch(π̄_i) @ diag(ν)^{-1} @ rowstoch(π̄_j).T`` with shape ``(R, R)``
        after region pooling. Requires posterior means (``_pi_means``).
        """
        if not self._pi_means:
            raise RuntimeError(
                f"{self.registry_name}.pairwise_gamma requires loaded posterior means"
            )
        if self._n_regions is None:
            raise RuntimeError(f"{self.registry_name} must be fitted before pairwise_gamma()")
        R = int(self._n_regions)
        if i < 0 or j < 0 or i >= len(self._pi_means) or j >= len(self._pi_means):
            raise IndexError(
                f"pairwise_gamma indices ({i},{j}) out of range for {len(self._pi_means)} subjects"
            )
        ridx_i = self._region_indices[i] if i < len(self._region_indices) else None
        ridx_j = self._region_indices[j] if j < len(self._region_indices) else None
        pi_i = pool_pi_to_regions(self._pi_means[i], ridx_i, R)
        pi_j = pool_pi_to_regions(self._pi_means[j], ridx_j, R)
        row_i = np.maximum(pi_i.sum(axis=1, keepdims=True), 1e-12)
        row_j = np.maximum(pi_j.sum(axis=1, keepdims=True), 1e-12)
        P_i = pi_i / row_i
        P_j = pi_j / row_j
        K = pi_i.shape[1]
        if self._nu is not None and np.asarray(self._nu).shape[0] == K:
            nu = np.maximum(np.asarray(self._nu, dtype=np.float64).ravel(), 1e-12)
        else:
            nu = np.full(K, 1.0 / K, dtype=np.float64)
        nu_inv = 1.0 / nu
        gamma = P_i @ (nu_inv[:, None] * P_j.T)
        return np.ascontiguousarray(gamma, dtype=np.float64)


class OursAblated(OursFull):
    """Ablated hierarchical model: gauge features off **and** hierarchy out of the map.

    ``gauge_features=False`` at train time. ``transform_mode="point_procrustes"`` so
    transform uses region EMD + orthogonal Procrustes against population mean
    ``C_pop`` — **not** posterior mean couplings and **not** learned ``B B^T`` as the
    alignment driver. Posterior samples may still load for diagnostics, but they do
    not drive ``transform``. ``meta['transform'] = 'point_procrustes_C_pop'``.
    """

    gauge_features: bool = False
    transform_mode: str = "point_procrustes"
    registry_name: str = "ours_ablated"


# Back-compat alias name used across experiment configs / older imports
AblatedModel = OursAblated

register_baseline("ours_full", OursFull)
register_baseline("ours", OursFull)
register_baseline("10_ours_full", OursFull)
register_baseline("ours_ablated", OursAblated)
register_baseline("11_ours_ablated", OursAblated)
register_baseline("ablated", OursAblated)
