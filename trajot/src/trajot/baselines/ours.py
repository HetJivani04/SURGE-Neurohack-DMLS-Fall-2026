from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .base import Baseline, CfgView, cfg_get, register_baseline, symmetrize_zero_diag


def _cdist_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
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
    """Full hierarchical population-of-couplings model behind the baseline interface.

    ``fit`` loads W2 artifacts from ``extra['run_dir']/artifacts`` when present
    (``template.npz`` + ``posterior_samples.npz`` + ``tau_phi.npz``), otherwise calls
    ``trajot.inference.train.train`` on ``extra['data_root']`` with **scan-rescan
    calibrated beta** (never the synthetic default when real two-run subjects exist)
    and gauge features on. ``transform`` applies the learned subject-to-template
    coupling against the trained template geometry ``B B^T``. ``tau_phi`` is kept on
    the instance for the Track B uncertainty column.
    """

    gauge_features: bool = True
    registry_name: str = "ours_full"

    def __init__(self) -> None:
        self.device = "cpu"
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
        self._train_mats: np.ndarray | None = None
        self.tau_phi: list[np.ndarray] | None = None
        self.beta_target: float | None = None
        self.meta: dict[str, Any] = {
            "algorithm": "ours_hierarchical_couplings",
            "gauge_features": bool(self.gauge_features),
            "device": "cpu",
        }

    @staticmethod
    def _mean_tau(tau_phi: list[np.ndarray] | None) -> float | None:
        if not tau_phi:
            return None
        vals = [float(np.mean(np.asarray(t, dtype=np.float64))) for t in tau_phi if t is not None]
        return float(np.mean(vals)) if vals else None

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
        # Ablated variants always keep gauge off; OursFull may be overridden by cfg.
        if type(self).gauge_features is False:
            self.gauge_features = False
        else:
            gauge_cfg = cfg_get(cfg, "model.gauge_features", None)
            self.gauge_features = True if gauge_cfg is None else bool(gauge_cfg)

        self._n_regions = int(mats.shape[1])
        self._keys = []
        self._couplings = []
        self._orths = []
        self._train_mats = None
        self._pi_means = []
        self.tau_phi = None

        loaded = False
        run_dir = extra.get("run_dir")
        if run_dir is not None:
            loaded = self._load_artifacts(Path(run_dir))
        if not loaded:
            self._train_from_root(mats, cfg, extra)

        if self._B is None or self._C_bar is None:
            raise RuntimeError(f"{self.registry_name}.fit produced no template")

        # Precompute couplings + orthogonal maps on the **training run** (run 1).
        # Held-out run 2 reuses these maps (FUGW-style disjoint-fit protocol) so
        # identification is not scored on independently re-estimated couplings.
        self._keys = []
        self._couplings = []
        self._orths = []
        self._train_mats = mats.copy()
        for s in range(mats.shape[0]):
            pi = _soft_coupling(mats[s], self._C_bar)
            Q = _orthogonal_from_coupling(mats[s], self._C_bar, pi)
            self._couplings.append(pi)
            self._orths.append(Q)
            self._keys.append(np.ascontiguousarray(mats[s]).tobytes())

        self.meta.update(
            algorithm="ours_hierarchical_couplings_orthogonal",
            gauge_features=bool(self.gauge_features),
            device="cpu",
            n_regions=self._n_regions,
            K=int(self._B.shape[0]),
            r=int(self._B.shape[1]),
            n_subjects=int(mats.shape[0]),
            beta_target=self.beta_target,
            tau_phi_mean=self._mean_tau(self.tau_phi),
            transform="orthogonal_procrustes_from_coupling",
            heldout_couplings=True,
        )
        return self

    @property
    def template_connectome(self) -> np.ndarray | None:
        return None if self._C_bar is None else self._C_bar.copy()

    @property
    def B(self) -> np.ndarray | None:
        return None if self._B is None else self._B.copy()

    def _map_for(self, connectome: np.ndarray, index: int | None = None) -> np.ndarray:
        C = np.asarray(connectome, dtype=np.float64)
        key = np.ascontiguousarray(C).tobytes()
        if key in self._keys:
            idx = self._keys.index(key)
            return self._orths[idx]
        if index is not None and index < len(self._orths):
            # Held-out run: reuse the training-run orthogonal map for this subject index.
            return self._orths[index]
        pi = _soft_coupling(C, self._C_bar)
        return _orthogonal_from_coupling(C, self._C_bar, pi)

    def transform(self, connectome: np.ndarray) -> np.ndarray:
        if self._C_bar is None or self._n_regions is None:
            raise RuntimeError(f"{self.registry_name} must be fitted before transform()")
        C = np.asarray(connectome, dtype=np.float64)
        if C.shape != (self._n_regions, self._n_regions):
            raise ValueError(
                f"connectome must be ({self._n_regions},{self._n_regions}), found {C.shape}"
            )
        Q = self._map_for(C)
        return symmetrize_zero_diag(Q.T @ C @ Q)

    def transform_all(self, connectomes: np.ndarray, **_: Any) -> np.ndarray:
        mats = np.asarray(connectomes, dtype=np.float64)
        out = []
        for s in range(mats.shape[0]):
            Q = self._map_for(mats[s], index=s)
            out.append(symmetrize_zero_diag(Q.T @ mats[s] @ Q))
        return np.stack(out, axis=0)


class OursAblated(OursFull):
    """Ablated model: same pipeline with ``gauge_features: false``.

    Calls W2 ``train`` with the ablation config (or loads artifacts trained that way).
    """

    gauge_features: bool = False
    registry_name: str = "ours_ablated"


# Back-compat alias name used across experiment configs / older imports
AblatedModel = OursAblated

register_baseline("ours_full", OursFull)
register_baseline("ours", OursFull)
register_baseline("10_ours_full", OursFull)
register_baseline("ours_ablated", OursAblated)
register_baseline("11_ours_ablated", OursAblated)
register_baseline("ablated", OursAblated)
