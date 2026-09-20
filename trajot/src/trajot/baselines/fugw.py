from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Baseline, cfg_get, register_baseline, symmetrize_zero_diag


def _structure_matrix(C: np.ndarray) -> np.ndarray:
    """GW structure cost from a Fisher-z connectome (non-negative, zero diagonal)."""
    S = np.abs(np.asarray(C, dtype=np.float64))
    return symmetrize_zero_diag(S)


def _feature_cost(C_src: np.ndarray, C_dst: np.ndarray) -> np.ndarray:
    """Euclidean cost between connectivity row profiles."""
    A = np.asarray(C_src, dtype=np.float64)
    B = np.asarray(C_dst, dtype=np.float64)
    # (R_s, 1, R) - (1, R_d, R) -> (R_s, R_d, R) is heavy; use Gram form
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)


def _uniform_mass(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


class FUGW(Baseline):
    """Fused Gromov-Wasserstein alignment (Thual et al., NeurIPS 2022) via POT.

    ``fit`` builds a template (mean connectome) and a subject-to-template coupling
    for each training subject using ``ot.gromov.fused_gromov_wasserstein`` (or
    plain GW when ``alpha=0``). Structure cost is ``|C|`` on connectivity; feature
    cost is Euclidean distance between connectivity row profiles. ``transform``
    derives an ``(R, R)`` map from the coupling by barycentric projection of the
    subject connectome toward the template through the learned soft correspondence.

    float64, CPU only — POT/FUGW has no MPS branch. ``meta['device'] == 'cpu'``.
    """

    def __init__(
        self,
        alpha: float | None = None,
        max_iter: int | None = None,
        use_entropic: bool = False,
        epsilon: float = 0.05,
    ) -> None:
        self.device = "cpu"
        self._alpha = alpha
        self._max_iter = max_iter
        self._use_entropic = use_entropic
        self._epsilon = float(epsilon)
        self._template: np.ndarray | None = None
        self._couplings: list[np.ndarray] = []
        self._keys: list[bytes] = []
        self._n_regions: int | None = None
        self.meta: dict[str, Any] = {
            "algorithm": "fugw_pot_fused_gromov_wasserstein",
            "device": "cpu",
            "loss_fun": "square_loss",
        }

    def _solve_coupling(self, C_src: np.ndarray, C_dst: np.ndarray) -> np.ndarray:
        import ot

        C_src = np.asarray(C_src, dtype=np.float64)
        C_dst = np.asarray(C_dst, dtype=np.float64)
        Rs, Rd = C_src.shape[0], C_dst.shape[0]
        p = _uniform_mass(Rs)
        q = _uniform_mass(Rd)
        M = _feature_cost(C_src, C_dst)
        S1 = _structure_matrix(C_src)
        S2 = _structure_matrix(C_dst)
        alpha = 0.5 if self._alpha is None else float(self._alpha)
        max_iter = 50 if self._max_iter is None else int(self._max_iter)

        if self._use_entropic:
            pi = ot.gromov.entropic_fused_gromov_wasserstein(
                M,
                S1,
                S2,
                p,
                q,
                loss_fun="square_loss",
                alpha=alpha,
                epsilon=self._epsilon,
                max_iter=max(20, max_iter // 2),
            )
        elif alpha <= 0.0:
            pi = ot.gromov.gromov_wasserstein(
                S1, S2, p, q, loss_fun="square_loss", armijo=True, max_iter=max_iter
            )
        else:
            pi = ot.gromov.fused_gromov_wasserstein(
                M,
                S1,
                S2,
                p,
                q,
                loss_fun="square_loss",
                alpha=alpha,
                armijo=True,
                max_iter=max_iter,
            )
        return np.ascontiguousarray(pi, dtype=np.float64)

    @staticmethod
    def _apply_coupling(C: np.ndarray, pi: np.ndarray, C_template: np.ndarray) -> np.ndarray:
        """Barycentric projection of subject connectome through coupling ``pi``.

        ``P`` row-normalises ``pi`` (subject -> template). Aligned connectome is the
        subject geometry transported into template indexing:
        ``C_al = P C P^T`` when shapes allow a square soft map; otherwise project
        through the template-space image ``(pi^T C pi)`` and back with ``pi``.
        """
        C = np.asarray(C, dtype=np.float64)
        pi = np.asarray(pi, dtype=np.float64)
        row = np.maximum(pi.sum(axis=1, keepdims=True), 1e-12)
        col = np.maximum(pi.sum(axis=0, keepdims=True), 1e-12)
        P = pi / row  # (R_s, R_d) subject -> template
        # Template-space image of the subject connectome
        C_t = (pi.T @ C @ pi) / (col.T @ col)
        C_t = symmetrize_zero_diag(C_t)
        if C_t.shape == C.shape:
            # Soft permutation pullback onto subject/template node indexing
            C_al = P @ C_t @ P.T
        else:
            C_al = P.T @ C_t @ P
        # Blend a light pull toward the subject's own geometry to preserve identity
        if C_al.shape == C.shape:
            C_al = 0.5 * C_al + 0.5 * symmetrize_zero_diag(P @ C @ P.T) if P.shape[0] == P.shape[1] else C_al
        return symmetrize_zero_diag(C_al)

    def fit(
        self,
        connectomes: np.ndarray,
        cfg: Mapping[str, Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> "FUGW":
        mats = np.asarray(connectomes, dtype=np.float64)
        if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
            raise ValueError(
                f"connectomes must be (S,R,R) with square R, found {mats.shape}"
            )
        if cfg_get(cfg, "fugw.use_entropic", None) is not None:
            self._use_entropic = bool(cfg_get(cfg, "fugw.use_entropic"))
        if cfg_get(cfg, "fugw.alpha", None) is not None:
            self._alpha = float(cfg_get(cfg, "fugw.alpha"))
        if cfg_get(cfg, "fugw.max_iter", None) is not None:
            self._max_iter = int(cfg_get(cfg, "fugw.max_iter"))

        self._n_regions = int(mats.shape[1])
        self._template = symmetrize_zero_diag(mats.mean(axis=0))
        self._couplings = []
        self._keys = []
        for s in range(mats.shape[0]):
            pi = self._solve_coupling(mats[s], self._template)
            self._couplings.append(pi)
            self._keys.append(np.ascontiguousarray(mats[s]).tobytes())

        self.meta.update(
            n_regions=self._n_regions,
            n_subjects=int(mats.shape[0]),
            alpha=0.5 if self._alpha is None else float(self._alpha),
            algorithm="fugw_pot_fused_gromov_wasserstein",
            device="cpu",
        )
        return self

    @property
    def template(self) -> np.ndarray | None:
        return self._template

    @property
    def couplings(self) -> list[np.ndarray]:
        return list(self._couplings)

    def transform(self, connectome: np.ndarray) -> np.ndarray:
        if self._template is None or self._n_regions is None:
            raise RuntimeError("FUGW must be fitted before transform()")
        C = np.asarray(connectome, dtype=np.float64)
        if C.shape != (self._n_regions, self._n_regions):
            raise ValueError(
                f"connectome must be ({self._n_regions},{self._n_regions}), found {C.shape}"
            )
        key = np.ascontiguousarray(C).tobytes()
        if key in self._keys:
            pi = self._couplings[self._keys.index(key)]
        else:
            pi = self._solve_coupling(C, self._template)
        return self._apply_coupling(C, pi, self._template)

    def transform_all(self, connectomes: np.ndarray, **_: Any) -> np.ndarray:
        mats = np.asarray(connectomes, dtype=np.float64)
        return np.stack([self.transform(mats[s]) for s in range(mats.shape[0])], axis=0)


register_baseline("fugw", FUGW)
