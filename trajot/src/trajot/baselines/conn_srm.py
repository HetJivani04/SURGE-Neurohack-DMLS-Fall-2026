from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Baseline, cfg_get, register_baseline, symmetrize_zero_diag


class ConnSRM(Baseline):
    """Connectivity shared-response model (SRM) on connectivity features.

    Related to connectivity-SRM / shared-response alignment of connectivity
    profiles (PMID 32325212 family). This is **not** independent eigenvalue
    truncation of each connectome.

    Model: for subject connectome ``C_s`` treated as a data matrix
    ``(R regions × R profile dims)``, SRM assumes

    ``C_s ≈ A_s S``

    with shared components ``S (k × R)`` and subject-specific loadings
    ``A_s (R × k)``. Alternating least squares estimates ``S`` and ``{A_s}``.
    ``transform`` estimates ``A`` for a (possibly new) connectome against the
    learned shared response, finds the region-space orthogonal map to the
    template loading ``A_bar`` via Procrustes on ``A A_bar^T``, and applies
    ``C_aligned = Q^T C Q``.
    """

    def __init__(self, rank: int | None = None, n_iter: int = 20, seed: int = 0) -> None:
        self.device = "cpu"
        self._rank = rank
        self._n_iter = int(n_iter)
        self._seed = int(seed)
        self._S: np.ndarray | None = None
        self._A_template: np.ndarray | None = None
        self._A_list: list[np.ndarray] = []
        self._keys: list[bytes] = []
        self._n_regions: int | None = None
        self.meta: dict[str, Any] = {
            "algorithm": "conn_srm_shared_response_on_connectivity_profiles",
            "device": "cpu",
        }

    @staticmethod
    def _estimate_A(C: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Least-squares subject loadings ``A`` s.t. ``A S ≈ C`` (``A`` is ``(R, k)``)."""
        SS = S @ S.T
        SS = SS + 1e-8 * np.eye(SS.shape[0])
        return C @ S.T @ np.linalg.inv(SS)

    @staticmethod
    def _region_procrustes(A: np.ndarray, A_bar: np.ndarray) -> np.ndarray:
        """Region-space orthogonal ``O (R, R)`` aligning loadings ``A`` to ``A_bar``."""
        M = np.asarray(A, dtype=np.float64) @ np.asarray(A_bar, dtype=np.float64).T
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        return np.ascontiguousarray(U @ Vt, dtype=np.float64)

    def _fit_srm(self, mats: np.ndarray, rank: int) -> tuple[np.ndarray, list[np.ndarray]]:
        S, R, _ = mats.shape
        cov = np.zeros((R, R), dtype=np.float64)
        for s in range(S):
            cov += mats[s] @ mats[s].T
        cov /= S
        evals, evecs = np.linalg.eigh(0.5 * (cov + cov.T))
        order = np.argsort(evals)[::-1][:rank]
        S_mat = evecs[:, order].T.astype(np.float64)
        scales = np.sqrt(np.clip(evals[order], 1e-8, None))
        S_mat = S_mat * scales[:, None]

        A_list = [self._estimate_A(mats[s], S_mat) for s in range(S)]
        for _ in range(self._n_iter):
            acc = np.zeros_like(S_mat)
            for s in range(S):
                A = A_list[s]
                AA = A.T @ A + 1e-8 * np.eye(A.shape[1])
                acc += np.linalg.inv(AA) @ A.T @ mats[s]
            S_mat = acc / S
            U, _, Vt = np.linalg.svd(S_mat, full_matrices=False)
            S_mat = (U @ Vt)[:rank]
            for s in range(S):
                A_list[s] = self._estimate_A(mats[s], S_mat)
                Qs, _ = np.linalg.qr(A_list[s])
                A_list[s] = Qs[:, :rank]
        return S_mat, A_list

    def fit(
        self,
        connectomes: np.ndarray,
        cfg: Mapping[str, Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> "ConnSRM":
        mats = np.asarray(connectomes, dtype=np.float64)
        if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
            raise ValueError(
                f"connectomes must be (S,R,R) with square R, found {mats.shape}"
            )
        self._n_regions = int(mats.shape[1])
        R = self._n_regions
        default_rank = min(8, max(1, R // 2))
        rank = self._rank
        if rank is None:
            model_cfg = cfg_get(cfg, "model.r", None)
            rank = int(model_cfg) if model_cfg is not None else default_rank
        rank = max(1, min(int(rank), R))

        self._S, self._A_list = self._fit_srm(mats, rank)
        A_bar = np.mean(np.stack(self._A_list, axis=0), axis=0)
        Qs, _ = np.linalg.qr(A_bar)
        self._A_template = Qs[:, :rank]
        self._keys = [np.ascontiguousarray(mats[s]).tobytes() for s in range(mats.shape[0])]
        self.meta.update(
            n_regions=R,
            rank=rank,
            n_iter=self._n_iter,
            algorithm="conn_srm_shared_response_on_connectivity_profiles",
            device="cpu",
        )
        return self

    @property
    def shared_response(self) -> np.ndarray | None:
        return self._S

    @property
    def template_loadings(self) -> np.ndarray | None:
        return self._A_template

    def transform(self, connectome: np.ndarray) -> np.ndarray:
        if self._S is None or self._A_template is None or self._n_regions is None:
            raise RuntimeError("ConnSRM must be fitted before transform()")
        C = np.asarray(connectome, dtype=np.float64)
        if C.shape != (self._n_regions, self._n_regions):
            raise ValueError(
                f"connectome must be ({self._n_regions},{self._n_regions}), found {C.shape}"
            )
        key = np.ascontiguousarray(C).tobytes()
        if key in self._keys:
            A = self._A_list[self._keys.index(key)]
        else:
            A = self._estimate_A(C, self._S)
            Qs, _ = np.linalg.qr(A)
            A = Qs[:, : self._A_template.shape[1]]
        O = self._region_procrustes(A, self._A_template)
        return symmetrize_zero_diag(O.T @ C @ O)

    def transform_all(self, connectomes: np.ndarray, **_: Any) -> np.ndarray:
        mats = np.asarray(connectomes, dtype=np.float64)
        return np.stack([self.transform(mats[s]) for s in range(mats.shape[0])], axis=0)


register_baseline("conn_srm", ConnSRM)
register_baseline("connectivity_srm", ConnSRM)
register_baseline("03_conn_srm", ConnSRM)
