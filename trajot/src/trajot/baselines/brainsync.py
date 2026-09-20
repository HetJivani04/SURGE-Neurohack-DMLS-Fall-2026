from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Baseline, register_baseline, symmetrize_zero_diag


def _connectome_from_ts(ts_vt: np.ndarray) -> np.ndarray:
    """Fisher-z connectivity from timeseries ``(V, T)``."""
    X = np.asarray(ts_vt, dtype=np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std <= 0, 1.0, std)
    X = X / std
    T = max(X.shape[1] - 1, 1)
    corr = np.clip((X @ X.T) / T, -0.999999, 0.999999)
    np.fill_diagonal(corr, 0.0)
    z = np.arctanh(corr)
    return symmetrize_zero_diag(z)


def brainsync_Q(X_vt: np.ndarray, Y_vt: np.ndarray) -> np.ndarray:
    """Closed-form BrainSync (Joshi et al. 2018; issue #5) on contract ``(V, T)`` data.

    Math form (pinned convention)
    -----------------------------
    Contract timeseries are ``(V, T)``. Cross-covariance in time is
    ``M = X^T Y`` ``(T, T)``; thin SVD ``M = U S V^T``; ``Q = U V^T`` orthogonal
    ``(T, T)``; ``X_aligned = X Q`` ``(V, T)``.

    This is the issue #5 operator (``Q (T,T)``, ``X_aligned = X Q``) with the SVD
    taken on the time-time cross-covariance that makes that right-action well-posed.
    Float64 on CPU via ``numpy.linalg.svd``.

    BrainSync maximises correlation even when its assumption of common networks fails
    (Joshi et al. 2018).
    """
    X = np.asarray(X_vt, dtype=np.float64)
    Y = np.asarray(Y_vt, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("BrainSync expects 2D (V, T) arrays")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"vertex count mismatch: X {X.shape}, Y {Y.shape}")
    T = min(X.shape[1], Y.shape[1])
    X = X[:, :T]
    Y = Y[:, :T]
    M = X.T @ Y  # (T, T)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    Q = U @ Vt
    return np.ascontiguousarray(Q, dtype=np.float64)


def vertex_brainsync_Q(X_vt: np.ndarray, Y_vt: np.ndarray) -> np.ndarray:
    """Vertex-space operator ``Q = U V^T`` from ``SVD(X Y^T)`` with ``(V, T)`` inputs.

    ``Q`` is ``(V, V)``; ``X_aligned = Q^T X``; connectome map ``C' = Q^T C Q``.
    """
    X = np.asarray(X_vt, dtype=np.float64)
    Y = np.asarray(Y_vt, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"vertex count mismatch: X {X.shape}, Y {Y.shape}")
    M = X @ Y.T
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return np.ascontiguousarray(U @ Vt, dtype=np.float64)


def to_connectome_transform(
    Q: np.ndarray,
    C: np.ndarray | None = None,
    timeseries: np.ndarray | None = None,
) -> np.ndarray:
    """Map a BrainSync operator into the ``(R, R)`` connectome interface.

    * ``timeseries`` given ``(V, T)`` (or ``(T, V)`` with T matching Q): apply
      time-domain ``X @ Q`` and recompute connectivity (issue #5 adapter).
    * Only ``C`` given and ``Q`` square on the same nodes: ``Q^T C Q``
      (vertex-space operator).
    * Time-space ``Q`` without timeseries cannot act on a spatial connectome.
    """
    Qm = np.asarray(Q, dtype=np.float64)
    if timeseries is not None:
        ts = np.asarray(timeseries, dtype=np.float64)
        if ts.ndim != 2:
            raise ValueError(f"timeseries must be 2D, found {ts.shape}")
        if Qm.ndim != 2 or Qm.shape[0] != Qm.shape[1]:
            raise ValueError(f"time-domain Q must be square, found {Qm.shape}")
        if ts.shape[1] == Qm.shape[0]:
            X_vt = ts  # (V, T)
        elif ts.shape[0] == Qm.shape[0]:
            X_vt = ts.T  # arrived as (T, V)
        else:
            raise ValueError(f"cannot apply Q{Qm.shape} to timeseries {ts.shape}")
        aligned = X_vt @ Qm
        return _connectome_from_ts(aligned)

    if C is None:
        raise ValueError("to_connectome_transform needs timeseries or a connectome")
    Cm = np.asarray(C, dtype=np.float64)
    if Qm.shape[0] == Qm.shape[1] == Cm.shape[0] == Cm.shape[1]:
        return symmetrize_zero_diag(Qm.T @ Cm @ Qm)
    raise ValueError(
        f"BrainSync Q{Qm.shape} cannot transform connectome {Cm.shape}; "
        "pass timeseries so connectivity can be recomputed"
    )


class BrainSync(Baseline):
    """Time-series BrainSync (Joshi et al. 2018; issue #5).

    Convention
    ----------
    Contract timeseries are ``(V, T)`` float32. Pinned math:

    ``M = X^T Y`` ``(T, T)``, thin ``SVD(M) = U S V^T``, ``Q = U V^T`` orthogonal
    ``(T, T)``, ``X_aligned = X Q`` ``(V, T)``.

    ``fit`` requires ``extra['timeseries_run1']`` (or ``extra['timeseries']``) at
    connectome resolution (``V == R``) so connectivity can be recomputed after sync.
    Connectome-only fit raises — Procrustes on connectomes is not BrainSync.

    ``space='time'`` (default): time-domain ``Q`` as above.
    ``space='vertex'``: ``Q`` from ``SVD(X Y^T)`` so ``Q`` is ``(V, V)`` and
    ``transform(C)`` is ``Q^T C Q``.

    BrainSync maximises correlation even when its assumption of common networks fails
    (Joshi et al. 2018) — which is why the no-alignment row and the null matter.
    """

    def __init__(self, space: str = "time") -> None:
        space = str(space).lower()
        if space not in {"time", "vertex"}:
            raise ValueError(f"space must be 'time' or 'vertex', got {space!r}")
        self.space = space
        self.device = "cpu"
        self._reference_vt: np.ndarray | None = None
        self._n_regions: int | None = None
        self._Q: np.ndarray | None = None
        self._Q_list: list[np.ndarray] = []
        self._aligned: np.ndarray | None = None
        self._aligned_by_key: dict[bytes, np.ndarray] = {}
        self.meta: dict[str, Any] = {
            "algorithm": "brainsync_svd_uvT",
            "space": space,
            "device": "cpu",
            "convention": "(V,T) contract; Q=(T,T) from SVD(X^T Y); X_aligned=X Q",
        }

    @staticmethod
    def _stack_timeseries(timeseries: Any) -> list[np.ndarray]:
        if isinstance(timeseries, (list, tuple)):
            return [np.asarray(t) for t in timeseries]
        arr = np.asarray(timeseries)
        if arr.ndim == 3:
            return [arr[i] for i in range(arr.shape[0])]
        if arr.ndim == 2:
            return [arr]
        raise ValueError(
            f"timeseries must be (S,V,T), list of (V,T), or (V,T); got {arr.shape}"
        )

    def _as_vt(self, ts: np.ndarray) -> np.ndarray:
        arr = np.asarray(ts, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"timeseries must be 2D, found {arr.shape}")
        if self._n_regions is not None:
            if arr.shape[0] == self._n_regions:
                return arr
            if arr.shape[1] == self._n_regions:
                return arr.T
        if arr.shape[0] > arr.shape[1]:
            return arr.T
        return arr

    def fit(
        self,
        connectomes: np.ndarray,
        cfg: Mapping[str, Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> "BrainSync":
        mats = np.asarray(connectomes, dtype=np.float64)
        if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
            raise ValueError(
                f"connectomes must be (S,R,R) with square R, found {mats.shape}"
            )
        self._n_regions = int(mats.shape[1])
        extra = dict(extra or {})
        timeseries = extra.get("timeseries_run1", extra.get("timeseries"))
        if timeseries is None:
            raise ValueError(
                "BrainSync requires timeseries; use harness with data['timeseries_run1']"
            )

        ts_stack = self._stack_timeseries(timeseries)
        ref_raw = extra.get("reference_timeseries", ts_stack[0])
        ref_vt = self._as_vt(ref_raw)
        if ref_vt.shape[0] != self._n_regions:
            raise ValueError(
                f"BrainSync reference has V={ref_vt.shape[0]} but connectomes have "
                f"R={self._n_regions}; region-resolution timeseries required "
                "(or space='vertex' with V==R)"
            )
        self._reference_vt = ref_vt

        self._Q_list = []
        self._aligned_by_key = {}
        aligned: list[np.ndarray] = []
        for s, ts in enumerate(ts_stack):
            X_vt = self._as_vt(ts)
            if X_vt.shape[0] != ref_vt.shape[0]:
                raise ValueError(
                    f"BrainSync subject timeseries V={X_vt.shape[0]} != reference "
                    f"V={ref_vt.shape[0]}"
                )
            if self.space == "vertex":
                Q = vertex_brainsync_Q(X_vt, ref_vt)
                X_al = Q.T @ X_vt
            else:
                Q = brainsync_Q(X_vt, ref_vt)
                X_al = X_vt @ Q
            self._Q_list.append(Q)
            C_al = symmetrize_zero_diag(_connectome_from_ts(X_al))
            if C_al.shape[0] != self._n_regions:
                raise ValueError(
                    f"BrainSync recomputed connectome {C_al.shape} != "
                    f"(R,R)=({self._n_regions},{self._n_regions})"
                )
            aligned.append(C_al)
            self._aligned_by_key[np.ascontiguousarray(mats[s]).tobytes()] = C_al

        self._aligned = np.stack(aligned, axis=0) if aligned else None
        self._Q = self._Q_list[0] if self._Q_list else None
        self.meta.update(
            n_regions=self._n_regions,
            n_timeseries=len(self._Q_list),
            device="cpu",
        )
        return self

    @property
    def Q(self) -> np.ndarray | None:
        return self._Q

    @property
    def aligned_connectomes(self) -> np.ndarray | None:
        return self._aligned

    def sync_timeseries(self, timeseries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(Q, X_aligned)`` with contract shapes ``(T, T)`` / ``(V, T)``."""
        if self._reference_vt is None or self._n_regions is None:
            raise RuntimeError("BrainSync must be fitted before sync_timeseries()")
        X_vt = self._as_vt(timeseries)
        Y_vt = self._reference_vt
        if self.space == "vertex":
            Q = vertex_brainsync_Q(X_vt, Y_vt)
            return Q, Q.T @ X_vt
        Q = brainsync_Q(X_vt, Y_vt)
        return Q, X_vt @ Q

    def transform(self, connectome: np.ndarray) -> np.ndarray:
        if self._n_regions is None:
            raise RuntimeError("BrainSync must be fitted before transform()")
        C = np.asarray(connectome, dtype=np.float64)
        if C.shape != (self._n_regions, self._n_regions):
            raise ValueError(
                f"connectome must be ({self._n_regions},{self._n_regions}), found {C.shape}"
            )
        key = np.ascontiguousarray(C).tobytes()
        if key in self._aligned_by_key:
            return self._aligned_by_key[key].copy()
        if self.space == "vertex" and self._Q is not None and self._Q.shape == C.shape:
            return to_connectome_transform(self._Q, C)
        raise RuntimeError(
            "BrainSync transform(connectome) needs timeseries to recompute connectivity "
            "for time-domain Q; use harness/run_baseline with timeseries_run1 "
            "(transform_all) or space='vertex' with V==R"
        )

    def transform_all(self, connectomes: np.ndarray, **kwargs: Any) -> np.ndarray:
        mats = np.asarray(connectomes, dtype=np.float64)
        extra_ts = kwargs.get("timeseries_run1", kwargs.get("timeseries"))
        if extra_ts is not None and self._reference_vt is not None:
            ts_stack = self._stack_timeseries(extra_ts)
            out = []
            for ts in ts_stack:
                _, X_al = self.sync_timeseries(ts)
                out.append(symmetrize_zero_diag(_connectome_from_ts(X_al)))
            aligned = np.stack(out, axis=0)
            if aligned.shape[1] != self._n_regions:
                raise ValueError(
                    f"aligned connectomes {aligned.shape} do not match R={self._n_regions}"
                )
            self._aligned = aligned
            return aligned
        if self._aligned is not None and self._aligned.shape[0] == mats.shape[0]:
            return self._aligned.copy()
        return np.stack(
            [self.transform(mats[s]) for s in range(mats.shape[0])], axis=0
        )


class _BrainSyncVertex(BrainSync):
    def __init__(self) -> None:
        super().__init__(space="vertex")


register_baseline("brainsync", BrainSync)
register_baseline("brainsync_vertex", _BrainSyncVertex)
