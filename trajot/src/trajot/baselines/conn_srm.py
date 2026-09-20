from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Baseline, register_baseline


class ConnSRM(Baseline):
    """Low-rank shared-space baseline via eigenvalue shrinkage in connectome space."""

    def __init__(self) -> None:
        self._rank: int | None = None
        self._shape: tuple[int, int] | None = None
        self.device = "cpu"

    def fit(self, connectomes: np.ndarray, cfg: Mapping[str, Any]) -> "ConnSRM":
        mats = np.asarray(connectomes, dtype=np.float64)
        if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
            raise ValueError(f"connectomes must be (S,R,R) with square R, found {mats.shape}")

        default_rank = min(32, mats.shape[1])
        model_cfg = cfg.get("model", {}) if isinstance(cfg, Mapping) else {}
        rank = int(model_cfg.get("r", default_rank)) if isinstance(model_cfg, Mapping) else default_rank
        rank = max(1, min(rank, mats.shape[1]))

        self._rank = rank
        self._shape = (int(mats.shape[1]), int(mats.shape[2]))
        return self

    def transform(self, connectome: np.ndarray) -> np.ndarray:
        if self._rank is None or self._shape is None:
            raise RuntimeError("ConnSRM must be fitted before transform()")

        C = np.asarray(connectome, dtype=np.float64)
        if C.shape != self._shape:
            raise ValueError(f"connectome must be {self._shape}, found {C.shape}")

        vals, vecs = np.linalg.eigh(0.5 * (C + C.T))
        idx = np.argsort(np.abs(vals))[::-1]
        keep = idx[: self._rank]
        recon = (vecs[:, keep] * vals[keep]) @ vecs[:, keep].T
        recon = 0.5 * (recon + recon.T)
        np.fill_diagonal(recon, 0.0)
        return recon


register_baseline("conn_srm", ConnSRM)
