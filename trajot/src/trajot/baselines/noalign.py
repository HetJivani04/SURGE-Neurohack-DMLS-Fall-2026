from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Baseline, register_baseline


class NoAlign(Baseline):
    """Identity baseline: preserves anatomical correspondence exactly."""

    def __init__(self) -> None:
        self._n_regions: int | None = None
        self.device = "cpu"

    def fit(self, connectomes: np.ndarray, cfg: Mapping[str, Any]) -> "NoAlign":
        mats = np.asarray(connectomes, dtype=np.float64)
        if mats.ndim != 3:
            raise ValueError(f"connectomes must be (S,R,R), found {mats.shape}")
        if mats.shape[1] != mats.shape[2]:
            raise ValueError(f"connectomes must be square per subject, found {mats.shape}")
        self._n_regions = int(mats.shape[1])
        return self

    def transform(self, connectome: np.ndarray) -> np.ndarray:
        if self._n_regions is None:
            raise RuntimeError("NoAlign must be fitted before transform()")
        C = np.asarray(connectome, dtype=np.float64)
        if C.shape != (self._n_regions, self._n_regions):
            raise ValueError(
                f"connectome must be ({self._n_regions},{self._n_regions}), found {C.shape}"
            )
        return C.copy()


register_baseline("noalign", NoAlign)
