from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Baseline, register_baseline


class AblatedModel(Baseline):
    """Bridge baseline for W2 outputs.

    This class is intentionally lightweight while W2 training artifacts are in flight:
    it offers the same fit/transform contract and acts as identity when no trained
    operator is present.
    """

    def __init__(self) -> None:
        self._shape: tuple[int, int] | None = None
        self.device = "cpu"

    def fit(self, connectomes: np.ndarray, cfg: Mapping[str, Any]) -> "AblatedModel":
        mats = np.asarray(connectomes, dtype=np.float64)
        if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
            raise ValueError(f"connectomes must be (S,R,R) with square R, found {mats.shape}")
        self._shape = (int(mats.shape[1]), int(mats.shape[2]))
        return self

    def transform(self, connectome: np.ndarray) -> np.ndarray:
        if self._shape is None:
            raise RuntimeError("AblatedModel must be fitted before transform()")
        C = np.asarray(connectome, dtype=np.float64)
        if C.shape != self._shape:
            raise ValueError(f"connectome must be {self._shape}, found {C.shape}")
        return C.copy()


register_baseline("ablated", AblatedModel)
