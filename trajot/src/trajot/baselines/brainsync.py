from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import Baseline, register_baseline


class BrainSync(Baseline):
    """Orthogonal Procrustes alignment to cohort mean connectome."""

    def __init__(self) -> None:
        self._template: np.ndarray | None = None
        self.device = "cpu"

    def fit(self, connectomes: np.ndarray, cfg: Mapping[str, Any]) -> "BrainSync":
        mats = np.asarray(connectomes, dtype=np.float64)
        if mats.ndim != 3:
            raise ValueError(f"connectomes must be (S,R,R), found {mats.shape}")
        if mats.shape[1] != mats.shape[2]:
            raise ValueError(f"connectomes must be square per subject, found {mats.shape}")

        template = mats.mean(axis=0)
        template = 0.5 * (template + template.T)
        np.fill_diagonal(template, 0.0)
        self._template = template
        return self

    def transform(self, connectome: np.ndarray) -> np.ndarray:
        if self._template is None:
            raise RuntimeError("BrainSync must be fitted before transform()")
        C = np.asarray(connectome, dtype=np.float64)
        if C.shape != self._template.shape:
            raise ValueError(f"connectome must be {self._template.shape}, found {C.shape}")

        U, _, Vt = np.linalg.svd(C @ self._template.T, full_matrices=False)
        Q = U @ Vt
        aligned = Q.T @ C @ Q
        aligned = 0.5 * (aligned + aligned.T)
        np.fill_diagonal(aligned, 0.0)
        return aligned


register_baseline("brainsync", BrainSync)
