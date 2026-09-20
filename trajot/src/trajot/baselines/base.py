from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Baseline(Protocol):
    def fit(self, connectomes: np.ndarray, cfg: Mapping[str, Any]) -> "Baseline":
        ...

    def transform(self, connectome: np.ndarray) -> np.ndarray:
        ...


@dataclass
class BaselineResult:
    name: str
    transforms: np.ndarray
    aligned_features: np.ndarray
    meta: dict[str, Any]
    device: str


BASELINE_REGISTRY: dict[str, type[Baseline]] = {}


def register_baseline(name: str, cls: type[Baseline]) -> None:
    BASELINE_REGISTRY[name.lower()] = cls


def get_baseline(name: str) -> Baseline:
    key = name.lower()
    if key not in BASELINE_REGISTRY:
        raise KeyError(f"Unknown baseline {name!r}. Known: {sorted(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[key]()


class FakeBaseline:
    """Synthetic baseline used to validate evaluation metrics before real methods.

    It applies a deterministic orthogonal transform and permutation.
    """

    def __init__(self) -> None:
        self._Q: np.ndarray | None = None
        self._perm: np.ndarray | None = None
        self.device = "cpu"

    def fit(self, connectomes: np.ndarray, cfg: Mapping[str, Any]) -> "FakeBaseline":
        connectomes = np.asarray(connectomes, dtype=np.float64)
        if connectomes.ndim != 3:
            raise ValueError(f"connectomes must be (S,R,R), found {connectomes.shape}")

        R = connectomes.shape[1]
        seed = int(cfg.get("run.seed", 0)) if isinstance(cfg, Mapping) else 0
        rng = np.random.default_rng(seed)

        M = rng.normal(size=(R, R))
        Q, _ = np.linalg.qr(M)
        perm = rng.permutation(R)

        self._Q = Q.astype(np.float64)
        self._perm = perm.astype(np.int64)
        return self

    def transform(self, connectome: np.ndarray) -> np.ndarray:
        if self._Q is None or self._perm is None:
            raise RuntimeError("FakeBaseline must be fitted before transform()")

        C = np.asarray(connectome, dtype=np.float64)
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError(f"connectome must be square (R,R), found {C.shape}")

        C1 = self._Q.T @ C @ self._Q
        P = np.eye(C.shape[0], dtype=np.float64)[self._perm]
        C2 = P.T @ C1 @ P

        C2 = 0.5 * (C2 + C2.T)
        np.fill_diagonal(C2, 0.0)
        return C2
