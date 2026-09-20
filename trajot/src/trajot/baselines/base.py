from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Dotted-key lookup that works for Config, nested dicts, and flat dicts."""
    if cfg is None:
        return default
    get = getattr(cfg, "get", None)
    if callable(get):
        sentinel = object()
        try:
            val = get(key, sentinel)
        except TypeError:
            val = sentinel
        if val is not sentinel:
            return val
    if isinstance(cfg, Mapping):
        if key in cfg:
            return cfg[key]
        node: Any = cfg
        for part in key.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node
    return default


class CfgView:
    """Adapter exposing dotted ``get`` over Config / nested / flat mappings."""

    def __init__(self, obj: Any = None) -> None:
        self._obj = {} if obj is None else obj

    def get(self, key: str, default: Any = None) -> Any:
        return cfg_get(self._obj, key, default)

    def __getitem__(self, key: str) -> Any:
        out = cfg_get(self._obj, key, None)
        if out is None:
            raise KeyError(key)
        return out


def upper_triangle_features(connectomes: np.ndarray) -> np.ndarray:
    """Flatten each (R, R) connectome to its strict upper triangle ``(S, R*(R-1)//2)`` float64."""
    mats = np.asarray(connectomes, dtype=np.float64)
    if mats.ndim == 2:
        mats = mats[None, ...]
    if mats.ndim != 3:
        raise ValueError(f"connectomes must be (S,R,R), found {mats.shape}")
    R = mats.shape[1]
    iu = np.triu_indices(R, k=1)
    return np.ascontiguousarray(mats[:, iu[0], iu[1]], dtype=np.float64)


def symmetrize_zero_diag(C: np.ndarray) -> np.ndarray:
    out = np.asarray(C, dtype=np.float64)
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 0.0)
    return out


@runtime_checkable
class Baseline(Protocol):
    def fit(
        self,
        connectomes: np.ndarray,
        cfg: Mapping[str, Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> "Baseline":
        ...

    def transform(self, connectome: np.ndarray) -> np.ndarray:
        ...


@dataclass
class BaselineResult:
    name: str
    transforms: np.ndarray
    aligned_features: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"


BASELINE_REGISTRY: dict[str, type] = {}


def register_baseline(name: str, cls: type) -> None:
    BASELINE_REGISTRY[name.lower()] = cls


def get_baseline(name: str) -> Any:
    key = str(name).lower()
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
        self.meta: dict[str, Any] = {"algorithm": "fake_orthogonal_perm", "device": "cpu"}

    def fit(
        self,
        connectomes: np.ndarray,
        cfg: Mapping[str, Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> "FakeBaseline":
        connectomes = np.asarray(connectomes, dtype=np.float64)
        if connectomes.ndim != 3:
            raise ValueError(f"connectomes must be (S,R,R), found {connectomes.shape}")

        R = connectomes.shape[1]
        seed = int(cfg_get(cfg, "run.seed", 0) or 0)
        rng = np.random.default_rng(seed)

        M = rng.normal(size=(R, R))
        Q, _ = np.linalg.qr(M)
        perm = rng.permutation(R)

        self._Q = Q.astype(np.float64)
        self._perm = perm.astype(np.int64)
        self.meta.update(n_regions=R, device="cpu")
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
        return symmetrize_zero_diag(C2)

    def transform_all(self, connectomes: np.ndarray, **_: Any) -> np.ndarray:
        mats = np.asarray(connectomes, dtype=np.float64)
        return np.stack([self.transform(mats[s]) for s in range(mats.shape[0])], axis=0)


register_baseline("fake", FakeBaseline)
