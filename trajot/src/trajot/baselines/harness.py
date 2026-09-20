from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import (
    BaselineResult,
    get_baseline,
    upper_triangle_features,
)


def run_baseline(
    name: str,
    data: Mapping[str, Any],
    folds: Any = None,
    cfg: Any = None,
) -> BaselineResult:
    """Fit and transform one registered baseline through the uniform interface.

    Parameters
    ----------
    name:
        Registry key (``noalign``, ``brainsync``, ``fugw``, ``conn_srm``,
        ``ours_full``, ``ours_ablated``, experiment filenames as aliases).
    data:
        Mapping with at least ``connectomes`` ``(S, R, R)`` float64. Optional keys
        consumed via ``extra``: ``timeseries_run1`` / ``timeseries``,
        ``timeseries_run2``, ``embeddings``, ``subjects``, ``data_root``,
        ``run_dir``, ``reference_timeseries``.
    folds:
        Optional fold assignment (eval ownership); stored in ``meta`` when present.
    cfg:
        Config / mapping forwarded to ``fit``. Falls back to ``data['cfg']``.

    Returns
    -------
    BaselineResult
        ``transforms`` ``(S, R, R)`` float64; ``aligned_features`` is the flattened
        strict upper triangle ``(S, R*(R-1)//2)`` float64 for connectome methods.
    """
    if "connectomes" not in data:
        raise KeyError("run_baseline data must include 'connectomes' (S,R,R)")
    connectomes = np.asarray(data["connectomes"], dtype=np.float64)
    if connectomes.ndim != 3 or connectomes.shape[1] != connectomes.shape[2]:
        raise ValueError(f"connectomes must be (S,R,R), found {connectomes.shape}")

    method = get_baseline(name)
    fit_cfg = cfg if cfg is not None else data.get("cfg", {})
    extra = {k: v for k, v in data.items() if k not in {"connectomes", "cfg"}}
    if folds is not None:
        extra.setdefault("folds", folds)

    method.fit(connectomes, fit_cfg, extra=extra)

    transform_kwargs = {
        k: extra[k]
        for k in ("timeseries_run1", "timeseries", "timeseries_run2")
        if k in extra
    }
    if hasattr(method, "transform_all"):
        transforms = np.asarray(
            method.transform_all(connectomes, **transform_kwargs), dtype=np.float64
        )
    else:
        transforms = np.stack(
            [method.transform(connectomes[s]) for s in range(connectomes.shape[0])],
            axis=0,
        ).astype(np.float64)

    if transforms.shape != connectomes.shape:
        raise ValueError(
            f"baseline {name!r} produced transforms {transforms.shape}, "
            f"expected {connectomes.shape}"
        )

    aligned_features = upper_triangle_features(transforms)
    meta = dict(getattr(method, "meta", {}) or {})
    meta.setdefault("algorithm", name)
    meta.setdefault("n_regions", int(connectomes.shape[1]))
    meta.setdefault("n_subjects", int(connectomes.shape[0]))
    meta["baseline"] = str(name).lower()
    meta["transforms_are_connectomes"] = True
    if folds is not None:
        meta["folds"] = folds
    subjects = data.get("subjects")
    if subjects is not None:
        meta["subjects"] = list(subjects)

    device = str(getattr(method, "device", "cpu") or "cpu")
    meta.setdefault("device", device)

    return BaselineResult(
        name=str(name).lower(),
        transforms=transforms,
        aligned_features=aligned_features,
        meta=meta,
        device=device,
    )


def align_features(
    result: BaselineResult,
    embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """Uniform features for downstream metrics.

    Connectome methods expose flattened upper-triangle features on
    ``BaselineResult.aligned_features``; those are returned when ``embeddings``
    is ``None``. If ``embeddings`` are supplied and ``meta['transforms_are_operators']``
    is true, each subject's embedding is pulled through the ``(R, R)`` operator.
    Otherwise embeddings are returned unchanged — connectome transforms are already
    folded into ``aligned_features``.
    """
    if result is None:
        raise ValueError("align_features requires a BaselineResult")

    if embeddings is None:
        return np.asarray(result.aligned_features, dtype=np.float64)

    emb = np.asarray(embeddings)
    if result.meta.get("transforms_are_operators", False):
        transforms = np.asarray(result.transforms, dtype=np.float64)
        if (
            emb.ndim == 3
            and transforms.ndim == 3
            and emb.shape[0] == transforms.shape[0]
            and emb.shape[1] == transforms.shape[1]
        ):
            out = np.einsum("sij,sjd->sid", transforms, emb.astype(np.float64))
            return out.astype(np.float32)
    return emb.astype(np.float32, copy=False)
