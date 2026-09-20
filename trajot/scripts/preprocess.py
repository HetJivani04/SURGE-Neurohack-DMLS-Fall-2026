from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
from scipy.spatial.distance import cdist

from trajot.geometry.connectivity import connectivity, parcellate, rank_factorize
from trajot.geometry.diffusion import diffusion_map
from trajot.io.contract import (
    SCHEMA_VERSION,
    read_subject_run,
    subject_run_path,
    write_manifest,
    write_subject_run,
)
from trajot.io.dataset import RunSpec, list_bold_runs
from trajot.io.preprocess import preprocess_run


def _parse_csv(values: str | None) -> set[str] | None:
    if values is None:
        return None
    out = {chunk.strip() for chunk in values.split(",") if chunk.strip()}
    return out or None


def _parallel_map(func, items: list[RunSpec], n_jobs: int) -> list[Path]:
    try:
        from trajot.runlog.parallel import parallel_map  # type: ignore

        return list(parallel_map(func, items, n_jobs=n_jobs))
    except Exception:
        return [func(item) for item in items]


def _filter_specs(
    specs: list[RunSpec],
    *,
    subjects: set[str] | None,
    runs: set[str] | None,
) -> list[RunSpec]:
    out = []
    for spec in specs:
        if subjects is not None and spec.subject_id not in subjects:
            continue
        if runs is not None and spec.run_id not in runs:
            continue
        out.append(spec)
    return out


def _derive_surface_features(coords: np.ndarray, k: int = 8) -> np.ndarray:
    """Build two anatomical feature channels: curvature proxy and sulcal-depth proxy."""

    xyz = coords.astype(np.float64)
    if xyz.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if xyz.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)

    dmat = cdist(xyz, xyz, metric="euclidean")
    k_eff = min(max(1, k), xyz.shape[0] - 1)
    nn = np.argpartition(dmat, kth=k_eff, axis=1)[:, 1 : k_eff + 1]

    local_centroid = xyz[nn].mean(axis=1)
    curvature_proxy = np.linalg.norm(xyz - local_centroid, axis=1, keepdims=True)

    centered = xyz - xyz.mean(axis=0, keepdims=True)
    depth_proxy = np.linalg.norm(centered, axis=1, keepdims=True)

    features = np.concatenate([curvature_proxy, depth_proxy], axis=1)
    return features.astype(np.float32)


def _vertex_affinity_from_timeseries(timeseries: np.ndarray) -> np.ndarray:
    ts = timeseries.astype(np.float64)
    ts = ts - ts.mean(axis=1, keepdims=True)
    std = ts.std(axis=1, keepdims=True)
    std[std == 0.0] = 1.0
    ts = ts / std

    corr = (ts @ ts.T) / max(ts.shape[1] - 1, 1)
    corr = np.clip(corr, -1.0, 1.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return np.maximum(corr, 0.0)


def _factor_path(npz_path: Path) -> Path:
    return npz_path.with_suffix("").with_name(npz_path.stem + "_factor.npz")


def _write_one(root: Path, spec: RunSpec, force: bool, cfg: SimpleNamespace) -> Path:
    out_path = subject_run_path(root, spec.subject_id, spec.run_id)
    factor_path = _factor_path(out_path)
    if out_path.exists() and factor_path.exists() and not force:
        return out_path

    timeseries, coords, tr = preprocess_run(spec, cfg)

    n_vertices, _ = timeseries.shape
    n_regions = int(cfg.n_regions)
    labels = np.floor(np.linspace(0, n_regions - 1, num=n_vertices)).astype(int)

    region_ts = parcellate(timeseries, labels=labels, n_regions=n_regions)
    conn = connectivity(region_ts)

    affinity = _vertex_affinity_from_timeseries(timeseries)
    embedding, _ = diffusion_map(affinity, d=int(cfg.embedding_dim))

    features = _derive_surface_features(coords)

    factors, retained_var = rank_factorize(conn, r=int(cfg.rank))
    factor_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        factor_path,
        A=factors.astype(np.float64),
        retained_variance=np.float64(retained_var),
    )

    return write_subject_run(
        out_path,
        connectivity=conn,
        timeseries=timeseries,
        embedding=embedding.astype(np.float32),
        features=features,
        coords=coords,
        tr=tr,
        n_volumes=timeseries.shape[1],
        subject_id=spec.subject_id,
        run_id=spec.run_id,
    )


def _rebuild_manifest(root: Path) -> Path:
    rows = []
    out_dir = root / "derivatives" / "trajot"
    for npz_path in sorted(out_dir.glob("sub-*_run-*.npz")):
        d = read_subject_run(npz_path)
        rows.append(
            {
                "subject_id": d["subject_id"],
                "run_id": d["run_id"],
                "path": str(npz_path.relative_to(root)),
                "n_volumes": int(d["n_volumes"]),
                "tr": float(d["tr"]),
                "n_regions": int(np.asarray(d["connectivity"]).shape[0]),
                "n_vertices": int(np.asarray(d["timeseries"]).shape[0]),
                "qc_pass": True,
                "contract_version": SCHEMA_VERSION,
            }
        )

    return write_manifest(rows, root)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W1 preprocessing entrypoint")
    parser.add_argument("--data-root", type=Path, default=Path("./data/ds000243"))
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--runs", type=str, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    subjects = _parse_csv(args.subjects)
    runs = _parse_csv(args.runs)

    specs = list_bold_runs(args.data_root)
    specs = _filter_specs(specs, subjects=subjects, runs=runs)

    if args.dry_run:
        for spec in specs:
            print(f"would process sub-{spec.subject_id} run-{spec.run_id}")
        return 0

    # Stage-1 development policy defaults to n_jobs=1.
    effective_n_jobs = 1
    if args.n_jobs != 1:
        print("warning: n_jobs > 1 requested; forcing n_jobs=1 during stage-1")

    cfg = SimpleNamespace(
        n_regions=100,
        embedding_dim=32,
        target_vertices_per_hemi=2000,
        rank=32,
        low=0.01,
        high=0.1,
        order=2,
        gsr=True,
        chunk_size=4000,
    )

    def _runner(spec: RunSpec) -> Path:
        return _write_one(args.data_root, spec, force=args.force, cfg=cfg)

    for path in _parallel_map(_runner, specs, n_jobs=effective_n_jobs):
        print(f"wrote {path}")

    manifest_path = _rebuild_manifest(args.data_root)
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
