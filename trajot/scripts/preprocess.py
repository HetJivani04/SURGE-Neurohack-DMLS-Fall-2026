#!/usr/bin/env python
"""W1 entry point: preprocess ds000243 subject-runs into the frozen npz data contract.

    python scripts/preprocess.py --subjects 001,002 --runs 1

One subject-run at a time (resumable: a run whose outputs all exist is skipped unless
``--force``). Per subject-run it writes, under ``<data_root>/derivatives/trajot/``:

* ``sub-<id>_run-<r>.npz``          the contract file (``trajot.io.contract``)
* ``sub-<id>_run-<r>_geometry.npz`` ``A`` (R, r) factor of the connectome, ``retained_variance``,
                                    ``mass`` (V,) vertex masses ``mu_s`` (uniform over valid vertices)
* ``sub-<id>_run-<r>_qc.json``      the run's QC record (motion, registration, denoising, band-pass)

and, once, ``template_geometry.npz`` (the fsaverage anatomical cost ``M^0`` ``(V, V)``, faces,
coordinates), then rebuilds ``manifest.parquet``. Each invocation is also recorded as a run
(``runs/<run_id>/``: manifest, log, and the band-pass check plot in ``artifacts/``).
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Iterable

from trajot.config import load_config
from trajot.runlog.parallel import parallel_map, setup_threads

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "10_ours_full.yaml"
PEAK_GIB_PER_WORKER = 2.0  # measured peak RSS is 1.3 GiB on the longest (724-volume) run
_CONTRACT_FILE = re.compile(r"^sub-(?P<subject>[A-Za-z0-9]+)_run-(?P<run>[A-Za-z0-9]+)\.npz$")


def _preflight(n_jobs: int, log: Any) -> None:
    """Report available RAM and abort with a clear message rather than be killed by the OS."""

    import psutil

    available = psutil.virtual_memory().available / 2**30
    required = PEAK_GIB_PER_WORKER * n_jobs
    log(f"available RAM {available:.1f} GiB; about {required:.1f} GiB needed for {n_jobs} worker(s)")
    if available < required:
        raise MemoryError(f"only {available:.1f} GiB of RAM is available but about {required:.1f} GiB is needed: "
                          "close other applications or lower --n-jobs")


def _parse_csv(values: str | None) -> set[str] | None:
    if values is None:
        return None
    out = {chunk.strip() for chunk in values.split(",") if chunk.strip()}
    return out or None


def _filter_specs(specs: list[Any], *, subjects: set[str] | None, runs: set[str] | None) -> list[Any]:
    out = []
    for spec in specs:
        if subjects is not None and spec.subject_id not in subjects:
            continue
        if runs is not None and spec.run_id not in runs:
            continue
        out.append(spec)
    return out


def _geometry_path(npz_path: Path) -> Path:
    return npz_path.with_name(npz_path.stem + "_geometry.npz")


def _is_done(root: Path, spec: Any) -> bool:
    from trajot.io.contract import subject_run_path
    from trajot.io.preprocess import qc_path

    npz = subject_run_path(root, spec.subject_id, spec.run_id)
    return npz.exists() and _geometry_path(npz).exists() and qc_path(root, spec.subject_id, spec.run_id).exists()


def _region_series(timeseries: Any, labels: Any, valid: Any, n_regions: int) -> Any:
    """``(R, T)`` float32 region means over the valid vertices (``parcellate``).

    A region with no valid vertex (a run whose field of view misses that part of the cortex) stays
    all zero, so its connectome row and column are zero; the QC record lists such regions.
    """

    import numpy as np

    from trajot.geometry.connectivity import parcellate

    present = np.unique(labels[valid])
    lookup = np.full(n_regions, -1)
    lookup[present] = np.arange(present.size)
    region_ts = np.zeros((n_regions, timeseries.shape[1]), dtype=np.float32)
    region_ts[present] = parcellate(timeseries[valid], lookup[labels[valid]], present.size)
    return region_ts


def _write_one(spec: Any, cfg: Any, force: bool) -> Path:
    """Preprocess one subject-run and write its outputs; the contract npz is written last."""

    import numpy as np

    from trajot.geometry.connectivity import connectivity, rank_factorize
    from trajot.geometry.cost import vertex_mass
    from trajot.geometry.diffusion import diffusion_map
    from trajot.io.contract import subject_run_path, write_subject_run
    from trajot.io.preprocess import load_surface_template, preprocess_run

    root = Path(cfg.data_root)
    out_path = subject_run_path(root, spec.subject_id, spec.run_id)
    if _is_done(root, spec) and not force:
        return out_path

    timeseries, coords, tr = preprocess_run(spec, cfg)
    surface = load_surface_template(cfg.get("preprocess.surface"), int(cfg.get("preprocess.n_regions")))
    n_vertices = timeseries.shape[0]
    valid = timeseries.std(axis=1) > 0

    conn = connectivity(_region_series(timeseries, surface.labels, valid, surface.n_regions))
    factors, retained = rank_factorize(conn, r=int(cfg.get("model.r")))

    z = timeseries[valid].astype(np.float64)
    z -= z.mean(axis=1, keepdims=True)
    z /= z.std(axis=1, keepdims=True)
    affinity = (z @ z.T) / max(z.shape[1] - 1, 1)  # vertex-level C_s = corr(X_s), transient
    del z
    embedding = np.zeros((n_vertices, int(cfg.get("model.d"))), dtype=np.float32)
    embedding[valid] = diffusion_map(affinity, d=embedding.shape[1])[0].astype(np.float32)
    del affinity

    geometry_path = _geometry_path(out_path)
    np.savez_compressed(geometry_path, A=factors, retained_variance=np.float64(retained),
                        mass=vertex_mass(n_vertices, valid))

    partial_path = out_path.with_name(out_path.stem + ".partial.npz")
    write_subject_run(partial_path, connectivity=conn, timeseries=timeseries, embedding=embedding,
                      features=surface.features, coords=coords, tr=tr, n_volumes=timeseries.shape[1],
                      subject_id=spec.subject_id, run_id=spec.run_id)
    os.replace(partial_path, out_path)
    return out_path


def _write_template_geometry(root: Path, cfg: Any, force: bool) -> Path:
    """``template_geometry.npz``: the anatomical cost ``M^0`` (V, V) float64 on the fsaverage surface.

    Without a per-subject surface reconstruction every subject shares this template mesh, so
    the cost is identical across subjects and is written once instead of per subject.
    """

    import numpy as np

    from trajot.geometry.cost import anatomical_cost
    from trajot.io.preprocess import load_surface_template

    path = root / "derivatives" / "trajot" / "template_geometry.npz"
    if path.exists() and not force:
        return path
    surface = load_surface_template(cfg.get("preprocess.surface"), int(cfg.get("preprocess.n_regions")))
    cost = anatomical_cost(surface.coords.astype(np.float64), surface.faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, anatomical_cost=cost, faces=surface.faces, coords=surface.coords,
             n_left=np.int64(surface.n_left), region_labels=surface.labels)
    return path


def _rebuild_manifest(root: Path) -> Path:
    import numpy as np

    from trajot.io.contract import SCHEMA_VERSION, read_subject_run, write_manifest
    from trajot.io.preprocess import qc_path

    rows = []
    out_dir = root / "derivatives" / "trajot"
    for npz_path in sorted(p for p in out_dir.glob("sub-*_run-*.npz") if _CONTRACT_FILE.match(p.name)):
        d = read_subject_run(npz_path)
        qc = qc_path(root, str(d["subject_id"]), str(d["run_id"]))
        rows.append(
            {
                "subject_id": d["subject_id"],
                "run_id": d["run_id"],
                "path": str(npz_path.relative_to(root)),
                "n_volumes": int(d["n_volumes"]),
                "tr": float(d["tr"]),
                "n_regions": int(np.asarray(d["connectivity"]).shape[0]),
                "n_vertices": int(np.asarray(d["timeseries"]).shape[0]),
                "qc_pass": bool(json.loads(qc.read_text())["qc_pass"]) if qc.exists() else False,
                "contract_version": SCHEMA_VERSION,
            }
        )

    return write_manifest(rows, root)


def _bandpass_check(root: Path, spec: Any, artifacts_dir: Path) -> dict[str, float]:
    """Plot the spot-checked run's spectrum before and after the band-pass into ``artifacts_dir``."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from trajot.io.preprocess import qc_path

    record = json.loads(qc_path(root, spec.subject_id, spec.run_id).read_text())
    band = record["bandpass"]
    freqs = band["freqs_hz"]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(freqs, band["psd_before"], label="before band-pass")
    ax.semilogy(freqs, band["psd_after"], label="after band-pass")
    ax.axvspan(band["low"], band["high"], color="0.9", label=f"{band['low']}-{band['high']} Hz")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("mean vertex power spectral density")
    ax.set_title(f"sub-{spec.subject_id} run-{spec.run_id}: in-band power kept {band['in_band_power_ratio']:.2f}, "
                 f"out-of-band {band['out_of_band_power_ratio']:.1e}", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(artifacts_dir / "bandpass_check.png", dpi=120)
    plt.close(fig)
    return {"in_band_power_ratio": band["in_band_power_ratio"], "out_of_band_power_ratio": band["out_of_band_power_ratio"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="W1 preprocessing entry point")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="experiment config supplying the preprocess.* keys and data.root (default: %(default)s)")
    parser.add_argument("--subjects", type=str, default=None, help="comma-separated subject ids, e.g. 001,002")
    parser.add_argument("--runs", type=str, default=None, help="comma-separated run ids, e.g. 1")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="reprocess runs whose outputs already exist")
    parser.add_argument("--dry-run", action="store_true", help="list the runs that would be processed")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config, {"experiment": "preprocess"})
    n_jobs = args.n_jobs
    threads = setup_threads("serial" if n_jobs == 1 else "outer", n_jobs)

    # After setup_threads: these import numpy / pandas.
    from trajot.io import contract, dataset
    from trajot.runlog.logging import RunLogger
    from trajot.runlog.manifest import git_commit, hash_data_root, make_run_id, write_manifest
    from trajot.runlog.registry import append_run

    if cfg.get("data.contract_version") != contract.SCHEMA_VERSION:
        raise contract.ContractError(
            f"configs/paths.yaml pins contract_version {cfg.get('data.contract_version')!r} but this code writes "
            f"{contract.SCHEMA_VERSION!r}: update contract_version in configs/paths.yaml")

    root = Path(cfg.data_root)
    specs = dataset.list_bold_runs(root)
    selected = _filter_specs(specs, subjects=_parse_csv(args.subjects), runs=_parse_csv(args.runs))

    if args.dry_run:
        for spec in selected:
            print(f"would process sub-{spec.subject_id} run-{spec.run_id}")
        return 0

    start_utc = datetime.now(timezone.utc)
    run_id = make_run_id(cfg.experiment, cfg.hash, start_utc)
    run_dir = RUNS_DIR / run_id
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    seed, operator = cfg.get("run.seed"), getpass.getuser()
    data_hash = hash_data_root(root, glob="sub-*/func/*_bold.nii.gz")
    manifest = dict(seed=seed, operator=operator, n_jobs=n_jobs, threads=threads,
                    data_hash=data_hash, start_utc=start_utc)
    write_manifest(run_dir, cfg, **manifest, end_utc=None)

    logger = RunLogger(run_dir, run_id)
    try:
        _preflight(n_jobs, logger.log)
        todo = [s for s in selected if args.force or not _is_done(root, s)]
        logger.log(f"{len(selected)} runs selected, {len(todo)} to process, n_jobs={n_jobs}")
        for path in parallel_map(partial(_write_one, cfg=cfg, force=args.force), todo, n_jobs=n_jobs):
            print(f"wrote {path}")

        template = _write_template_geometry(root, cfg, force=args.force)
        print(f"wrote {template}")

        manifest_path = _rebuild_manifest(root)
        table = contract.read_manifest(root)
        subjects = contract.two_run_subjects(table, strict=args.subjects is None and args.runs is None)
        print(f"wrote {manifest_path}: {len(table)} rows, {len(subjects)} two-run subjects with equal n_volumes")

        spot = next((s for s in (todo or selected) if _is_done(root, s)), None)
        if spot is not None:
            ratios = _bandpass_check(root, spot, run_dir / "artifacts")
            logger.log(f"band-pass check sub-{spot.subject_id} run-{spot.run_id}: {ratios}")
    finally:
        logger.close()

    end_utc = datetime.now(timezone.utc)
    write_manifest(run_dir, cfg, **manifest, end_utc=end_utc)
    append_run(RUNS_DIR / "index.csv", {
        "run_id": run_id, "experiment": cfg.experiment, "config_hash": cfg.hash,
        "git_commit": git_commit(), "data_hash": data_hash, "seed": seed, "operator": operator,
        "n_jobs": n_jobs, "start_utc": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_utc": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "status": "ok",
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
