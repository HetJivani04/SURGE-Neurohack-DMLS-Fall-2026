"""``manifest.json``: what is needed to reproduce and diagnose one run.

Data contract. W0 owns the contract's version (``data.contract_version``, from
``configs/paths.yaml``) and records it in every resolved config and its digest in
``manifest.json``; W1 implements the contract in ``src/trajot/io/contract.py``. **Freezing
this contract is the single highest-priority deliverable in the whole project, because it
unblocks W2 and W3 simultaneously.** ``CONTRACT_SPEC`` below is the pinned definition.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trajot.config import Config

_CHUNK = 1 << 20

CONTRACT_SPEC = {
    "file": "<data_root>/derivatives/trajot/sub-<id>_run-<r>.npz",
    "npz_keys": {
        "connectivity": ["(R, R)", "float64"],  # Fisher-z region-to-region functional connectivity, symmetric
        "timeseries": ["(V, T)", "float32"],    # preprocessed BOLD per vertex
        "embedding": ["(V, d)", "float32"],     # diffusion-map coordinates
        "features": ["(V, F)", "float32"],      # curvature and sulcal depth stacked
        "coords": ["(V, 3)", "float32"],        # vertex coordinates
        "tr": ["scalar", "float64"],            # repetition time in seconds
        "n_volumes": ["scalar", "int"],         # volumes actually present
        "subject_id": ["scalar", "str"],        # BIDS subject label
        "run_id": ["scalar", "str"],            # BIDS run label
    },
    "index": "<data_root>/derivatives/trajot/manifest.parquet",  # one row per subject-run
    "index_columns": {
        "subject_id": "str",
        "run_id": "str",
        "path": "str",
        "n_volumes": "int32",
        "tr": "float64",
        "n_regions": "int32",
        "n_vertices": "int32",
        "qc_pass": "bool",
        "contract_version": "str",
    },
}


def contract_digest(contract_version: str) -> str:
    """SHA-256 over the pinned contract spec and its version."""
    blob = json.dumps({"version": contract_version, "spec": CONTRACT_SPEC},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def make_run_id(experiment: str, cfg_hash: str, utc: datetime | None = None) -> str:
    """``<experiment>__<cfg_hash[:8]>__<YYYYmmddTHHMMSSZ>``."""
    utc = _utc(utc or datetime.now(timezone.utc))
    return f"{experiment}__{cfg_hash[:8]}__{utc:%Y%m%dT%H%M%SZ}"


def git_commit() -> str:
    """Short SHA via ``git rev-parse --short HEAD``, or ``"unknown"`` outside a repository."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def package_versions() -> dict[str, str]:
    """Python version plus numpy, scipy, torch, joblib, pandas, pot."""
    import platform

    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "torch", "joblib", "pandas", "pot"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def hash_data_root(root: Path, glob: str = "*.npz", max_bytes: int = 2**31) -> str:
    """Streaming SHA-256 over file paths and contents under ``root``, sorted by path.

    Reads in 1 MiB chunks, so it never holds more than a few MB. Paths are taken relative
    to ``root``. Once ``max_bytes`` of content has been read, the remaining files
    contribute their path and size only. Returns ``"missing"`` if ``root`` does not exist.
    """
    root = Path(root)
    if not root.exists():
        return "missing"
    digest = hashlib.sha256()
    budget = max_bytes
    for path in sorted(p for p in root.glob(glob) if p.is_file()):
        size = path.stat().st_size
        digest.update(f"{path.relative_to(root).as_posix()}\0{size}\0".encode())
        if size > budget:
            continue
        with path.open("rb") as fh:
            while chunk := fh.read(_CHUNK):
                digest.update(chunk)
        budget -= size
    return digest.hexdigest()


def write_manifest(
    run_dir: Path,
    cfg: "Config",
    *,
    seed: int,
    operator: str,
    n_jobs: int,
    threads: dict[str, str],
    data_hash: str,
    start_utc: datetime,
    end_utc: datetime | None,
) -> Path:
    """Write ``<run_dir>/manifest.json`` and return its path."""
    stamp = lambda dt: None if dt is None else _utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    manifest = {
        "experiment": cfg.experiment,
        "config": cfg.raw,
        "config_hash": cfg.hash,
        "git_commit": git_commit(),
        "data_hash": data_hash,
        "contract_digest": contract_digest(cfg.get("data.contract_version")),
        "seed": seed,
        "package_versions": package_versions(),
        "operator": operator,
        "n_jobs": n_jobs,
        "threads": threads,
        "start_utc": stamp(start_utc),
        "end_utc": stamp(end_utc),
    }
    path = Path(run_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
