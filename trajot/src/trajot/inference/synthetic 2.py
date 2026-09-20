"""Contract-exact synthetic data with an optional planted vertex permutation.

``make_synthetic_npz`` writes ``derivatives/trajot/sub-<id>_run-{1,2}.npz`` and ``manifest.parquet`` that satisfy
W1's ``validate_subject_run`` unchanged, so W2 (and W3) can develop and test without waiting for real data.

The generative story mirrors the model. A template has ``V`` nodes, each with a geometry loading ``B_true[k]``, a
feature vector ``F_true[k]`` and a position ``coords_true[k]``. Subject ``s`` observes those nodes through a
hidden permutation ``perm[s]``: its vertex ``i`` *is* template node ``perm[s][i]`` (identity when
``planted_permutation=False``). Everything a subject reports about vertex ``i`` (time series, embedding,
features, coordinates) is a noisy function of that node, so the correct alignment is known. The two runs of a
subject share the subject's loadings but have independent temporal factors and noise (scan-rescan variation).
The ground truth is saved next to the data in ``synthetic_truth.npz``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trajot.geometry.connectivity import connectivity, parcellate
from trajot.io.contract import SCHEMA_VERSION, subject_run_path, write_manifest, write_subject_run

TRUTH_FILE = "synthetic_truth.npz"


def make_synthetic_npz(
    root: Path, n_subjects: int = 12, V: int = 200, R: int = 100, T: int = 132, d: int = 32, seed: int = 0,
    planted_permutation: bool = True,
) -> Path:
    """Write ``n_subjects`` two-run subjects with ``V`` vertices, an ``(R, R)`` connectome, and ``d``-dimensional
    embeddings under ``root``; return ``root``.

    ``perm`` (``n_subjects, V``) in ``synthetic_truth.npz`` gives, for every subject, the template node each of
    its vertices is (a random permutation when ``planted_permutation`` is true, the identity otherwise).
    Region ``k % R`` contains template node ``k``, so region-level connectomes are comparable across subjects.
    """
    if R > V:
        raise ValueError(f"R={R} regions need at least R vertices, got V={V}")
    root = Path(root)
    rng = np.random.default_rng(seed)
    r_true, n_features = 8, 2

    B_true = rng.normal(size=(V, r_true)) / np.sqrt(r_true)
    F_true = rng.normal(size=(V, d))
    G_true = rng.normal(size=(V, n_features))
    directions = rng.normal(size=(V, 3))
    coords_true = 50.0 * directions / np.linalg.norm(directions, axis=1, keepdims=True)
    node_region = np.arange(V) % R

    perms = np.stack([rng.permutation(V) if planted_permutation else np.arange(V) for _ in range(n_subjects)])
    ids = [f"{s + 1:03d}" for s in range(n_subjects)]
    rows = []
    for s, subject_id in enumerate(ids):
        node = perms[s]  # vertex i of this subject is template node node[i]
        loadings = B_true[node] * np.sqrt(r_true) + 0.3 * rng.normal(size=(V, r_true))
        anatomy = coords_true[node] + 12.0 * rng.normal(size=(V, 3))
        for run in ("1", "2"):
            timeseries = loadings @ rng.normal(size=(T, r_true)).T / np.sqrt(r_true) + rng.normal(size=(V, T))
            embedding = F_true[node] + 0.5 * rng.normal(size=(V, d))
            features = G_true[node] + 0.3 * rng.normal(size=(V, n_features))
            region_ts = parcellate(timeseries.astype(np.float32), node_region[node], R)
            path = write_subject_run(
                subject_run_path(root, subject_id, run), connectivity=connectivity(region_ts),
                timeseries=timeseries.astype(np.float32), embedding=embedding.astype(np.float32),
                features=features.astype(np.float32), coords=anatomy.astype(np.float32), tr=2.5, n_volumes=T,
                subject_id=subject_id, run_id=run)
            rows.append({"subject_id": subject_id, "run_id": run, "path": str(path.relative_to(root)), "n_volumes": T,
                         "tr": 2.5, "n_regions": R, "n_vertices": V, "qc_pass": True, "contract_version": SCHEMA_VERSION})

    write_manifest(rows, root)
    np.savez(root / "derivatives" / "trajot" / TRUTH_FILE, perm=perms, subject_ids=np.array(ids),
             B_true=B_true, F_true=F_true, coords_true=coords_true, planted=np.bool_(planted_permutation))
    return root


def read_truth(root: Path) -> dict[str, np.ndarray]:
    """The saved ground truth: ``perm (n_subjects, V)``, ``subject_ids``, and the template quantities."""
    with np.load(Path(root) / "derivatives" / "trajot" / TRUTH_FILE) as truth:
        return {key: truth[key] for key in truth.files}
