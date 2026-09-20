"""Calibration of ``beta`` from scan-rescan reliability.

``E_GW`` has no tractable normalizer in ``pi``, so the model is a Gibbs (generalized) posterior in the
Bissiri-Holmes-Walker sense, defensible only if ``beta`` is fixed from data rather than tuned::

    sigma_hat_C^2 = 0.5 * mean_s || C_s^(1) - C_s^(2) ||_F^2 / R^2 ,        beta = sigma_hat_C^-2

with the mean over the **83 two-run subjects** only. :func:`calibrate_beta` is the only place ``beta`` is computed.

**Normalizer (locked).** ``R = C.shape[-1]`` is the connectome node count: the side of the region-level
connectivity matrix ``C_s (R, R)`` (parcels in the connectivity), **not** the surface vertex count
``n_vertices``. The symbol ``V`` that appears in some issue prose for this formula denotes ``R``. ``beta`` is
therefore the inverse **per-entry** scan-rescan variance, which is what the unit-row-mass ``E_GW`` scale in
:mod:`trajot.inference.train` presupposes. Issue text saying ``V = n_vertices`` is wrong relative to this
observation model. :func:`calibrate_beta` writes ``n_regions`` (= ``R``) into ``beta.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from trajot.io.contract import load_connectomes, read_manifest, two_run_subjects


def _per_subject_terms(C1: np.ndarray, C2: np.ndarray) -> np.ndarray:
    C1, C2 = np.asarray(C1, dtype=np.float64), np.asarray(C2, dtype=np.float64)
    if C1.shape != C2.shape or C1.ndim != 3 or C1.shape[1] != C1.shape[2]:
        raise ValueError(f"C1 and C2 must both be (n, R, R), found {C1.shape} and {C2.shape}")
    return 0.5 * ((C1 - C2) ** 2).sum(axis=(1, 2)) / C1.shape[-1] ** 2


def sigma_c_squared(C1: np.ndarray, C2: np.ndarray) -> float:
    """``0.5 * mean_s ||C1[s] - C2[s]||_F^2 / R^2`` for two-run connectomes ``C1, C2`` ``(n, R, R)`` float64.

    ``R = C.shape[-1]`` is the connectome node count (parcels), not the surface vertex count. The result is
    the per-entry variance of a subject's connectome between two scans: half the mean squared difference of a
    scan-rescan pair, normalised by ``R^2`` so it matches the unit-row-mass ``E_GW`` convention.
    """
    return float(_per_subject_terms(C1, C2).mean())


def calibrate_beta(root: Path, subjects: Sequence[str] | None = None, out_dir: Path | None = None) -> dict:
    """Compute ``sigma_hat_C^2`` and ``beta = sigma_hat_C^-2`` from the two runs of the two-run subjects.

    With ``subjects=None`` the subjects come from :func:`trajot.io.contract.two_run_subjects` and the count is
    asserted to be 83 (it raises otherwise); an explicit ``subjects`` list (each must have two runs) is for
    tests of the estimator. Loads both runs' connectomes, writes ``<out_dir>/beta.json`` (``out_dir`` defaults
    to ``artifacts/``) with ``sigma_hat_C_squared``, ``beta``, ``n_regions`` (``R = C.shape[-1]``, connectome
    nodes / parcels — not surface vertices), ``n_subjects``, ``subject_ids`` and ``per_subject_terms``, and
    returns that dict.
    """
    manifest = read_manifest(Path(root))
    if subjects is None:
        ids = two_run_subjects(manifest)  # strict: exactly 83 subjects with two runs of equal length
        assert len(ids) == 83, f"expected 83 two-run subjects, found {len(ids)}"
    else:
        ids = [str(s) for s in subjects]
        counts = manifest.groupby("subject_id").size()
        missing = [s for s in ids if counts.get(s, 0) != 2]
        if missing:
            raise ValueError(f"subjects without exactly two runs: {missing}")

    C1, ids1 = load_connectomes(Path(root), ids, run="1")
    C2, ids2 = load_connectomes(Path(root), ids, run="2")
    assert ids1 == ids2, "run 1 and run 2 must list the same subjects in the same order"

    terms = _per_subject_terms(C1, C2)
    sigma2 = float(terms.mean())
    result = {"sigma_hat_C_squared": sigma2, "beta": 1.0 / sigma2, "n_regions": int(C1.shape[-1]),
              "n_subjects": len(ids1), "subject_ids": list(ids1), "per_subject_terms": [float(t) for t in terms]}

    out_dir = Path(out_dir) if out_dir is not None else Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "beta.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
