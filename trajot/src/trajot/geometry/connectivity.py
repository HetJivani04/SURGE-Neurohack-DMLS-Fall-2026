from __future__ import annotations

import numpy as np


def parcellate(timeseries: np.ndarray, labels: np.ndarray, n_regions: int) -> np.ndarray:
    """Average vertex time series into regional parcels.

    Parameters
    ----------
    timeseries:
        Array with shape (V, T).
    labels:
        Integer labels per vertex. Labels are expected in [0, n_regions - 1].
    n_regions:
        Number of regions R.
    """

    ts = np.asarray(timeseries, dtype=np.float32)
    lab = np.asarray(labels)
    if ts.ndim != 2:
        raise ValueError(f"timeseries must be 2D (V, T), found shape {ts.shape}")
    if lab.shape != (ts.shape[0],):
        raise ValueError(
            f"labels must have shape ({ts.shape[0]},), found {lab.shape}"
        )

    out = np.zeros((n_regions, ts.shape[1]), dtype=np.float32)
    counts = np.zeros(n_regions, dtype=np.int64)

    for region in range(n_regions):
        mask = lab == region
        if not np.any(mask):
            continue
        out[region] = ts[mask].mean(axis=0, dtype=np.float64).astype(np.float32)
        counts[region] = int(mask.sum())

    if np.any(counts == 0):
        missing = np.where(counts == 0)[0]
        raise ValueError(f"Empty region(s) in parcellation labels: {missing.tolist()}")

    return out


def correlation_matrix(ts: np.ndarray) -> np.ndarray:
    """Compute region-wise Pearson correlation matrix over time."""

    x = np.asarray(ts, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"ts must be 2D (R, T), found shape {x.shape}")

    x = x - x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std[std == 0.0] = 1.0
    x = x / std

    corr = (x @ x.T) / max(x.shape[1] - 1, 1)
    corr = np.clip(corr, -1.0, 1.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 0.0)
    return corr.astype(np.float64)


def fisher_z(r: np.ndarray) -> np.ndarray:
    """Apply Fisher-z transform to off-diagonal entries of a correlation matrix."""

    corr = np.asarray(r, dtype=np.float64)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"r must be square 2D, found shape {corr.shape}")

    clipped = corr.copy()
    diag_idx = np.diag_indices_from(clipped)
    clipped[diag_idx] = 0.0
    clipped = np.clip(clipped, -0.999999, 0.999999)

    z = np.arctanh(clipped)
    np.fill_diagonal(z, 0.0)
    return z.astype(np.float64)


def connectivity(ts: np.ndarray) -> np.ndarray:
    """Compute Fisher-z transformed connectivity for parcel time series."""

    return fisher_z(correlation_matrix(ts))


def rank_factorize(C: np.ndarray, r: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Truncated eigendecomposition ``C ~ A A^T`` with ``A`` ``(R, r)`` float64.

    ``A A^T`` is positive semi-definite, so only positive eigenvalues can be represented: the
    ``r`` largest are kept and negative ones are clipped to zero. The connectome ``C`` has a
    zero diagonal, hence is indefinite (its eigenvalues sum to zero) and ``A A^T`` reproduces
    only its positive part. The second return value is the retained-variance fraction, the
    share of ``||C||_F^2`` carried by the kept eigenvalues, so that
    ``||C - A A^T||_F / ||C||_F`` is about ``sqrt(1 - retained)`` (exact when the dropped
    eigenvalues are all the negative ones plus the smaller positive ones).
    """

    matrix = np.asarray(C, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"C must be square 2D, found shape {matrix.shape}")

    evals, evecs = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    r_eff = min(r, matrix.shape[0])
    evals_r = np.clip(evals[:r_eff], a_min=0.0, a_max=None)
    A = evecs[:, :r_eff] * np.sqrt(evals_r)[None, :]

    total = float((evals**2).sum())
    retained = float((evals_r**2).sum()) / total if total > 0 else 0.0

    return A.astype(np.float64), np.float64(retained)
