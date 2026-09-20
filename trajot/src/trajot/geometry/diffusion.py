from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


def diffusion_map(
    C: np.ndarray,
    d: int = 32,
    alpha: float = 1.0,
    t: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Diffusion-map coordinates of a symmetric affinity matrix ``C`` ``(V, V)``.

    ``W = max(C, 0)`` (diffusion geometry needs non-negative affinities), ``q = W 1``,
    ``W_alpha = q^-alpha W q^-alpha``, ``D = diag(W_alpha 1)``. The random-walk matrix
    ``D^-1 W_alpha`` is solved through the symmetric problem on ``D^-1/2 W_alpha D^-1/2``,
    keeping the ``d`` largest non-trivial eigenvalues (``lambda_0 = 1`` is skipped) with
    eigenvectors ``psi_k = D^-1/2 v_k`` normalized so that ``sum_i pi_i psi_k[i]^2 = 1`` under
    the stationary measure ``pi ~ D``. Returns ``z_i = (lambda_k^t psi_k[i])_k`` as ``(V, d)``
    float64 and the eigenvalues ``(d,)`` float64, strictly decreasing.

    The factor ``lambda_k^t`` is what makes Euclidean distance between rows of the embedding
    equal the diffusion distance at scale ``t``; ``t = 1`` (one diffusion step) is the default.

    Only the ``d + 1`` largest eigenpairs are needed, so large matrices use Lanczos (ARPACK)
    with a fixed starting vector, which is deterministic and about 30x faster than a dense
    solve at ``V = 5,000``; small matrices, or a failure to converge, use a dense solve. The
    ``V x V`` work is done in place (peak about two ``V x V`` float64 arrays), and the sign of
    each coordinate is fixed (its largest entry is positive) so repeated runs agree exactly.
    """

    affinity = np.asarray(C, dtype=np.float64)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError(f"C must be square 2D, found shape {affinity.shape}")
    n = affinity.shape[0]

    W = np.maximum(affinity, 0.0)
    q = W.sum(axis=1)
    q[q <= 0] = 1e-12
    q_alpha = np.power(q, -alpha)
    W *= q_alpha[:, None]
    W *= q_alpha[None, :]

    D = W.sum(axis=1)
    D[D <= 0] = 1e-12
    D_inv_sqrt = np.power(D, -0.5)
    W *= D_inv_sqrt[:, None]
    W *= D_inv_sqrt[None, :]

    k = min(d, n - 1)
    evals = evecs = None
    if n > 200 and 2 * (k + 1) < n:
        try:
            evals, evecs = eigsh(W, k=k + 1, which="LA", v0=np.random.default_rng(0).standard_normal(n), tol=1e-10)
        except ArpackNoConvergence:
            evals = evecs = None
    if evals is None:
        evals, evecs = eigh(W, subset_by_index=[n - k - 1, n - 1], overwrite_a=True)
    del W
    order = np.argsort(evals)[::-1]  # descending; index 0 is the trivial component
    evals, evecs = evals[order], evecs[:, order]

    evals_nt = evals[1 : k + 1]
    psi = D_inv_sqrt[:, None] * evecs[:, 1 : k + 1]

    pi = D / D.sum()
    norms = np.sqrt((pi[:, None] * (psi**2)).sum(axis=0))
    norms[norms == 0.0] = 1.0
    psi = psi / norms[None, :]

    psi *= np.where(psi[np.abs(psi).argmax(axis=0), np.arange(psi.shape[1])] < 0, -1.0, 1.0)[None, :]

    coords = psi * np.power(evals_nt[None, :], t)
    return coords.astype(np.float64), evals_nt.astype(np.float64)


def laplacian_eigenvectors(C: np.ndarray, m: int = 8) -> np.ndarray:
    """The ``m`` smallest non-trivial eigenvectors of the normalized Laplacian of ``max(C, 0)``, ``(V, m)`` float64.

    These are the largest non-trivial eigenvectors of ``D^-1/2 W D^-1/2``. Large matrices use Lanczos with a
    fixed starting vector (deterministic, far faster than a dense solve at ``V = 5,000``); small ones, or a
    failure to converge, use a dense solve.
    """

    affinity = np.asarray(C, dtype=np.float64)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError(f"C must be square 2D, found shape {affinity.shape}")

    n = affinity.shape[0]
    W = np.maximum(affinity, 0.0)
    deg = W.sum(axis=1)
    deg[deg <= 0] = 1e-12

    inv_sqrt_deg = np.power(deg, -0.5)
    W *= inv_sqrt_deg[:, None]
    W *= inv_sqrt_deg[None, :]  # now the normalized affinity; L = I - W

    m_eff = min(m, max(0, n - 1))
    evals = evecs = None
    if n > 200 and 2 * (m_eff + 1) < n:
        try:
            evals, evecs = eigsh(W, k=m_eff + 1, which="LA", v0=np.random.default_rng(0).standard_normal(n), tol=1e-10)
        except ArpackNoConvergence:
            evals = evecs = None
    if evals is None:
        evals, evecs = np.linalg.eigh(0.5 * (W + W.T))
    order = np.argsort(evals)[::-1]  # largest affinity eigenvalue = smallest Laplacian eigenvalue = trivial
    return evecs[:, order][:, 1 : m_eff + 1].astype(np.float64)


def gauge_velocity(z: np.ndarray, coords: np.ndarray, dt: float) -> np.ndarray:
    """Spatial finite-difference of the diffusion embedding along lex-sorted coords, divided by ``dt`` (TR).

    This is a **spatial** gradient of the diffusion embedding used as a gauge-breaking channel. It is not a
    time derivative of neural activity and not a model of dynamics. The Gromov-Wasserstein term is
    isometry-invariant; a feature channel linear in the coupling breaks that isometry down to finitely many
    optima. ``dt`` only sets the numerical scale (TR from the contract); the quantity is not motion over time.
    """

    emb = np.asarray(z, dtype=np.float64)
    xyz = np.asarray(coords, dtype=np.float64)

    if emb.ndim != 2:
        raise ValueError(f"z must be 2D (V, d), found shape {emb.shape}")
    if xyz.shape != (emb.shape[0], 3):
        raise ValueError(
            f"coords must have shape ({emb.shape[0]}, 3), found {xyz.shape}"
        )
    if dt <= 0:
        raise ValueError("dt must be positive")

    v = np.zeros_like(emb, dtype=np.float64)

    hemi_boundary = np.median(xyz[:, 0])
    hemispheres = [np.where(xyz[:, 0] <= hemi_boundary)[0], np.where(xyz[:, 0] > hemi_boundary)[0]]

    for idx in hemispheres:
        if idx.size <= 1:
            continue
        local = xyz[idx]
        order = np.lexsort((local[:, 2], local[:, 1], local[:, 0]))
        ordered_idx = idx[order]

        diff = np.diff(emb[ordered_idx], axis=0) / dt
        v[ordered_idx[:-1]] = diff
        v[ordered_idx[-1]] = 0.0

    return v


def gauge_features(z: np.ndarray, v: np.ndarray, beta: float) -> np.ndarray:
    """Return the gauge features ``f_i = [z_i ; beta * v_i]`` as ``(V, 2d)`` float64.

    ``v`` is the spatial gradient of the diffusion embedding (a gauge-breaking channel), not a time-series
    feature. ``beta`` is the calibrated inverse temperature ``sigma_C^-2`` from scan-rescan reliability
    (W2), not a free parameter, so the weight of that channel is set by how reliable an individual's own
    connectome is between two scans.

    Why the features are needed: the Gromov-Wasserstein term is quadratic in the coupling and
    invariant to isometries (Memoli 2011), so a coupling is defined only up to a symmetry
    group. A feature term that is *linear* in the coupling selects a label assignment and
    reduces the isometry orbit to finitely many optima. It never yields "a unique minimizer".
    """

    emb = np.asarray(z, dtype=np.float64)
    vel = np.asarray(v, dtype=np.float64)

    if emb.shape != vel.shape:
        raise ValueError(
            f"z and v must have matching shape (V, d), found {emb.shape} and {vel.shape}"
        )

    return np.concatenate([emb, float(beta) * vel], axis=1).astype(np.float64)
