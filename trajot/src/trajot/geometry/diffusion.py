from __future__ import annotations

import numpy as np


def diffusion_map(
    C: np.ndarray,
    d: int = 32,
    alpha: float = 1.0,
    t: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute diffusion-map coordinates from a non-negative affinity matrix.

    Notes
    -----
    Coordinates are scaled by lambda_k**t, which sets the diffusion scale. With
    t=1.0 (default), this is a single-step diffusion geometry.
    """

    affinity = np.asarray(C, dtype=np.float64)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError(f"C must be square 2D, found shape {affinity.shape}")

    W = np.maximum(affinity, 0.0)
    q = W.sum(axis=1)
    q[q <= 0] = 1e-12

    q_alpha = np.power(q, -alpha)
    W_alpha = (q_alpha[:, None] * W) * q_alpha[None, :]

    D = W_alpha.sum(axis=1)
    D[D <= 0] = 1e-12
    D_inv_sqrt = np.power(D, -0.5)

    S = (D_inv_sqrt[:, None] * W_alpha) * D_inv_sqrt[None, :]
    S = 0.5 * (S + S.T)

    evals, evecs = np.linalg.eigh(S)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    # Skip the trivial lambda_0 ~ 1 component.
    evals_nt = evals[1 : d + 1]
    evecs_nt = evecs[:, 1 : d + 1]

    psi = D_inv_sqrt[:, None] * evecs_nt

    # Normalize under stationary measure pi_i proportional to D_i.
    pi = D / D.sum()
    norms = np.sqrt((pi[:, None] * (psi**2)).sum(axis=0))
    norms[norms == 0.0] = 1.0
    psi = psi / norms[None, :]

    coords = psi * np.power(evals_nt[None, :], t)
    return coords.astype(np.float64), evals_nt.astype(np.float64)


def laplacian_eigenvectors(C: np.ndarray, m: int = 8) -> np.ndarray:
    """Return smallest non-trivial eigenvectors of the normalized Laplacian."""

    affinity = np.asarray(C, dtype=np.float64)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError(f"C must be square 2D, found shape {affinity.shape}")

    W = np.maximum(affinity, 0.0)
    deg = W.sum(axis=1)
    deg[deg <= 0] = 1e-12

    inv_sqrt_deg = np.power(deg, -0.5)
    norm_aff = (inv_sqrt_deg[:, None] * W) * inv_sqrt_deg[None, :]
    L = np.eye(W.shape[0], dtype=np.float64) - norm_aff
    L = 0.5 * (L + L.T)

    evals, evecs = np.linalg.eigh(L)
    order = np.argsort(evals)
    evecs = evecs[:, order]

    m_eff = min(m, max(0, W.shape[0] - 1))
    return evecs[:, 1 : m_eff + 1].astype(np.float64)


def gauge_velocity(z: np.ndarray, coords: np.ndarray, dt: float) -> np.ndarray:
    """Compute a spatial finite-difference velocity over lexicographically ordered vertices.

    This is a spatial gradient of the diffusion embedding, not a temporal
    derivative of neural activity.
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
    """Return fused gauge features f_i = [z_i ; beta * v_i].

    The beta term is expected to be the calibrated inverse noise temperature
    beta = sigma_C^-2 from scan-rescan reliability, not a free hyperparameter.
    """

    emb = np.asarray(z, dtype=np.float64)
    vel = np.asarray(v, dtype=np.float64)

    if emb.shape != vel.shape:
        raise ValueError(
            f"z and v must have matching shape (V, d), found {emb.shape} and {vel.shape}"
        )

    return np.concatenate([emb, float(beta) * vel], axis=1).astype(np.float64)
