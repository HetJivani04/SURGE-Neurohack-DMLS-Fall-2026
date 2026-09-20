from __future__ import annotations

import numpy as np

from trajot.geometry.diffusion import (
    diffusion_map,
    gauge_features,
    gauge_velocity,
    laplacian_eigenvectors,
)


def _two_cluster_affinity(n_per_cluster: int = 20) -> np.ndarray:
    n = 2 * n_per_cluster
    rng = np.random.default_rng(123)

    C = np.full((n, n), 0.05, dtype=np.float64)
    C[:n_per_cluster, :n_per_cluster] = 0.9
    C[n_per_cluster:, n_per_cluster:] = 0.9
    C += 0.01 * rng.normal(size=C.shape)
    C = np.clip(C, 0.0, None)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C


def test_diffusion_map_recovers_two_cluster_structure_and_eigs_decrease() -> None:
    C = _two_cluster_affinity()
    z, eigs = diffusion_map(C, d=3, alpha=1.0, t=1.0)

    assert z.shape == (40, 3)
    assert eigs.shape == (3,)
    assert np.all(np.diff(eigs) < 0)

    left_mean = z[:20, 0].mean()
    right_mean = z[20:, 0].mean()
    separation = abs(left_mean - right_mean)
    spread = z[:, 0].std()
    assert separation > 0.5 * spread


def test_laplacian_eigenvectors_shape() -> None:
    C = _two_cluster_affinity(10)
    vecs = laplacian_eigenvectors(C, m=5)
    assert vecs.shape == (20, 5)


def test_gauge_velocity_and_features_have_expected_shapes_and_scaling() -> None:
    rng = np.random.default_rng(9)
    V, d = 16, 4

    z = rng.normal(size=(V, d))
    coords = rng.normal(size=(V, 3))
    v = gauge_velocity(z, coords, dt=0.5)

    assert v.shape == (V, d)
    assert v.dtype == np.float64

    beta = 2.25
    f = gauge_features(z, v, beta=beta)
    assert f.shape == (V, 2 * d)
    assert f.dtype == np.float64
    assert np.allclose(f[:, :d], z)
    assert np.allclose(f[:, d:], beta * v)
