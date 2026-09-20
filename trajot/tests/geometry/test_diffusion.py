from __future__ import annotations

import numpy as np
import pytest

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


def _clustered_correlation(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((8, 90))
    return np.corrcoef(0.7 * latent[rng.integers(0, 8, n)] + rng.standard_normal((n, 90)))


def _reference_eigenvalues(C: np.ndarray, d: int) -> np.ndarray:
    W = np.maximum(C, 0.0)
    q = W.sum(axis=1)
    Wa = W / q[:, None] / q[None, :]
    D = Wa.sum(axis=1)
    S = Wa / np.sqrt(D)[:, None] / np.sqrt(D)[None, :]
    return np.sort(np.linalg.eigvalsh(S))[::-1][1 : d + 1]


@pytest.mark.parametrize("n", [150, 700])  # dense solver and Lanczos solver
def test_diffusion_map_matches_a_full_eigendecomposition_and_is_reproducible(n: int) -> None:
    C = _clustered_correlation(n)
    z, evals = diffusion_map(C, d=12)
    z_again, _ = diffusion_map(C, d=12)

    assert z.shape == (n, 12) and z.dtype == np.float64 and evals.shape == (12,)
    assert np.allclose(evals, _reference_eigenvalues(C, 12), atol=1e-8)
    assert np.all(np.diff(evals) < 0)
    assert np.array_equal(z, z_again)


def test_diffusion_map_fixes_the_sign_of_each_coordinate() -> None:
    z, _ = diffusion_map(_clustered_correlation(400), d=6)
    largest = z[np.abs(z).argmax(axis=0), np.arange(6)]
    assert np.all(largest > 0)


def test_diffusion_map_scales_coordinates_by_lambda_to_the_t() -> None:
    C = _clustered_correlation(300)
    z1, evals = diffusion_map(C, d=5, t=1.0)
    z2, _ = diffusion_map(C, d=5, t=2.0)
    assert np.allclose(z2, z1 * evals[None, :], atol=1e-10)


def test_diffusion_map_with_fewer_vertices_than_dimensions() -> None:
    z, evals = diffusion_map(_clustered_correlation(10), d=32)
    assert z.shape == (10, 9) and evals.shape == (9,)


@pytest.mark.parametrize("n", [120, 500])  # dense path and Lanczos path
def test_laplacian_eigenvectors_span_the_same_subspace_as_a_dense_solve(n: int) -> None:
    C = _clustered_correlation(n)
    W = np.maximum(C, 0.0)
    inv_sqrt = 1.0 / np.sqrt(W.sum(axis=1))
    laplacian = np.eye(n) - inv_sqrt[:, None] * W * inv_sqrt[None, :]
    values, vectors = np.linalg.eigh(laplacian)
    expected = vectors[:, 1:7]

    got = laplacian_eigenvectors(C, m=6)
    assert got.shape == (n, 6) and got.dtype == np.float64
    assert np.allclose(got.T @ got, np.eye(6), atol=1e-6)
    assert np.allclose(expected @ expected.T, got @ got.T, atol=1e-6)  # same subspace (eigenvector signs are free)
    assert np.array_equal(got, laplacian_eigenvectors(C, m=6))  # deterministic
