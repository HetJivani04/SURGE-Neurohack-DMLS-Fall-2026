from __future__ import annotations

import numpy as np

from trajot.geometry.cost import anatomical_cost, feature_cost, geodesic_cost, vertex_mass


def test_geodesic_cost_is_symmetric_and_zero_diagonal() -> None:
    rng = np.random.default_rng(1234)
    emb = rng.normal(size=(30, 6))
    coords = rng.normal(size=(30, 3))

    D = geodesic_cost(emb, coords, k=6, normalize=True)
    assert D.shape == (30, 30)
    assert D.dtype == np.float64
    assert np.allclose(D, D.T, atol=1e-8)
    assert np.allclose(np.diag(D), 0.0, atol=1e-8)
    assert float(D.min()) >= 0.0
    assert float(D.max()) <= 1.0 + 1e-12


def test_vertex_mass_sums_exactly_to_one() -> None:
    mass = vertex_mass(11)
    assert mass.shape == (11,)
    assert mass.dtype == np.float64
    assert float(mass.sum()) == 1.0


def test_feature_cost_shape_and_nonnegative() -> None:
    rng = np.random.default_rng(5)
    Y = rng.normal(size=(7, 3))
    F = rng.normal(size=(4, 3))

    C = feature_cost(Y, F)
    assert C.shape == (7, 4)
    assert C.dtype == np.float64
    assert np.all(C >= 0.0)


def test_anatomical_cost_is_symmetric_and_zero_diagonal() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    M = anatomical_cost(coords, faces, method="geodesic")
    assert M.shape == (4, 4)
    assert M.dtype == np.float64
    assert np.allclose(M, M.T, atol=1e-8)
    assert np.allclose(np.diag(M), 0.0, atol=1e-8)
