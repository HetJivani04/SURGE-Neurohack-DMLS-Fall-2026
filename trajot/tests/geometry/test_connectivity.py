from __future__ import annotations

import numpy as np

from trajot.geometry.connectivity import connectivity, fisher_z, rank_factorize


def test_connectivity_is_symmetric_with_zero_diagonal_over_random_inputs() -> None:
    rng = np.random.default_rng(42)

    for _ in range(20):
        ts = rng.normal(size=(10, 60)).astype(np.float32)
        C = connectivity(ts)
        assert C.dtype == np.float64
        assert np.allclose(C, C.T, atol=1e-8)
        assert np.allclose(np.diag(C), 0.0, atol=1e-8)


def test_fisher_z_clips_off_diagonal_to_avoid_infinite_values() -> None:
    corr = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    z = fisher_z(corr)
    assert np.isfinite(z).all()


def test_rank_factorize_reconstructs_low_rank_psd_matrix() -> None:
    rng = np.random.default_rng(7)
    B = rng.normal(size=(12, 4))
    C = B @ B.T

    A, retained = rank_factorize(C, r=4)
    C_hat = A @ A.T

    rel_err = np.linalg.norm(C - C_hat) / np.linalg.norm(C)
    assert A.shape == (12, 4)
    assert rel_err < 1e-8
    assert float(retained) > 0.999
