from __future__ import annotations

import numpy as np
import pytest

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


def test_rank_factorize_retained_variance_is_the_share_of_frobenius_energy() -> None:
    """For an indefinite (zero-diagonal) connectome the fraction must reflect the real error."""
    rng = np.random.default_rng(0)
    C = connectivity(rng.standard_normal((100, 132)).astype(np.float32))
    A, retained = rank_factorize(C, r=32)

    assert A.shape == (100, 32) and A.dtype == np.float64
    error = np.linalg.norm(C - A @ A.T) / np.linalg.norm(C)
    assert 0.0 < retained < 1.0
    assert error == pytest.approx(np.sqrt(1.0 - retained), rel=1e-6)  # not the misleading 1.0 of positive-part variance


def test_rank_factorize_is_exact_for_a_low_rank_psd_matrix() -> None:
    B = np.random.default_rng(1).standard_normal((30, 4))
    A, retained = rank_factorize(B @ B.T, r=8)
    assert retained == pytest.approx(1.0)
    assert np.allclose(A @ A.T, B @ B.T)
