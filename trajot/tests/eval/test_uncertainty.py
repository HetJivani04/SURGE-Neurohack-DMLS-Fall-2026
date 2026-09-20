from __future__ import annotations

import numpy as np
import pytest

from trajot.eval.uncertainty import (
    UNCERTAINTY_KEYS,
    heldout_predictive_score,
    nonident_auroc,
    posterior_coverage,
    shrinkage_transform,
    subject_mean_tau,
)


def test_shrinkage_limits():
    C = np.random.randn(20, 20)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 0)
    U, _, Vt = np.linalg.svd(np.random.randn(20, 20))
    Q = U @ Vt
    assert np.allclose(shrinkage_transform(C, Q, 0.0), C)
    assert np.allclose(shrinkage_transform(C, Q, 1.0), Q.T @ C @ Q)


def test_shrinkage_midpoint():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 6))
    C = 0.5 * (A + A.T)
    U, _, Vt = np.linalg.svd(rng.normal(size=(6, 6)))
    Q = U @ Vt
    lam = 0.25
    expected = (1.0 - lam) * C + lam * (Q.T @ C @ Q)
    assert np.allclose(shrinkage_transform(C, Q, lam), expected)


def test_shrinkage_rejects_bad_lambda():
    C = np.eye(3)
    Q = np.eye(3)
    with pytest.raises(ValueError):
        shrinkage_transform(C, Q, -0.1)
    with pytest.raises(ValueError):
        shrinkage_transform(C, Q, 1.1)


def test_coverage_perfect_posterior():
    pi_star = np.eye(8)[None].repeat(50, 0)
    assert posterior_coverage(pi_star + 0.0, pi_star[0], 0.1) == 1.0


def test_coverage_misses_when_shifted():
    pi_star = np.zeros((4, 4))
    np.fill_diagonal(pi_star, 1.0)
    draws = pi_star[None].repeat(40, 0) + 5.0
    assert posterior_coverage(draws, pi_star, 0.1) == 0.0


def test_coverage_interval_quantiles():
    # 1D edge case via (M, 1, 1): uniform draws on [0, 1]; alpha=0.1 => [0.05, 0.95]
    draws = np.linspace(0.0, 1.0, 100)[:, None, None]
    assert posterior_coverage(draws, np.array([[0.5]]), alpha=0.1) == 1.0
    assert posterior_coverage(draws, np.array([[-0.01]]), alpha=0.1) == 0.0


def test_auroc_separation():
    tau = np.array([0.01] * 10 + [0.5] * 10)
    amb = np.array([False] * 10 + [True] * 10)
    assert nonident_auroc(tau, amb) > 0.95


def test_auroc_empty_class_is_half():
    tau = np.array([0.1, 0.2, 0.3])
    assert nonident_auroc(tau, np.zeros(3, dtype=bool)) == 0.5
    assert nonident_auroc(tau, np.ones(3, dtype=bool)) == 0.5


def test_auroc_sequence_of_subject_taus():
    tau_phi = [np.full(5, 0.01), np.full(5, 0.02), np.full(5, 0.8), np.full(5, 0.9)]
    amb = np.array([False, False, True, True])
    assert nonident_auroc(tau_phi, amb) == 1.0


def test_auroc_ties_score_half():
    tau = np.ones(6)
    amb = np.array([False, True, False, True, False, True])
    assert nonident_auroc(tau, amb) == pytest.approx(0.5)


def test_heldout_predictive_score():
    C2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    aligned = C2.copy()
    worse = C2 + 1.0
    perfect = heldout_predictive_score(C2, aligned)
    bad = heldout_predictive_score(C2, worse)
    assert perfect == pytest.approx(0.0)
    assert bad < perfect
    assert perfect == pytest.approx(-np.linalg.norm(C2 - aligned, "fro") ** 2)


def test_subject_mean_tau():
    tau_phi = [np.array([0.1, 0.3]), np.array([0.5, 0.5])]
    means = subject_mean_tau(tau_phi)
    assert means.shape == (2,)
    assert means[0] == pytest.approx(0.2)
    assert means[1] == pytest.approx(0.5)


def test_subject_mean_tau_accepts_stacked():
    stacked = np.array([[0.0, 2.0], [1.0, 1.0]])
    means = subject_mean_tau(stacked)
    assert means.shape == (2,)
    assert np.allclose(means, [1.0, 1.0])


def test_uncertainty_keys():
    assert set(UNCERTAINTY_KEYS) == {
        "posterior_coverage",
        "nonident_auroc",
        "heldout_score",
        "group_neff",
    }


def test_metrics_keys_exported():
    from trajot.eval import (
        UNCERTAINTY_KEYS,
        heldout_predictive_score,
        nonident_auroc,
        posterior_coverage,
        shrinkage_transform,
        subject_mean_tau,
    )

    assert callable(shrinkage_transform)
    assert callable(posterior_coverage)
    assert callable(nonident_auroc)
    assert callable(heldout_predictive_score)
    assert callable(subject_mean_tau)
    assert "posterior_coverage" in UNCERTAINTY_KEYS


def test_optional_uncertainty_keys_separate_from_method_keys():
    from trajot.eval.metrics import METHOD_KEYS, OPTIONAL_UNCERTAINTY_KEYS, UNCERTAINTY_KEYS

    assert OPTIONAL_UNCERTAINTY_KEYS == set(UNCERTAINTY_KEYS)
    assert not (OPTIONAL_UNCERTAINTY_KEYS & METHOD_KEYS)


def test_validate_metrics_payload_without_uncertainty_keys():
    from trajot.eval.metrics import (
        method_metrics_template,
        validate_metrics_payload,
    )

    payload = {
        "experiment": "x",
        "run_id": "r",
        "n_subjects": 2,
        "n_pairs": 1,
        "pairs_seed": 0,
        "permutations_B": 10,
        "methods": {"baseline": method_metrics_template()},
        "beta": 29.189,
        "notes": None,
    }
    validate_metrics_payload(payload)


def test_validate_metrics_payload_allows_optional_uncertainty_keys():
    from trajot.eval.metrics import (
        method_metrics_template,
        validate_metrics_payload,
    )

    stats = method_metrics_template()
    stats["posterior_coverage"] = 0.91
    stats["nonident_auroc"] = 0.88
    stats["heldout_score"] = -1.5
    stats["group_neff"] = 12.0
    payload = {
        "experiment": "x",
        "run_id": "r",
        "n_subjects": 2,
        "n_pairs": 1,
        "pairs_seed": 0,
        "permutations_B": 10,
        "methods": {"ours_full": stats},
        "beta": 29.189,
        "notes": None,
    }
    validate_metrics_payload(payload)
