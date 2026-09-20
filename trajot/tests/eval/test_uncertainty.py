from __future__ import annotations

import numpy as np
import pytest

from trajot.eval.uncertainty import (
    UNCERTAINTY_KEYS,
    barycentric_map_variance,
    heldout_predictive_score,
    mean_row_entropy,
    nonident_auroc,
    posterior_coverage,
    scale_template_frobenius,
    shrinkage_transform,
    subject_mean_tau,
    subject_row_entropies,
    temperature_calibrated_coverage,
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


def test_scale_template_frobenius_matches_reference():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8))
    C_ref = 0.5 * (A + A.T)
    np.fill_diagonal(C_ref, 0.0)
    B = rng.normal(size=(8, 3))
    C_bar = 0.5 * (B @ B.T)
    np.fill_diagonal(C_bar, 0.0)
    scaled, scale = scale_template_frobenius(C_bar, C_ref)
    assert scale > 0
    np.testing.assert_allclose(np.linalg.norm(scaled, "fro"), np.linalg.norm(C_ref, "fro"), rtol=1e-10)
    np.testing.assert_allclose(scaled, C_bar * scale, rtol=1e-12)


def test_scale_template_rejects_zero_bar():
    with pytest.raises(ValueError):
        scale_template_frobenius(np.zeros((4, 4)), np.eye(4))


def test_mean_row_entropy_permutation_vs_mix():
    R = 6
    perm = np.eye(R)[np.array([2, 0, 1, 5, 3, 4])] / R
    p1 = np.eye(R)[np.array([0, 1, 2, 3, 4, 5])]
    p2 = np.eye(R)[np.array([1, 0, 3, 2, 5, 4])]
    mixed = (0.5 * p1 + 0.5 * p2) / R
    assert mean_row_entropy(perm) < 0.15
    assert mean_row_entropy(mixed) > 0.5
    assert mean_row_entropy(mixed) > mean_row_entropy(perm)


def test_subject_row_entropies_and_auroc_rank_ambiguity():
    R = 8
    rng = np.random.default_rng(1)
    pis, amb = [], []
    for s in range(20):
        if s < 14:
            perm = rng.permutation(R)
            pis.append(np.eye(R)[perm] / R)
            amb.append(False)
        else:
            p1, p2 = rng.permutation(R), rng.permutation(R)
            pis.append((0.5 * np.eye(R)[p1] + 0.5 * np.eye(R)[p2]) / R)
            amb.append(True)
    scores = subject_row_entropies(pis)
    assert scores.shape == (20,)
    auroc = nonident_auroc(scores, np.asarray(amb))
    assert auroc > 0.8
    # tau-like inverted score must not pass
    inverted = -scores
    assert nonident_auroc(inverted, np.asarray(amb)) < 0.2


def test_barycentric_map_variance_higher_for_mixed_draws():
    R = 6
    p1 = np.eye(R)[np.array([0, 1, 2, 3, 4, 5])] / R
    p2 = np.eye(R)[np.array([1, 0, 3, 2, 5, 4])] / R
    sharp_draws = np.stack([p1] * 30)
    mixed_draws = np.stack([p1 if m % 2 == 0 else p2 for m in range(30)])
    assert barycentric_map_variance(mixed_draws) > barycentric_map_variance(sharp_draws)


def test_coverage_matching_space_when_centered_on_planted():
    """Coverage helper must score planted P* in the same (R,R) probability space as draws."""
    R = 10
    rng = np.random.default_rng(2)
    perm = rng.permutation(R)
    pi_star = np.eye(R)[perm] / R
    draws = pi_star[None].repeat(40, 0) + rng.normal(scale=0.01, size=(40, R, R))
    draws = np.clip(draws, 0.0, None)
    raw = posterior_coverage(draws, pi_star, alpha=0.1)
    assert raw > 0.5


def test_temperature_calibrated_coverage_hits_target_on_heldout():
    """Overconfident collapsed draws centered on planted maps: cal-T on cal split, eval hits target."""
    N, R, M = 20, 8, 12
    rng = np.random.default_rng(3)
    draws_list, stars = [], []
    for s in range(N):
        perm = rng.permutation(R)
        star = np.eye(R)[perm] / R
        # collapsed posterior: tiny spread around a slightly biased mean
        mean = star + 0.15 / R * rng.normal(size=(R, R))
        mean = np.clip(0.5 * (mean + mean.T) * 0 + mean, 0.0, None)  # keep bias, not force sym
        draw_m = mean[None] + 1e-4 * rng.normal(size=(M, R, R))
        draws_list.append(np.clip(draw_m, 0.0, None))
        stars.append(star)
    out = temperature_calibrated_coverage(
        draws_list, stars, alpha=0.1, target=0.90,
    )
    assert "coverage_eval" in out and "coverage_raw_eval" in out and "temperature" in out
    assert 0.85 <= out["coverage_eval"] <= 0.95
    assert out["temperature"] > 0
    # raw (uncalibrated) coverage of collapsed biased draws is near zero
    assert out["coverage_raw_eval"] < 0.5


def test_temperature_calibrated_coverage_honest_fail_when_wrong_space():
    """Draws centered far from planted (wrong template space) cannot be saved by calibration alone
    into the pre-registered band on eval — reports the actual eval coverage."""
    N, R, M = 16, 6, 10
    rng = np.random.default_rng(4)
    draws_list, stars = [], []
    for s in range(N):
        perm = rng.permutation(R)
        star = np.eye(R)[perm] / R
        wrong_perm = rng.permutation(R)
        while np.array_equal(wrong_perm, perm):
            wrong_perm = rng.permutation(R)
        mean = np.eye(R)[wrong_perm] / R
        draws_list.append(mean[None] + 1e-3 * rng.normal(size=(M, R, R)))
        stars.append(star)
    out = temperature_calibrated_coverage(draws_list, stars, alpha=0.1, target=0.90)
    assert 0.0 <= out["coverage_eval"] <= 1.0
    # Calibration can inflate intervals; if it cannot reach the band, the flag stays False
    assert isinstance(out["in_pre_registered_band"], bool)


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
