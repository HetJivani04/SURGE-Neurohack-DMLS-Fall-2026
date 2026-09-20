from __future__ import annotations

import itertools
import warnings
import weakref

import numpy as np
import pytest
from scipy import stats

from trajot.report.group import (
    alignment_covariance, delta_method_cov, meta_analysis, meta_analysis_map, one_sample_ttest, sign_flip_null,
    sign_flip_test, subject_maps,
)

RESULT_KEYS = ["tau_hat_k2", "u_sk", "theta_hat_k", "se_theta", "t_k", "df", "p_k"]


def coupling(rng, V, K):
    """A random (V, K) coupling with uniform mass: rows sum to 1/V, columns to 1/K (Sinkhorn on random scores)."""
    pi = rng.random((V, K)) + 0.05
    for _ in range(300):
        pi *= (1.0 / V) / pi.sum(1, keepdims=True)
        pi *= (1.0 / K) / pi.sum(0, keepdims=True)
    return pi


def problem(rng, S=12, K=40, tau=0.7, scale=0.3):
    """Simulated ``m_sk`` and known ``sigma2_sk``: m = theta + b + e with b ~ N(0, tau^2), e ~ N(0, sigma2)."""
    sigma2 = scale**2 * rng.uniform(0.2, 3.0, size=(S, K))
    theta = rng.normal(size=K)
    return theta + tau * rng.normal(size=(S, K)) + np.sqrt(sigma2) * rng.normal(size=(S, K)), sigma2


# ---- subject_maps ----------------------------------------------------------------------------------------
def test_subject_maps_matches_the_definition_for_random_shapes() -> None:
    rng = np.random.default_rng(0)
    for _ in range(30):
        V, K = int(rng.integers(2, 30)), int(rng.integers(2, 12))
        pi, z, nu = rng.random((V, K)), rng.normal(size=V), rng.random(K) + 0.1
        w = subject_maps(z, pi, nu)
        brute = np.array([sum(pi[v, k] * z[v] for v in range(V)) / nu[k] for k in range(K)])
        assert w.shape == (K,) and w.dtype == np.float64
        np.testing.assert_allclose(w, brute, rtol=1e-12, atol=1e-14)


def test_subject_maps_is_linear_in_z_and_equivariant_to_template_permutation() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        V, K = 25, 7
        pi, nu = rng.random((V, K)), rng.random(K) + 0.1
        z1, z2, a, b = rng.normal(size=V), rng.normal(size=V), rng.normal(), rng.normal()
        np.testing.assert_allclose(subject_maps(a * z1 + b * z2, pi, nu),
                                   a * subject_maps(z1, pi, nu) + b * subject_maps(z2, pi, nu), atol=1e-12)
        perm = rng.permutation(K)
        np.testing.assert_allclose(subject_maps(z1, pi[:, perm], nu[perm]), subject_maps(z1, pi, nu)[perm], atol=1e-14)


def test_a_constant_map_is_carried_to_the_same_constant_when_the_coupling_has_marginal_nu() -> None:
    rng = np.random.default_rng(2)
    for _ in range(10):
        V, K = int(rng.integers(3, 20)), int(rng.integers(2, 9))
        mu, nu = rng.random(V) + 0.1, rng.random(K) + 0.1
        mu, nu = mu / mu.sum(), nu / nu.sum()
        c = float(rng.normal())
        np.testing.assert_allclose(subject_maps(np.full(V, c), np.outer(mu, nu), nu), np.full(K, c), atol=1e-12)


@pytest.mark.parametrize("z,pi,nu", [(np.ones(3), np.ones((4, 2)), np.ones(2)), (np.ones(4), np.ones((4, 2)), np.ones(3)),
                                      (np.ones(4), np.ones((4, 2)), np.array([1.0, 0.0])),
                                      (np.ones(4), np.ones((4, 2)), np.array([1.0, -1.0]))])
def test_subject_maps_rejects_inconsistent_shapes_and_nonpositive_mass(z, pi, nu) -> None:
    with pytest.raises(ValueError):
        subject_maps(z, pi, nu)


# ---- alignment_covariance --------------------------------------------------------------------------------
def draws(rng, M, V, K, spread=0.3):
    base = coupling(rng, V, K)
    return base[None] * np.exp(spread * rng.normal(size=(M, V, K)))


def test_alignment_covariance_is_the_covariance_of_the_draws_maps() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        M, V, K = int(rng.integers(3, 12)), int(rng.integers(3, 15)), int(rng.integers(2, 8))
        pis, z, nu = draws(rng, M, V, K), rng.normal(size=V), np.full(K, 1.0 / K)
        cov = alignment_covariance(pis, z, nu)
        maps = np.stack([subject_maps(z, pis[m], nu) for m in range(M)])
        assert cov.shape == (K, K) and cov.dtype == np.float64
        np.testing.assert_allclose(cov, np.cov(maps, rowvar=False), rtol=1e-10, atol=1e-14)


def test_alignment_covariance_is_symmetric_positive_semidefinite_and_scales_quadratically() -> None:
    rng = np.random.default_rng(4)
    for _ in range(20):
        M, V, K = 9, 12, 6
        pis, z, nu = draws(rng, M, V, K), rng.normal(size=V), np.full(K, 1.0 / K)
        cov = alignment_covariance(pis, z, nu)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        assert np.linalg.eigvalsh(cov).min() > -1e-10 * max(1.0, np.abs(cov).max()) and (np.diag(cov) >= 0).all()
        a = float(rng.normal()) + 2.0
        np.testing.assert_allclose(alignment_covariance(pis, a * z, nu), a**2 * cov, rtol=1e-9, atol=1e-12)


def test_identical_draws_have_zero_alignment_covariance_and_one_draw_is_rejected() -> None:
    rng = np.random.default_rng(5)
    pi, z, nu = coupling(rng, 10, 4), rng.normal(size=10), np.full(4, 0.25)
    np.testing.assert_array_equal(alignment_covariance(np.stack([pi, pi, pi]), z, nu), np.zeros((4, 4)))
    with pytest.raises(ValueError, match="draws"):
        alignment_covariance(pi[None], z, nu)


# ---- delta_method_cov ------------------------------------------------------------------------------------
def test_delta_method_cov_matches_the_definition_and_is_symmetric_psd() -> None:
    rng = np.random.default_rng(6)
    for _ in range(20):
        K, V = int(rng.integers(2, 9)), int(rng.integers(2, 20))
        J, tau2 = rng.normal(size=(K, V)), rng.random(V)
        cov = delta_method_cov(J, tau2)
        brute = sum(tau2[v] * np.outer(J[:, v], J[:, v]) for v in range(V))
        assert cov.shape == (K, K) and cov.dtype == np.float64
        np.testing.assert_allclose(cov, brute, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(cov, cov.T, atol=1e-13)
        assert np.linalg.eigvalsh(cov).min() > -1e-10
        np.testing.assert_array_equal(delta_method_cov(J, np.zeros(V)), np.zeros((K, K)))


def perturbed_draws(rng, M, V, K, tau2, whiten):
    """pi^(m) = pi0 + sum_v eps_v^(m) e_v g_v^T, so w^(m) = w0 + J eps^(m) exactly with J[:, v] = z_v g_v / nu."""
    pi0, G, z, nu = coupling(rng, V, K), rng.normal(size=(V, K)), rng.normal(size=V), np.full(K, 1.0 / K)
    eps = rng.normal(size=(M, V))
    if whiten:  # zero mean and sample covariance exactly I, so the empirical covariance is exactly diag(tau2)
        eps -= eps.mean(0)
        eps = np.linalg.qr(eps)[0] * np.sqrt(M - 1)
    eps = eps * np.sqrt(tau2)
    return pi0[None] + eps[:, :, None] * G[None], z, nu, (z[:, None] * G / nu[None, :]).T


def test_delta_method_equals_the_empirical_covariance_exactly_for_whitened_draws() -> None:
    rng = np.random.default_rng(7)
    for _ in range(10):
        V, K = int(rng.integers(3, 9)), int(rng.integers(2, 7))
        tau2 = rng.random(V) + 0.05
        pis, z, nu, J = perturbed_draws(rng, M=V + 6, V=V, K=K, tau2=tau2, whiten=True)
        np.testing.assert_allclose(delta_method_cov(J, tau2), alignment_covariance(pis, z, nu), rtol=1e-8, atol=1e-10)


def test_delta_method_matches_the_empirical_covariance_for_many_random_draws() -> None:
    rng = np.random.default_rng(8)
    V, K, tau2 = 5, 4, np.array([0.5, 1.0, 0.2, 0.8, 0.3])
    pis, z, nu, J = perturbed_draws(rng, M=40000, V=V, K=K, tau2=tau2, whiten=False)
    exact, empirical = delta_method_cov(J, tau2), alignment_covariance(pis, z, nu)
    assert np.linalg.norm(empirical - exact) / np.linalg.norm(exact) < 0.03


# ---- one_sample_ttest ------------------------------------------------------------------------------------
def test_one_sample_ttest_is_the_scipy_t_test_per_column() -> None:
    rng = np.random.default_rng(9)
    for _ in range(10):
        m = rng.normal(loc=0.3, size=(int(rng.integers(3, 30)), int(rng.integers(1, 20))))
        out, ref = one_sample_ttest(m), stats.ttest_1samp(m, 0.0, axis=0)
        assert list(out) == RESULT_KEYS
        np.testing.assert_allclose(out["t_k"], ref.statistic, rtol=1e-12)
        np.testing.assert_allclose(out["p_k"], ref.pvalue, rtol=1e-10, atol=1e-300)
        np.testing.assert_allclose(out["theta_hat_k"], m.mean(0), rtol=1e-12)
        np.testing.assert_allclose(out["se_theta"], m.std(0, ddof=1) / np.sqrt(len(m)), rtol=1e-12)
        np.testing.assert_array_equal(out["df"], np.full(m.shape[1], len(m) - 1.0))
        np.testing.assert_allclose(out["tau_hat_k2"], m.var(0, ddof=1), rtol=1e-12)


def test_one_sample_ttest_rejects_a_single_subject() -> None:
    with pytest.raises(ValueError, match="two subjects"):
        one_sample_ttest(np.ones((1, 3)))


# ---- meta_analysis ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["REML", "MoM"])
def test_meta_analysis_collapses_exactly_to_the_one_sample_t_test_when_sigma_is_zero(method: str) -> None:
    rng = np.random.default_rng(10)
    for _ in range(25):
        m, _ = problem(rng, S=int(rng.integers(3, 40)), K=int(rng.integers(1, 30)))
        a, b = meta_analysis(m, np.zeros_like(m), method=method), one_sample_ttest(m)
        assert list(a) == RESULT_KEYS
        for key in RESULT_KEYS:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-10, atol=0, err_msg=key)


@pytest.mark.parametrize("method", ["REML", "MoM"])
def test_meta_analysis_converges_to_the_one_sample_t_test_as_sigma_shrinks(method: str) -> None:
    rng = np.random.default_rng(11)
    m, sigma2 = problem(rng, S=15, K=60)
    reference = one_sample_ttest(m)
    errors = []
    for scale in (1e-1, 1e-3, 1e-5, 1e-7, 1e-9):
        out = meta_analysis(m, scale * sigma2, method=method)
        errors.append(max(np.abs(out[k] - reference[k]).max() / max(1.0, np.abs(reference[k]).max()) for k in ("t_k", "theta_hat_k", "df")))
    assert all(a > b for a, b in zip(errors, errors[1:])) and errors[-1] < 1e-6


def test_the_weights_are_one_over_tau_plus_sigma_and_the_estimates_follow_the_formulas() -> None:
    rng = np.random.default_rng(12)
    m, sigma2 = problem(rng, S=14, K=25)
    for method in ("REML", "MoM"):
        out = meta_analysis(m, sigma2, method=method)
        u = 1.0 / (out["tau_hat_k2"][None, :] + sigma2)
        assert out["u_sk"].shape == m.shape and out["tau_hat_k2"].shape == (25,)
        np.testing.assert_allclose(out["u_sk"], u, rtol=1e-13)
        np.testing.assert_allclose(out["theta_hat_k"], (u * m).sum(0) / u.sum(0), rtol=1e-12)
        np.testing.assert_allclose(out["se_theta"], u.sum(0) ** -0.5, rtol=1e-12)
        np.testing.assert_allclose(out["t_k"], out["theta_hat_k"] / out["se_theta"], rtol=1e-12)
        np.testing.assert_allclose(out["p_k"], 2 * stats.t.sf(np.abs(out["t_k"]), out["df"]), rtol=1e-10)
        assert (out["tau_hat_k2"] >= 0).all() and (out["df"] >= 1).all() and ((out["p_k"] >= 0) & (out["p_k"] <= 1)).all()


def reml_loglik(tau2, m, s2):
    """Restricted log-likelihood (up to a constant) at each tau2 in a vector, for one node's (S,) m and s2."""
    v = np.atleast_1d(tau2)[:, None] + s2[None, :]
    u = 1.0 / v
    theta = (u * m).sum(1) / u.sum(1)
    return -0.5 * (np.log(v).sum(1) + np.log(u.sum(1)) + (u * (m - theta[:, None]) ** 2).sum(1))


def test_reml_attains_the_maximum_of_the_restricted_likelihood_found_by_brute_force() -> None:
    rng = np.random.default_rng(13)
    grid = np.concatenate([[0.0], np.geomspace(1e-9, 1e3, 6000)])
    interior = boundary = 0
    for _ in range(150):
        S = int(rng.integers(3, 25))
        s2 = rng.uniform(0.01, 2.0, size=S)
        tau = float(rng.choice([0.0, 0.05, 0.3, 1.0]))
        m = 0.4 + tau * rng.normal(size=S) + np.sqrt(s2) * rng.normal(size=S)
        tau2_hat = meta_analysis(m[:, None], s2[:, None])["tau_hat_k2"][0]
        at_hat = reml_loglik(tau2_hat, m, s2)[0]
        assert at_hat >= reml_loglik(grid, m, s2).max() - 1e-8
        if tau2_hat > 0:
            interior += 1
            assert at_hat >= reml_loglik(tau2_hat * 1.001, m, s2)[0] - 1e-12
            assert at_hat >= reml_loglik(tau2_hat * 0.999, m, s2)[0] - 1e-12
        else:
            boundary += 1
    assert interior > 10 and boundary > 3  # both regimes were exercised


def test_meta_analysis_is_invariant_to_subject_order_and_equivariant_to_shift_and_scale() -> None:
    rng = np.random.default_rng(14)
    for method in ("REML", "MoM"):
        for _ in range(8):
            m, sigma2 = problem(rng, S=11, K=20)
            base = meta_analysis(m, sigma2, method=method)
            perm = rng.permutation(len(m))
            shuffled = meta_analysis(m[perm], sigma2[perm], method=method)
            for key in ("tau_hat_k2", "theta_hat_k", "se_theta", "t_k", "df", "p_k"):
                np.testing.assert_allclose(shuffled[key], base[key], rtol=1e-11, atol=1e-13)
            shift = meta_analysis(m + 5.0, sigma2, method=method)  # translation moves theta only
            np.testing.assert_allclose(shift["theta_hat_k"], base["theta_hat_k"] + 5.0, rtol=1e-10)
            for key in ("tau_hat_k2", "se_theta", "df"):
                np.testing.assert_allclose(shift[key], base[key], rtol=1e-8, atol=1e-12)
            c = 3.7
            scaled = meta_analysis(c * m, c**2 * sigma2, method=method)
            np.testing.assert_allclose(scaled["theta_hat_k"], c * base["theta_hat_k"], rtol=1e-10)
            np.testing.assert_allclose(scaled["tau_hat_k2"], c**2 * base["tau_hat_k2"], rtol=1e-8, atol=1e-12)
            np.testing.assert_allclose(scaled["se_theta"], c * base["se_theta"], rtol=1e-8)
            for key in ("t_k", "df", "p_k"):
                np.testing.assert_allclose(scaled[key], base[key], rtol=1e-7, atol=1e-12)


def test_a_subject_whose_alignment_is_uncertain_is_down_weighted() -> None:
    rng = np.random.default_rng(15)
    for _ in range(20):
        m = 0.2 * rng.normal(size=(10, 1))
        sigma2 = np.full((10, 1), 0.01)
        widened = sigma2.copy()
        widened[0] = 1.0
        share = lambda out: out["u_sk"][0, 0] / out["u_sk"].sum()
        assert share(meta_analysis(m, widened)) < share(meta_analysis(m, sigma2))
        assert share(meta_analysis(m, widened)) < 0.1 / 3  # equal weights would be 1/10


def test_down_weighting_beats_silently_averaging_an_undetermined_subject_in() -> None:
    rng = np.random.default_rng(16)
    S, K, theta = 12, 400, 0.5
    sigma2 = np.full((S, K), 0.02)
    sigma2[0] = 25.0  # subject 0's alignment is not determined by the data
    m = theta + 0.3 * rng.normal(size=(S, K)) + np.sqrt(sigma2) * rng.normal(size=(S, K))
    weighted = meta_analysis(m, sigma2)["theta_hat_k"]
    plain = one_sample_ttest(m)["theta_hat_k"]
    assert np.mean((weighted - theta) ** 2) < 0.5 * np.mean((plain - theta) ** 2)


def test_meta_analysis_recovers_known_parameters_from_simulated_maps() -> None:
    rng = np.random.default_rng(17)
    S, K, tau2, theta = 60, 3000, 0.25, 0.4
    sigma2 = rng.uniform(0.05, 0.6, size=(S, K))
    m = theta + np.sqrt(tau2) * rng.normal(size=(S, K)) + np.sqrt(sigma2) * rng.normal(size=(S, K))
    for method, tol in (("REML", 0.03), ("MoM", 0.03)):
        out = meta_analysis(m, sigma2, method=method)
        assert abs(out["tau_hat_k2"].mean() / tau2 - 1) < tol
        assert abs(out["theta_hat_k"].mean() - theta) < 0.01


@pytest.mark.parametrize("S,tau,low,high", [(30, 0.4, 0.035, 0.062), (30, 1.0, 0.042, 0.066), (80, 1.0, 0.042, 0.066),
                                            (12, 0.0, 0.0, 0.05)])
def test_the_test_keeps_its_size_under_the_null_and_never_exceeds_the_nominal_level(S, tau, low, high) -> None:
    """Calibrated when the weights are informative; conservative, never anti-conservative, when tau^2 is at the boundary."""
    rng = np.random.default_rng(18)
    K = 5000
    sigma2 = rng.uniform(0.05, 1.5, size=(S, K))
    m = tau * rng.normal(size=(S, K)) + np.sqrt(sigma2) * rng.normal(size=(S, K))  # theta = 0
    assert low < (meta_analysis(m, sigma2)["p_k"] < 0.05).mean() < high


def test_the_meta_analysis_is_more_powerful_than_the_plain_t_test_when_uncertainty_is_heterogeneous() -> None:
    rng = np.random.default_rng(19)
    S, K = 30, 2000
    sigma2 = np.where(rng.random((S, K)) < 0.3, 4.0, 0.02)
    m = 0.5 + 0.2 * rng.normal(size=(S, K)) + np.sqrt(sigma2) * rng.normal(size=(S, K))
    assert (meta_analysis(m, sigma2)["p_k"] < 0.05).mean() > (one_sample_ttest(m)["p_k"] < 0.05).mean() + 0.1


def test_zero_variances_never_divide_by_zero() -> None:
    rng = np.random.default_rng(20)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = rng.normal(size=(8, 5))
        sigma2 = rng.uniform(0.1, 1.0, size=(8, 5))
        sigma2[:, 1] = 0.0  # a node where every subject's alignment is perfectly determined
        one_exact = sigma2.copy()
        one_exact[0] = 0.0  # ... and one subject that is at every node
        for method, s2 in (("REML", sigma2), ("REML", one_exact), ("MoM", sigma2)):
            out = meta_analysis(m, s2, method=method)
            assert all(np.isfinite(out[k]).all() for k in ("tau_hat_k2", "theta_hat_k", "se_theta", "t_k", "df", "p_k"))
            assert out["u_sk"].shape == m.shape and (out["u_sk"] > 0).all()
        constant = meta_analysis(np.ones((6, 2)), np.zeros((6, 2)))  # nothing varies at all: the t statistic is undefined
        reference = one_sample_ttest(np.ones((6, 2)))
    np.testing.assert_array_equal(constant["theta_hat_k"], reference["theta_hat_k"])
    np.testing.assert_array_equal(constant["t_k"], reference["t_k"])
    np.testing.assert_array_equal(constant["se_theta"], np.zeros(2))
    np.testing.assert_array_equal(constant["p_k"], reference["p_k"])


@pytest.mark.parametrize("m,sigma2,match", [
    (np.ones((3, 4)), np.ones((3, 5)), "shape"), (np.ones(4), np.ones(4), "2-D"), (np.ones((1, 4)), np.ones((1, 4)), "two subjects"),
    (np.ones((3, 4)), -np.ones((3, 4)), "non-negative"), (np.full((3, 4), np.nan), np.ones((3, 4)), "finite"),
    (np.ones((3, 4)), np.full((3, 4), np.inf), "finite"),
])
def test_meta_analysis_rejects_malformed_input(m, sigma2, match) -> None:
    with pytest.raises(ValueError, match=match):
        meta_analysis(m, sigma2)


def test_meta_analysis_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        meta_analysis(np.ones((3, 2)), np.ones((3, 2)), method="ML")


# ---- meta_analysis_map -----------------------------------------------------------------------------------
def simulated_subjects(rng, S=9, M=6, K=5, V_range=(8, 14), spread=None):
    pis, taus, zs = [], [], []
    for s in range(S):
        V = int(rng.integers(*V_range))
        width = 0.3 if spread is None else spread[s]
        pis.append(coupling(rng, V, K)[None] * np.exp(width * rng.normal(size=(M, V, K))))
        taus.append(np.full(V, width))
        zs.append(rng.normal(loc=0.3, size=V))
    return pis, taus, zs, np.full(K, 1.0 / K)


def test_the_orchestrator_builds_maps_and_covariances_and_calls_the_meta_analysis() -> None:
    rng = np.random.default_rng(21)
    pis, taus, zs, nu = simulated_subjects(rng)  # a different number of vertices per subject
    out = meta_analysis_map(pis, taus, zs, nu)
    m = np.stack([np.mean([subject_maps(z, p, nu) for p in pi], axis=0) for pi, z in zip(pis, zs)])
    sigma2 = np.stack([np.diag(alignment_covariance(pi, z, nu)) for pi, z in zip(pis, zs)])
    expected = meta_analysis(m, sigma2)
    for key in RESULT_KEYS:
        np.testing.assert_allclose(out[key], expected[key], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(out["m"], m, rtol=1e-12)
    np.testing.assert_allclose(out["sigma2"], sigma2, rtol=1e-10, atol=1e-16)


def test_the_orchestrator_frees_each_block_before_the_next_is_produced() -> None:
    S, M, K, V = 7, 5, 4, 9
    nu = np.full(K, 1.0 / K)
    block = lambda s: coupling(np.random.default_rng(100 + s), V, K)[None] * np.exp(0.3 * np.random.default_rng(s).normal(size=(M, V, K)))
    taus = [np.full(V, 0.3)] * S
    zs = [np.random.default_rng(200 + s).normal(size=V) for s in range(S)]
    refs = []

    def lazily():
        for s in range(S):
            assert all(ref() is None for ref in refs), "an earlier block is still alive"
            refs.append(weakref.ref(b := block(s)))
            yield b
            del b

    out = meta_analysis_map(lazily(), taus, zs, nu)
    reference = meta_analysis_map([block(s) for s in range(S)], taus, zs, nu)
    for key in RESULT_KEYS:
        np.testing.assert_array_equal(out[key], reference[key])
    assert len(refs) == S


def test_the_diagnostics_show_who_was_down_weighted() -> None:
    rng = np.random.default_rng(23)
    spread = [2.0] + [0.05] * 9  # subject 0 has a far wider posterior
    pis, taus, zs, nu = simulated_subjects(rng, S=10, M=12, spread=spread)
    out = meta_analysis_map(pis, taus, zs, nu)
    assert out["mean_tau_phi"].shape == (10,) and out["mean_sigma2"].shape == (10,) and out["mean_weight"].shape == (10,)
    assert out["n_eff"].shape == (5,)
    assert out["mean_weight"].argmin() == 0 and out["mean_sigma2"].argmax() == 0 and out["mean_tau_phi"].argmax() == 0
    assert out["mean_weight"][0] < 0.5 * np.delete(out["mean_weight"], 0).min()
    np.testing.assert_allclose(out["mean_weight"].sum(), 1.0, rtol=1e-10)  # each column of weights sums to 1, so the subject means sum to 1
    assert (out["n_eff"] <= 10 + 1e-9).all() and (out["n_eff"] >= 1 - 1e-9).all()


def test_with_no_alignment_uncertainty_the_orchestrator_reduces_to_the_one_sample_t_test() -> None:
    rng = np.random.default_rng(24)
    pis, taus, zs, nu = simulated_subjects(rng, S=8, M=4)
    frozen = [np.stack([pi[0]] * 4) for pi in pis]  # every draw identical
    out = meta_analysis_map(frozen, taus, zs, nu)
    reference = one_sample_ttest(out["m"])
    for key in RESULT_KEYS:
        if key != "u_sk":
            np.testing.assert_allclose(out[key], reference[key], rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(out["sigma2"], 0.0, atol=1e-25)


def test_the_orchestrator_rejects_mismatched_inputs() -> None:
    rng = np.random.default_rng(25)
    pis, taus, zs, nu = simulated_subjects(rng, S=4)
    with pytest.raises(ValueError, match="same number of subjects"):
        meta_analysis_map(pis, taus[:-1], zs, nu)
    with pytest.raises(ValueError, match="vertices"):
        meta_analysis_map(pis, [t[:-1] for t in taus], zs, nu)
    with pytest.raises(ValueError, match="two subjects"):
        meta_analysis_map(pis[:1], taus[:1], zs[:1], nu)


# ---- sign_flip -------------------------------------------------------------------------------------------
def test_the_sign_flip_null_reproduces_exactly_from_its_declared_seed() -> None:
    contrast = np.random.default_rng(30).normal(0.1, 1.0, size=500)
    first = sign_flip_null(contrast, B=2000, seed=2026)
    np.testing.assert_array_equal(first, sign_flip_null(contrast, B=2000, seed=2026))
    assert not np.array_equal(first, sign_flip_null(contrast, B=2000, seed=2027))
    a, b = sign_flip_test(contrast, 500, 2000, 2026), sign_flip_test(contrast, 500, 2000, 2026)
    np.testing.assert_array_equal(a["null"], b["null"])
    assert a["p_value"] == b["p_value"] and a["seed"] == 2026 and a["B"] == 2000 and a["n_pairs"] == 500


@pytest.mark.parametrize("B", [1, 7, 250, 251, 1234])
def test_the_sign_flip_null_has_exactly_B_values_whatever_the_chunking(B: int) -> None:
    null = sign_flip_null(np.random.default_rng(31).normal(size=40), B=B, seed=1)
    assert null.shape == (B,) and np.isfinite(null).all()


def test_the_sign_flip_null_does_not_depend_on_n_jobs() -> None:
    contrast = np.random.default_rng(32).normal(0.05, 1.0, size=500)
    serial = sign_flip_null(contrast, B=1500, seed=2026, n_jobs=1)
    np.testing.assert_array_equal(serial, sign_flip_null(contrast, B=1500, seed=2026, n_jobs=2))


def test_the_sign_flip_null_is_the_exact_distribution_of_flipped_means() -> None:
    contrast = np.array([0.5, -1.2, 0.3, 2.0, -0.7, 0.9, 1.1, -0.4])
    exact = np.array([np.dot(signs, contrast) / 8 for signs in itertools.product([-1.0, 1.0], repeat=8)])
    null = sign_flip_null(contrast, B=40000, seed=5)
    assert all(np.isclose(exact, v, atol=1e-12).any() for v in np.unique(null))  # only real sign patterns occur
    assert abs(null.mean()) < 0.01 and abs(null.mean() - exact.mean()) < 0.01  # exact null is symmetric about 0
    assert abs(np.mean(np.isclose(null, exact.max())) - 1 / 256) < 0.002  # each sign pattern is equally likely


def test_the_sign_flip_p_value_counts_null_values_at_least_as_extreme_with_the_plus_one_correction() -> None:
    contrast = np.random.default_rng(33).normal(0.08, 1.0, size=500)
    out = sign_flip_test(contrast, 500, 999, 7)
    extreme = int((np.abs(out["null"]) >= abs(out["statistic"])).sum())
    assert out["statistic"] == pytest.approx(contrast.mean()) and out["p_value"] == (extreme + 1) / (999 + 1)
    assert 1 / 1000 <= out["p_value"] <= 1.0


def test_the_sign_flip_test_keeps_its_size_under_the_null_and_detects_a_shift() -> None:
    rng = np.random.default_rng(34)
    # B = 99: P(p <= 0.05) = 5 / 100 exactly under the null, whatever the (symmetric) distribution
    pvals = np.array([sign_flip_test(rng.standard_t(3, size=30), 30, 99, seed=i)["p_value"] for i in range(1500)])
    assert 0.035 < np.mean(pvals <= 0.05) < 0.065 and 0.7 < np.mean(pvals) / 0.5 < 1.3
    assert sign_flip_test(rng.normal(0.4, 1.0, size=500), 500, 199, seed=0)["p_value"] == 1 / 200


def test_the_sign_flip_test_rejects_an_undeclared_pair_count_and_bad_input() -> None:
    contrast = np.ones(300)
    with pytest.raises(ValueError, match="500.*300"):
        sign_flip_test(contrast, 500, 100, 0)
    for bad, match in ((np.ones((5, 2)), "1-D"), (np.array([1.0, np.nan]), "finite"), (np.array([]), "empty")):
        with pytest.raises(ValueError, match=match):
            sign_flip_null(bad, B=10, seed=0)
    with pytest.raises(ValueError, match="B"):
        sign_flip_null(np.ones(5), B=0, seed=0)
