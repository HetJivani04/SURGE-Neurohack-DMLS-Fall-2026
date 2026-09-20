from __future__ import annotations

import numpy as np
import pytest

from trajot.eval.alignment_gain import alignment_gain, pair_gain
from trajot.eval.controls import (
    banded_coupling,
    random_permutation_control,
    shuffle_time_control,
)
from trajot.eval.folds import (
    FoldAssignment,
    identification_subjects,
    make_folds,
    sample_pairs,
)
from trajot.eval.identifiability import (
    nonidentifiable_count,
    nonidentifiable_pairs,
    posterior_width_flag,
)
from trajot.eval.identification import (
    IdentificationResult,
    accuracy_ci,
    flatten_features,
    identification_accuracy,
    identify_subjects,
    pearson_scores,
)
from trajot.eval.metrics import METHOD_KEYS, method_metrics_template, validate_metrics_payload
from trajot.eval.permutation import (
    PermutationResult,
    null_max,
    permutation_null,
    permutation_p,
    sign_flip_permutation,
)


def _vec_to_connectome(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.shape != (6,):
        raise ValueError(v.shape)
    C = np.zeros((4, 4), dtype=np.float64)
    iu = np.triu_indices(4, k=1)
    C[iu] = v
    C = C + C.T
    np.fill_diagonal(C, 0.0)
    return C


def _synthetic_unique_connectomes() -> np.ndarray:
    base = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=np.float64,
    )
    return np.stack([_vec_to_connectome(v) for v in base], axis=0)


def _sym_connectomes(S: int, R: int, seed: int, noise: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    run1 = np.empty((S, R, R), dtype=np.float64)
    run2 = np.empty((S, R, R), dtype=np.float64)
    for s in range(S):
        A = rng.normal(size=(R, R))
        C = A @ A.T
        C = C / max(np.std(C), 1e-12)
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 0.0)
        N = rng.normal(scale=noise, size=(R, R))
        N = 0.5 * (N + N.T)
        np.fill_diagonal(N, 0.0)
        run1[s] = C
        run2[s] = C + N
    return run1, run2


# ---------------------------------------------------------------------------
# 1. flatten_features
# ---------------------------------------------------------------------------

def test_flatten_features_shape_and_dtype():
    S, R = 5, 7
    x = np.random.default_rng(0).normal(size=(S, R, R))
    flat = flatten_features(x)
    assert flat.shape == (S, R * (R - 1) // 2)
    assert flat.dtype == np.float64
    # Values match upper triangle k=1
    iu = np.triu_indices(R, k=1)
    np.testing.assert_allclose(flat[0], x[0][iu])


def test_flatten_features_passthrough_2d():
    x = np.random.default_rng(1).normal(size=(4, 10))
    flat = flatten_features(x)
    assert flat.shape == (4, 10)
    np.testing.assert_allclose(flat, x)


# ---------------------------------------------------------------------------
# 2. pearson_scores + identification
# ---------------------------------------------------------------------------

def test_pearson_scores_identical_and_orthogonal():
    q = np.random.default_rng(2).normal(size=(6, 20))
    scores = pearson_scores(q, q)
    assert scores.shape == (6, 6)
    np.testing.assert_allclose(np.diag(scores), np.ones(6), atol=1e-10)

    # Centered orthogonal rows -> Pearson corr 0
    orth = np.zeros((2, 20))
    orth[0] = np.concatenate([np.ones(10), -np.ones(10)])
    orth[1] = np.concatenate([np.ones(5), -np.ones(5), np.ones(5), -np.ones(5)])
    # Make row1 explicitly orthogonal to row0 after centering (already is via construction check)
    v0 = orth[0] - orth[0].mean()
    v1 = orth[1] - orth[1].mean()
    if abs(float(v0 @ v1)) > 1e-12:
        # Gram-Schmidt fallback
        orth[1] = v1 - (float(v0 @ v1) / float(v0 @ v0)) * v0
    cross = pearson_scores(orth[:1], orth[1:])
    assert abs(float(cross[0, 0])) < 1e-8

    # Independent noise -> low |corr|
    rng = np.random.default_rng(21)
    a = rng.normal(size=(1, 50))
    b = rng.normal(size=(1, 50))
    noise_corr = float(pearson_scores(a, b)[0, 0])
    assert abs(noise_corr) < 0.5


def test_identification_accuracy_perfect_when_runs_equal():
    run1 = _synthetic_unique_connectomes()
    run2 = run1.copy()
    res = identification_accuracy(run1, run2, metric="pearson")
    assert isinstance(res, IdentificationResult)
    assert res.accuracy == 1.0
    assert res.correct_mask.all()
    assert res.metric == "pearson"

    legacy = identify_subjects(run1, run2)
    assert legacy.accuracy == 1.0


def test_identification_perfect_and_zero_accuracy_edges():
    run1 = _synthetic_unique_connectomes()
    run2 = run1.copy()
    perfect = identify_subjects(run1, run2)
    assert perfect.accuracy == 1.0
    assert perfect.correct_mask.all()

    shifted = run2[[1, 2, 3, 0]]
    zero = identify_subjects(run1, shifted)
    assert zero.accuracy == 0.0
    assert (~zero.correct_mask).all()


def test_identification_independent_noise_near_chance():
    rng = np.random.default_rng(3)
    S, R = 20, 8
    run1 = rng.normal(size=(S, R, R))
    run2 = rng.normal(size=(S, R, R))
    res = identification_accuracy(run1, run2)
    # Chance is 1/S; allow generous band for finite sample
    assert 0.0 <= res.accuracy <= 0.35


# ---------------------------------------------------------------------------
# 3. accuracy_ci
# ---------------------------------------------------------------------------

def test_accuracy_ci_contains_estimate_and_bounds():
    correct = np.array([True] * 40 + [False] * 10)
    acc = float(np.mean(correct))
    lo, hi = accuracy_ci(correct, n_boot=2000, seed=0)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= acc <= hi

    lo2, hi2 = accuracy_ci(np.ones(30, dtype=bool), n_boot=500, seed=1)
    assert lo2 >= 0.99
    assert hi2 <= 1.0


def test_accuracy_ci_on_point_mass():
    lo, hi = accuracy_ci(np.ones(16, dtype=bool), n_boot=200, seed=0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. make_folds
# ---------------------------------------------------------------------------

def test_make_folds_unique_without_replacement_deterministic():
    subjects = ["s01", "s02", "s03", "s04", "s05"]
    fold = make_folds(subjects, scheme="pairs_without_replacement", n_pairs=8, seed=42)
    assert isinstance(fold, FoldAssignment)
    assert fold.n_pairs == 8
    assert fold.seed == 42
    assert len(fold.pairs) == 8
    assert len(set(fold.pairs)) == 8
    assert all(a != b for a, b in fold.pairs)
    assert "without replacement" in fold.procedure

    again = make_folds(subjects, scheme="pairs_without_replacement", n_pairs=8, seed=42)
    assert again.pairs == fold.pairs
    other = make_folds(subjects, scheme="pairs_without_replacement", n_pairs=8, seed=43)
    assert other.pairs != fold.pairs


def test_make_folds_lexicographic_pool_ordered():
    subjects = ["b", "a", "c"]
    fold = make_folds(subjects, scheme="pairs_without_replacement", n_pairs=6, seed=0)
    # All 3*2=6 ordered distinct pairs, drawn without replacement
    assert set(fold.pairs) == {("a", "b"), ("a", "c"), ("b", "a"), ("b", "c"), ("c", "a"), ("c", "b")}


def test_make_folds_rejects_too_many_pairs():
    with pytest.raises(ValueError):
        make_folds(["a", "b"], scheme="pairs_without_replacement", n_pairs=5, seed=0)


def test_sample_pairs_without_replacement_when_possible():
    subjects = ["001", "002", "003", "004"]
    a = sample_pairs(subjects, n_pairs=6, seed=9)
    b = sample_pairs(subjects, n_pairs=6, seed=9)
    c = sample_pairs(subjects, n_pairs=6, seed=10)

    assert a == b
    assert a != c
    assert all(x != y for x, y in a)
    assert len(set(a)) == len(a)


def test_identification_subjects_from_manifest():
    pd = pytest.importorskip("pandas")
    rows = []
    for sid in ["sub-01", "sub-02", "sub-03"]:
        for run in ["1", "2"]:
            rows.append({"subject_id": sid, "run_id": run, "n_volumes": 132})
    # single-run subject excluded
    rows.append({"subject_id": "sub-04", "run_id": "1", "n_volumes": 132})
    manifest = pd.DataFrame(rows)
    got = identification_subjects(manifest)
    assert got == ["sub-01", "sub-02", "sub-03"]


# ---------------------------------------------------------------------------
# 5. alignment_gain
# ---------------------------------------------------------------------------

def test_alignment_gain_zero_when_aligned_equals_raw():
    raw1, raw2 = _sym_connectomes(S=8, R=6, seed=5)
    pairs = [(i, i) for i in range(8)]
    gain = alignment_gain(raw1, raw2, raw1, raw2, pairs)
    assert gain == pytest.approx(0.0, abs=1e-12)
    per = pair_gain(raw1, raw2, raw1, raw2, pairs)
    assert per.shape == (8,)
    np.testing.assert_allclose(per, 0.0, atol=1e-12)


def test_alignment_gain_positive_with_stronger_shared_signal():
    rng = np.random.default_rng(6)
    S, R = 10, 8
    true = rng.normal(size=(S, R, R))
    true = 0.5 * (true + true.transpose(0, 2, 1))

    def _sym_noise(scale):
        n = rng.normal(scale=scale, size=true.shape)
        return 0.5 * (n + n.transpose(0, 2, 1))

    # Raw runs are noisy, weak fingerprint
    raw1 = true + _sym_noise(2.5)
    raw2 = true + _sym_noise(2.5)
    # Aligned runs recover the subject fingerprint
    aligned1 = true + _sym_noise(0.05)
    aligned2 = true + _sym_noise(0.05)

    pairs = [(i, i) for i in range(S)]
    gain = alignment_gain(aligned1, aligned2, raw1, raw2, pairs)
    assert gain > 0.05


def test_alignment_gain_cross_subject_pairs():
    raw1, raw2 = _sym_connectomes(S=6, R=5, seed=8)
    pairs = [(0, 1), (2, 3), (4, 5)]
    gain_same = alignment_gain(raw1, raw2, raw1, raw2, pairs)
    assert gain_same == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 6. nonidentifiable_count
# ---------------------------------------------------------------------------

def test_nonidentifiable_count_inside_vs_above_null():
    rng = np.random.default_rng(9)
    null = rng.normal(0.0, 1.0, size=2000)
    n = 40

    gains_inside = np.full(n, float(np.median(null)))
    count_in, flags_in = nonidentifiable_count(gains_inside, null, alpha=0.05)
    assert count_in == n
    assert flags_in.shape == (n,)
    assert flags_in.dtype == np.bool_
    assert flags_in.all()

    gains_high = np.full(n, float(null.max()) + 50.0)
    count_hi, flags_hi = nonidentifiable_count(gains_high, null, alpha=0.05)
    assert count_hi == 0
    assert not flags_hi.any()

    assert nonidentifiable_pairs(gains_inside, null, alpha=0.05) == n


def test_nonidentifiable_pairs_backward_compat_correct_mask():
    mask = np.array([True, False, True, False, False])
    assert nonidentifiable_pairs(mask) == 3


def test_posterior_width_flag():
    tau = np.array([0.1, 0.5, 2.0, 1.01], dtype=np.float64)
    flags = posterior_width_flag(tau, threshold=1.0)
    assert flags.dtype == np.bool_
    np.testing.assert_array_equal(flags, [False, False, True, True])

    tau_s = np.array([[0.2, 0.4], [3.0, 3.0], [1.0, 0.1]])
    flags2 = posterior_width_flag(tau_s, threshold=1.0)
    assert flags2.shape == (3,)
    np.testing.assert_array_equal(flags2, [False, True, False])


# ---------------------------------------------------------------------------
# 7. null_max + metrics schema
# ---------------------------------------------------------------------------

def test_permutation_result_has_null_max():
    run1 = _synthetic_unique_connectomes()
    run2 = run1.copy()
    out = permutation_p(run1, run2, B=17, seed=123)
    assert isinstance(out, PermutationResult)
    assert out.null_max == pytest.approx(float(np.max(out.null_distribution)))
    assert null_max(out) == out.null_max

    ge = int(np.sum(out.null_distribution >= out.observed))
    expected = (ge + 1) / (17 + 1)
    assert out.p_value == expected


def test_permutation_null_generic_and_sign_flip():
    values = np.array([0.5, -0.2, 0.3, -0.1, 0.4], dtype=np.float64)
    items = values.copy()

    def mean_stat(arr):
        return float(np.mean(arr))

    res = permutation_null(mean_stat, items, B=50, seed=3)
    assert res.null_distribution.shape == (50,)
    assert res.null_max == pytest.approx(float(np.max(res.null_distribution)))
    assert 0.0 < res.p_value <= 1.0
    assert res.observed == pytest.approx(float(np.mean(values)))

    flipped = sign_flip_permutation(values, B=200, seed=4)
    assert flipped.shape == (200,)
    # Symmetric values: null mean near 0
    assert abs(float(np.mean(flipped))) < 0.25


def test_method_keys_include_null_max_and_validate_none():
    assert "null_max" in METHOD_KEYS
    assert "per_pair_uncertainty" in METHOD_KEYS
    assert METHOD_KEYS >= {
        "ident_accuracy",
        "ident_ci",
        "perm_p",
        "null_max",
        "alignment_gain",
        "nonidentifiable_pairs",
        "per_pair_uncertainty",
        "per_pair_flags",
    }

    stats = method_metrics_template()
    stats["ident_accuracy"] = 0.4
    stats["perm_p"] = 0.5
    stats["null_max"] = None
    stats["per_pair_uncertainty"] = None
    payload = {
        "experiment": "00_noalign",
        "run_id": "r1",
        "n_subjects": 4,
        "n_pairs": 500,
        "pairs_seed": 2026,
        "permutations_B": 100,
        "methods": {"noalign": stats},
        "beta": None,
        "notes": "",
    }
    validate_metrics_payload(payload)

    bad = dict(payload)
    bad_methods = {"noalign": dict(stats)}
    bad_methods["noalign"]["unexpected_key"] = 1
    bad["methods"] = bad_methods
    with pytest.raises(ValueError):
        validate_metrics_payload(bad)

    missing = dict(payload)
    miss_stats = {k: v for k, v in stats.items() if k != "null_max"}
    missing["methods"] = {"noalign": miss_stats}
    with pytest.raises(ValueError):
        validate_metrics_payload(missing)


# ---------------------------------------------------------------------------
# 8. controls N1 / N2
# ---------------------------------------------------------------------------

def test_banded_coupling_returns_control_result():
    run1, run2 = _sym_connectomes(S=6, R=8, seed=11)
    ctrl = banded_coupling(run1, run2, band=2, seed=0)
    assert 0.0 <= ctrl.mean_accuracy <= 1.0
    assert ctrl.flags.dtype == np.bool_
    assert ctrl.flags.shape == (6,)
    assert isinstance(ctrl.meta, dict)
    assert ctrl.meta.get("control") == "N1"

    rng = np.random.default_rng(0)
    coords = rng.normal(size=(8, 3))
    ctrl2 = banded_coupling(run1, run2, band=3, seed=1, coords=coords)
    assert isinstance(ctrl2.mean_accuracy, float)


def test_shuffle_time_control_returns_control_result_and_drives_to_chance():
    rng = np.random.default_rng(12)
    S, V, T = 8, 12, 40
    # Distinct subject-specific temporal structure
    ts1 = np.empty((S, V, T), dtype=np.float64)
    ts2 = np.empty((S, V, T), dtype=np.float64)
    for s in range(S):
        latent = rng.normal(size=(4, T))
        mixing = rng.normal(size=(V, 4))
        shared = mixing @ latent
        ts1[s] = shared + rng.normal(scale=0.3, size=(V, T))
        ts2[s] = shared + rng.normal(scale=0.3, size=(V, T))

    ctrl = shuffle_time_control(ts1, run2=ts2, seed=3)
    assert 0.0 <= ctrl.mean_accuracy <= 1.0
    assert ctrl.flags.dtype == np.bool_
    assert ctrl.flags.shape == (S,)
    assert isinstance(ctrl.meta, dict)
    assert ctrl.meta.get("control") == "N2"
    # Chance is 1/S; allow sampling noise
    assert ctrl.mean_accuracy < 0.5


def test_shuffle_time_control_connectome_proxy():
    run1, run2 = _sym_connectomes(S=5, R=6, seed=13)
    ctrl = shuffle_time_control(run1, run2=run2, seed=2)
    assert isinstance(ctrl.mean_accuracy, float)
    assert "limitation" in ctrl.meta or ctrl.meta.get("input") == "connectomes"


def test_random_permutation_control_contract():
    run1 = _synthetic_unique_connectomes()
    run2 = run1.copy()
    ctrl = random_permutation_control(run1, run2, seed=4)

    assert 0.0 <= ctrl.mean_accuracy <= 1.0
    assert 0.0 <= ctrl.per_pair_uncertainty <= 1.0
    assert ctrl.flags.dtype == np.bool_
    assert ctrl.flags.shape == (run1.shape[0],)


# ---------------------------------------------------------------------------
# 9. package exports
# ---------------------------------------------------------------------------

def test_eval_package_exports():
    import trajot.eval as ev

    required = [
        "alignment_gain",
        "pair_gain",
        "identify_subjects",
        "identification_accuracy",
        "flatten_features",
        "pearson_scores",
        "accuracy_ci",
        "nonidentifiable_pairs",
        "nonidentifiable_count",
        "posterior_width_flag",
        "permutation_p",
        "permutation_null",
        "sign_flip_permutation",
        "null_max",
        "random_permutation_control",
        "banded_coupling",
        "shuffle_time_control",
        "make_folds",
        "identification_subjects",
        "FoldAssignment",
        "validate_metrics_payload",
        "write_metrics",
        "IdentificationResult",
        "PermutationResult",
        "ControlResult",
        "method_metrics_template",
    ]
    for name in required:
        assert hasattr(ev, name), f"missing export: {name}"
