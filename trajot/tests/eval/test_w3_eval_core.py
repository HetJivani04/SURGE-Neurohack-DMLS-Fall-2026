from __future__ import annotations

import numpy as np

from trajot.eval.controls import random_permutation_control
from trajot.eval.folds import sample_pairs
from trajot.eval.identification import identify_subjects
from trajot.eval.permutation import permutation_p


def _vec_to_connectome(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    # For R=4, upper triangle has 6 entries.
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


def test_identification_perfect_and_zero_accuracy_edges():
    run1 = _synthetic_unique_connectomes()
    run2 = run1.copy()
    perfect = identify_subjects(run1, run2)
    assert perfect.accuracy == 1.0
    assert perfect.correct_mask.all()

    # Circular shift has no fixed points for 4 elements.
    shifted = run2[[1, 2, 3, 0]]
    zero = identify_subjects(run1, shifted)
    assert zero.accuracy == 0.0
    assert (~zero.correct_mask).all()


def test_permutation_p_uses_count_plus_one_formula():
    run1 = _synthetic_unique_connectomes()
    run2 = run1.copy()
    out = permutation_p(run1, run2, B=17, seed=123)

    ge = int(np.sum(out.null_distribution >= out.observed))
    expected = (ge + 1) / (17 + 1)
    assert out.p_value == expected


def test_sample_pairs_reproducible_and_no_self_pairs():
    subjects = ["001", "002", "003", "004"]
    a = sample_pairs(subjects, n_pairs=20, seed=9)
    b = sample_pairs(subjects, n_pairs=20, seed=9)
    c = sample_pairs(subjects, n_pairs=20, seed=10)

    assert a == b
    assert a != c
    assert all(x != y for x, y in a)


def test_random_permutation_control_contract():
    run1 = _synthetic_unique_connectomes()
    run2 = run1.copy()
    ctrl = random_permutation_control(run1, run2, seed=4)

    assert 0.0 <= ctrl.mean_accuracy <= 1.0
    assert 0.0 <= ctrl.per_pair_uncertainty <= 1.0
    assert ctrl.flags.dtype == np.bool_
    assert ctrl.flags.shape == (run1.shape[0],)
