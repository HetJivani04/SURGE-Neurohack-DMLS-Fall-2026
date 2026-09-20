from __future__ import annotations

import numpy as np

from trajot.baselines import FakeBaseline, get_baseline


def _toy_connectomes(S: int = 5, R: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mats = np.empty((S, R, R), dtype=np.float64)
    for s in range(S):
        A = rng.normal(size=(R, R))
        C = A @ A.T
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 0.0)
        mats[s] = C
    return mats


def test_noalign_is_identity_and_registered():
    run = _toy_connectomes()
    model = get_baseline("noalign")
    model.fit(run, {"run.seed": 0})

    out = model.transform(run[0])
    np.testing.assert_allclose(out, run[0])


def test_fakebaseline_is_deterministic_for_same_seed():
    run = _toy_connectomes()

    a = FakeBaseline().fit(run, {"run.seed": 42}).transform(run[0])
    b = FakeBaseline().fit(run, {"run.seed": 42}).transform(run[0])
    c = FakeBaseline().fit(run, {"run.seed": 43}).transform(run[0])

    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)
