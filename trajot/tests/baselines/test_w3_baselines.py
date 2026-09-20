from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trajot.baselines import (
    BASELINE_REGISTRY,
    AblatedModel,
    BaselineResult,
    BrainSync,
    ConnSRM,
    FUGW,
    FakeBaseline,
    NoAlign,
    OursAblated,
    OursFull,
    align_features,
    brainsync_Q,
    get_baseline,
    run_baseline,
    to_connectome_transform,
    upper_triangle_features,
)


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


def _planted_shared_connectomes(S: int = 6, R: int = 12, noise: float = 0.15, seed: int = 0) -> np.ndarray:
    """Subjects observe a shared region graph through independent permutations + noise."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(R, R))
    shared = shared @ shared.T
    np.fill_diagonal(shared, 0.0)
    mats = []
    for _ in range(S):
        perm = rng.permutation(R)
        P = np.eye(R)[perm]
        eps = noise * rng.normal(size=(R, R))
        eps = 0.5 * (eps + eps.T)
        np.fill_diagonal(eps, 0.0)
        C = P.T @ shared @ P + eps
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, 0.0)
        mats.append(C)
    return np.stack(mats, axis=0)


def _region_timeseries(connectomes: np.ndarray, T: int = 40, seed: int = 0) -> np.ndarray:
    """Build (S, R, T) region time courses whose correlation structure approximates each connectome."""
    rng = np.random.default_rng(seed)
    S, R, _ = connectomes.shape
    out = np.empty((S, R, T), dtype=np.float64)
    for s in range(S):
        C = connectomes[s]
        vals, vecs = np.linalg.eigh(0.5 * (C + C.T))
        vals = np.clip(vals, 0.0, None)
        A = vecs * np.sqrt(vals)  # (R, R)
        factors = rng.normal(size=(R, T))
        ts = A @ factors
        ts = (ts - ts.mean(axis=1, keepdims=True)) / np.maximum(ts.std(axis=1, keepdims=True), 1e-8)
        out[s] = ts
    return out


def _mean_pair_corr(mats: np.ndarray) -> float:
    S = mats.shape[0]
    feats = [mats[s][np.triu_indices(mats.shape[1], k=1)] for s in range(S)]
    cors = []
    for i in range(S):
        for j in range(i + 1, S):
            a, b = feats[i], feats[j]
            if a.std() > 0 and b.std() > 0:
                cors.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(cors)) if cors else 0.0


# --------------------------------------------------------------------------- B1
def test_noalign_is_identity_and_registered():
    run = _toy_connectomes()
    model = get_baseline("noalign")
    model.fit(run, {"run.seed": 0})
    out = model.transform(run[0])
    np.testing.assert_allclose(out, run[0])
    assert isinstance(model, NoAlign)


def test_fakebaseline_is_deterministic_for_same_seed():
    run = _toy_connectomes()
    a = FakeBaseline().fit(run, {"run.seed": 42}).transform(run[0])
    b = FakeBaseline().fit(run, {"run.seed": 42}).transform(run[0])
    c = FakeBaseline().fit(run, {"run.seed": 43}).transform(run[0])
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_registry_exports_expected_names():
    for name in (
        "noalign",
        "brainsync",
        "fugw",
        "conn_srm",
        "ours_full",
        "ours_ablated",
        "ours",
        "10_ours_full",
        "11_ours_ablated",
        "ablated",
        "fake",
    ):
        assert name in BASELINE_REGISTRY, name
        assert get_baseline(name) is not None


def test_harness_returns_baseline_result_schema():
    run = _toy_connectomes(S=4, R=5)
    result = run_baseline("noalign", {"connectomes": run, "subjects": ["a", "b", "c", "d"]}, None, {"run.seed": 0})
    assert isinstance(result, BaselineResult)
    assert result.name == "noalign"
    assert result.transforms.shape == (4, 5, 5)
    n_edges = 5 * 4 // 2
    assert result.aligned_features.shape == (4, n_edges)
    assert result.aligned_features.dtype == np.float64
    assert result.device == "cpu"
    assert isinstance(result.meta, dict)
    assert result.meta.get("algorithm")
    assert result.meta.get("device") == "cpu"
    # noalign features equal upper triangle of inputs
    np.testing.assert_allclose(result.aligned_features, upper_triangle_features(run))


def test_align_features_passthrough_and_embedding_path():
    run = _toy_connectomes(S=3, R=4)
    result = run_baseline("noalign", {"connectomes": run}, None, {})
    feats = align_features(result, None)
    np.testing.assert_allclose(feats, result.aligned_features)
    emb = np.random.default_rng(0).normal(size=(3, 4, 2)).astype(np.float32)
    out = align_features(result, emb)
    assert out.shape == (3, 4, 2)
    # Connectome transforms are already folded into aligned_features; embeddings pass through
    assert np.allclose(out, emb)


def test_fakebaseline_through_harness():
    run = _toy_connectomes(S=3, R=6)
    result = run_baseline("fake", {"connectomes": run}, None, {"run.seed": 1})
    assert result.transforms.shape == (3, 6, 6)
    for s in range(3):
        C = result.transforms[s]
        np.testing.assert_allclose(C, C.T, atol=1e-10)
        np.testing.assert_allclose(np.diag(C), 0.0, atol=1e-12)


# --------------------------------------------------------------------------- B2
def test_brainsync_fit_requires_timeseries():
    run = _toy_connectomes(S=3, R=8)
    with pytest.raises(ValueError, match="timeseries"):
        BrainSync().fit(run, {})


def test_brainsync_Q_orthogonal_and_raises_correlation():
    rng = np.random.default_rng(0)
    V, T = 8, 30
    X = rng.normal(size=(V, T))  # contract convention (V, T)
    # Planted orthogonal misalignment on the time axis: Y ≈ X W
    W, _ = np.linalg.qr(rng.normal(size=(T, T)))
    Y = X @ W + 0.05 * rng.normal(size=(V, T))
    Q = brainsync_Q(X, Y)
    assert Q.shape == (T, T)
    np.testing.assert_allclose(Q.T @ Q, np.eye(T), atol=1e-8)

    X_sync = X @ Q  # issue #5: X_aligned = X Q

    def mean_vertex_corr(A, B):
        cors = []
        for i in range(A.shape[0]):
            a, b = A[i], B[i]
            if a.std() > 0 and b.std() > 0:
                cors.append(np.corrcoef(a, b)[0, 1])
        return float(np.mean(cors))

    assert mean_vertex_corr(X_sync, Y) > mean_vertex_corr(X, Y)


def test_brainsync_vertex_space_transform_symmetric_zero_diag():
    rng = np.random.default_rng(1)
    R, T, S = 6, 24, 4
    shared = rng.normal(size=(R, T))
    connectomes = []
    ts = []
    for s in range(S):
        W, _ = np.linalg.qr(rng.normal(size=(R, R)))
        X = W @ shared + 0.05 * rng.normal(size=(R, T))
        ts.append(X)
        C = np.corrcoef(X)
        np.fill_diagonal(C, 0.0)
        connectomes.append(C)
    connectomes = np.stack(connectomes)
    model = BrainSync(space="vertex")
    model.fit(connectomes, {}, extra={"timeseries_run1": np.stack(ts)})
    out = model.transform(connectomes[0])
    assert out.shape == (R, R)
    np.testing.assert_allclose(out, out.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-12)
    assert np.isfinite(out).all()


def test_to_connectome_transform_vertex_Q_and_timeseries_path():
    rng = np.random.default_rng(2)
    R, T = 5, 20
    X = rng.normal(size=(R, T))
    Y = rng.normal(size=(R, T))
    C = np.corrcoef(X)
    np.fill_diagonal(C, 0.0)
    Qv = to_connectome_transform(np.eye(R), C)  # identity vertex Q
    np.testing.assert_allclose(Qv, C, atol=1e-10)

    Q = brainsync_Q(X, Y)  # (T, T) from (V, T) inputs
    C_al = to_connectome_transform(Q, timeseries=X)
    assert C_al.shape == (R, R)
    np.testing.assert_allclose(C_al, C_al.T, atol=1e-10)
    with pytest.raises(ValueError):
        to_connectome_transform(Q, C)  # time Q cannot act on spatial C alone


def test_brainsync_harness_and_improves_cross_run_similarity():
    run = _toy_connectomes(S=4, R=8, seed=3)
    # Region-resolution timeseries consistent with connectomes
    ts = _region_timeseries(run, T=48, seed=4)
    data = {
        "connectomes": run,
        "timeseries_run1": ts,
        "subjects": [f"s{i}" for i in range(4)],
    }
    result = run_baseline("brainsync", data, None, {})
    assert result.transforms.shape == (4, 8, 8)
    assert result.meta["algorithm"].startswith("brainsync")
    for s in range(4):
        C = result.transforms[s]
        np.testing.assert_allclose(C, C.T, atol=1e-8)
        np.testing.assert_allclose(np.diag(C), 0.0, atol=1e-10)

    # Vertex-space BrainSync on planted shared time courses should beat raw similarity
    rng = np.random.default_rng(5)
    R, T, S = 8, 40, 5
    shared = rng.normal(size=(R, T))
    ts_list, conn = [], []
    for s in range(S):
        W, _ = np.linalg.qr(rng.normal(size=(R, R)))
        X = W @ shared + 0.02 * rng.normal(size=(R, T))
        ts_list.append(X)
        C = np.corrcoef(X)
        np.fill_diagonal(C, 0.0)
        conn.append(C)
    conn = np.stack(conn)
    raw = _mean_pair_corr(conn)
    model = BrainSync(space="vertex").fit(conn, {}, extra={"timeseries_run1": np.stack(ts_list)})
    aligned = model.transform_all(conn, timeseries_run1=np.stack(ts_list))
    assert _mean_pair_corr(aligned) > raw


# --------------------------------------------------------------------------- B3
def test_fugw_real_pot_alignment():
    run = _toy_connectomes(S=4, R=16, seed=7)
    model = FUGW(max_iter=20)
    model.fit(run, {"run.seed": 0})
    out = model.transform(run[0])
    assert out.shape == (16, 16)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, out.T, atol=1e-8)
    np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-10)
    assert model.meta["device"] == "cpu"
    algo = str(model.meta.get("algorithm", "")).lower()
    assert "fugw" in algo or "gromov" in algo or "gw" in algo
    # Not a silent identity placeholder
    assert not np.allclose(out, run[0])

    result = run_baseline("fugw", {"connectomes": run}, None, {})
    assert result.transforms.shape == (4, 16, 16)
    assert result.device == "cpu"
    assert "fugw" in str(result.meta.get("algorithm", "")).lower() or "gw" in str(result.meta.get("algorithm", "")).lower()


def test_conn_srm_is_shared_response_not_eigen_truncation():
    run = _planted_shared_connectomes(S=6, R=12, noise=0.12, seed=8)
    model = ConnSRM(rank=4, n_iter=8)
    model.fit(run, {"model.r": 4})
    assert model.shared_response is not None
    assert model.shared_response.shape[0] == 4
    algo = str(model.meta["algorithm"]).lower()
    assert "srm" in algo or "shared" in algo
    assert "eigen" not in algo

    aligned = model.transform_all(run)
    assert aligned.shape == run.shape
    raw_sim = _mean_pair_corr(run)
    aligned_sim = _mean_pair_corr(aligned)
    noalign_sim = _mean_pair_corr(NoAlign().fit(run, {}).transform_all(run))
    assert aligned_sim > noalign_sim
    assert aligned_sim > raw_sim * 0.99  # SRM should not collapse similarity

    for s in range(run.shape[0]):
        C = aligned[s]
        np.testing.assert_allclose(C, C.T, atol=1e-8)
        np.testing.assert_allclose(np.diag(C), 0.0, atol=1e-10)


def test_conn_srm_through_harness():
    run = _planted_shared_connectomes(S=4, R=10, seed=9)
    result = run_baseline("conn_srm", {"connectomes": run}, None, {"model.r": 3})
    assert result.transforms.shape == (4, 10, 10)
    assert result.aligned_features.shape == (4, 10 * 9 // 2)
    assert "srm" in str(result.meta.get("algorithm", "")).lower() or "shared" in str(result.meta.get("algorithm", "")).lower()


# --------------------------------------------------------------------------- B4
@pytest.fixture(scope="module")
def synthetic_root(tmp_path_factory) -> Path:
    from trajot.inference.synthetic import make_synthetic_npz

    root = tmp_path_factory.mktemp("ours_synthetic")
    make_synthetic_npz(root, n_subjects=4, V=24, R=8, T=32, d=6, seed=11, planted_permutation=True)
    return root


def _load_synthetic_connectomes(root: Path) -> tuple[np.ndarray, list[str]]:
    from trajot.io.contract import load_connectomes

    return load_connectomes(root, subjects=None, run="1")


def _tiny_cfg(gauge: bool = True, epochs: int = 2) -> dict:
    return {
        "run.seed": 0,
        "model.K": 8,
        "model.r": 4,
        "model.d": 6,
        "model.m_draws": 2,
        "model.batch_subjects": 2,
        "model.sinkhorn.L": 5,
        "model.sinkhorn.eps": [0.1, 0.05],
        "model.beta.warmup_frac": 0.3,
        "model.beta.synthetic": 50.0,
        "model.sigma_f2": 1.0,
        "model.gauge_features": gauge,
        "model.prior.a_eps": 3.0,
        "model.prior.b_eps": 0.4,
        "model.prior.B_max_row_norm": 1.3,
        "model.prior.sigma_B": 1.0,
        "model.prior.sigma_F": 1.0,
        "model.prior.F0": 0.0,
        "model.encoder.p": 16,
        "model.encoder.n_blocks": 1,
        "model.encoder.heads": 2,
        "model.encoder.m_eigvecs": 4,
        "model.encoder.lambda_init": 1.0,
        "model.band.seconds": [10.0, 100.0],
        "model.band.bandwidth": 0.05,
        "model.band.weight": 0.0,
        "model.train.epochs": epochs,
        "model.train.lr": 0.003,
    }


def test_ours_registered_and_meta_gauge_differs(synthetic_root: Path):
    assert isinstance(get_baseline("ours_full"), OursFull)
    assert isinstance(get_baseline("ours_ablated"), OursAblated)
    assert isinstance(get_baseline("10_ours_full"), OursFull)
    assert isinstance(get_baseline("11_ours_ablated"), OursAblated)
    assert isinstance(get_baseline("ablated"), AblatedModel)

    mats, subjects = _load_synthetic_connectomes(synthetic_root)
    extra = {"data_root": synthetic_root, "epochs": 2}

    full = OursFull()
    full.fit(mats, _tiny_cfg(gauge=True), extra=extra)
    abl = OursAblated()
    abl.fit(mats, _tiny_cfg(gauge=False), extra=extra)

    assert full.meta["gauge_features"] is True
    assert abl.meta["gauge_features"] is False
    assert full.meta != abl.meta
    assert full.meta.get("source") in {"train", "artifacts"}
    assert abl.meta.get("source") in {"train", "artifacts"}

    out_full = full.transform(mats[0])
    out_abl = abl.transform(mats[0])
    assert out_full.shape == mats[0].shape
    assert out_abl.shape == mats[0].shape
    assert np.isfinite(out_full).all() and np.isfinite(out_abl).all()
    # Learned alignment is applied — not an unmodified identity copy
    assert not np.allclose(out_full, mats[0])
    assert not np.allclose(out_abl, mats[0])
    for out in (out_full, out_abl):
        np.testing.assert_allclose(out, out.T, atol=1e-8)
        np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-10)


def test_ours_harness_aligned_features_shape(synthetic_root: Path):
    mats, subjects = _load_synthetic_connectomes(synthetic_root)
    data = {
        "connectomes": mats,
        "subjects": subjects,
        "data_root": synthetic_root,
        "epochs": 2,
    }
    result = run_baseline("ours_full", data, None, _tiny_cfg(gauge=True))
    S, R, _ = mats.shape
    assert result.transforms.shape == (S, R, R)
    assert result.aligned_features.shape == (S, R * (R - 1) // 2)
    assert result.meta["gauge_features"] is True
    assert result.device == "cpu"
    assert np.isfinite(result.aligned_features).all()

    feats = align_features(result, None)
    np.testing.assert_allclose(feats, result.aligned_features)


def test_ours_artifacts_path_when_present(synthetic_root: Path, tmp_path: Path):
    from trajot.inference.train import load_train_data, save_artifacts, train
    from trajot.baselines.base import CfgView
    from trajot.runlog.parallel import pick_device

    mats, _ = _load_synthetic_connectomes(synthetic_root)
    nested = {
        "run": {"seed": 0},
        "model": {
            "K": 8,
            "d": 6,
            "r": 4,
            "m_draws": 2,
            "batch_subjects": 2,
            "sinkhorn": {"L": 5, "eps": [0.1, 0.05]},
            "beta": {"warmup_frac": 0.3, "synthetic": 50.0},
            "sigma_f2": 1.0,
            "gauge_features": True,
            "prior": {"sigma_B": 1.0, "sigma_F": 1.0, "F0": 0.0, "a_eps": 3.0, "b_eps": 0.4, "B_max_row_norm": 1.3},
            "encoder": {"p": 16, "n_blocks": 1, "heads": 2, "m_eigvecs": 4, "lambda_init": 1.0},
            "band": {"seconds": [10.0, 100.0], "bandwidth": 0.05, "weight": 0.0},
            "train": {"epochs": 1, "lr": 0.003},
        },
    }
    view = CfgView(nested)
    data = load_train_data(synthetic_root, view)
    result = train(view, data, pick_device(prefer_mps=False))
    run_dir = tmp_path / "run_ours"
    save_artifacts(run_dir, result)

    model = OursFull()
    model.fit(mats, _tiny_cfg(), extra={"run_dir": run_dir, "data_root": synthetic_root})
    assert model.meta.get("source") == "artifacts"
    out = model.transform(mats[0])
    assert out.shape == mats[0].shape
    assert np.isfinite(out).all()
    assert not np.allclose(out, mats[0])


def test_all_registered_methods_share_harness_path():
    """FakeBaseline and NoAlign run through the identical harness entry point."""
    run = _toy_connectomes(S=3, R=6)
    for name in ("fake", "noalign"):
        result = run_baseline(name, {"connectomes": run}, folds=None, cfg={"run.seed": 0})
        assert result.transforms.shape == (3, 6, 6)
        assert result.aligned_features.shape == (3, 15)
        assert isinstance(result.meta, dict)
