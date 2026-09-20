"""Posterior-gated tau-shrink transform for OursFull (Task 2).

Primary scientific criterion (real data): held-out run-2 residual under
``transform_mode='posterior_shrink'`` vs noalign / point_procrustes.
Synthetic recovery is secondary and must not be claimed without measurement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trajot.baselines.ours import (
    OursAblated,
    OursFull,
    _lambda_from_tau,
    _local_shrinkage,
    mix_identity_orthogonal,
    pool_pi_to_regions,
)


def _sym(C: np.ndarray) -> np.ndarray:
    out = 0.5 * (C + C.T)
    np.fill_diagonal(out, 0.0)
    return out


def _random_connectomes(S: int, R: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mats = np.empty((S, R, R), dtype=np.float64)
    for s in range(S):
        A = rng.normal(size=(R, R))
        mats[s] = _sym(A @ A.T)
    return mats


def _orthogonal(R: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U, _, Vt = np.linalg.svd(rng.normal(size=(R, R)))
    return U @ Vt


def _write_posterior_artifacts(
    run_dir: Path,
    *,
    S: int,
    R: int,
    K: int,
    r: int,
    subject_ids: list[str] | None = None,
    pi_shapes: list[tuple[int, int]] | None = None,
    taus: list[np.ndarray] | None = None,
    seed: int = 0,
) -> Path:
    """Write template + posterior_samples + tau_phi artifacts for OursFull.fit."""
    art = Path(run_dir) / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if subject_ids is None:
        subject_ids = [f"{i + 1:03d}" for i in range(S)]
    B = rng.normal(size=(K, r)) / np.sqrt(r)
    np.savez(
        art / "template.npz",
        B=B,
        F_bar=rng.normal(size=(K, 4)),
        nu=np.full(K, 1.0 / K),
        eps=np.ones(S, dtype=np.float64),
        subject_ids=np.array(subject_ids),
    )
    post = {}
    tau_out = {}
    for s, sid in enumerate(subject_ids):
        V = pi_shapes[s][0] if pi_shapes else R
        Ks = pi_shapes[s][1] if pi_shapes else K
        pi = np.abs(rng.normal(size=(2, V, Ks))) + 0.05
        pi = pi / pi.sum(axis=(1, 2), keepdims=True)
        post[f"sub-{sid}"] = pi
        if taus is not None:
            tau_out[f"sub-{sid}"] = np.asarray(taus[s], dtype=np.float64)
        else:
            tau_out[f"sub-{sid}"] = np.full(V, 0.05, dtype=np.float64)
    np.savez_compressed(art / "posterior_samples.npz", **post)
    np.savez(art / "tau_phi.npz", **tau_out)
    return run_dir


class _ExplodingEMD:
    """ot.emd stand-in that fails the test if the posterior path re-solves EMD."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("ot.emd must not be called when posterior means are loaded")


# --------------------------------------------------------------------------- helpers
def test_pool_pi_to_regions_already_region_level_returns_copy():
    rng = np.random.default_rng(0)
    pi = rng.random((8, 5))
    out = pool_pi_to_regions(pi, None, 8)
    assert out.shape == (8, 5)
    np.testing.assert_allclose(out, pi)
    assert out is not pi


def test_pool_pi_to_regions_mass_preserving_block_sum():
    # V=6 vertices → R=3 regions, K=2
    pi = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [0.0, 3.0],
            [1.0, 1.0],
            [0.0, 0.5],
            [4.0, 0.0],
        ]
    )
    region_index = np.array([0, 0, 1, 1, 2, 2])
    out = pool_pi_to_regions(pi, region_index, 3)
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out.sum(), pi.sum())
    np.testing.assert_allclose(out[0], [3.0, 1.0])
    np.testing.assert_allclose(out[1], [1.0, 4.0])
    np.testing.assert_allclose(out[2], [4.0, 0.5])


def test_pool_pi_to_regions_requires_index_when_v_neq_r():
    with pytest.raises(ValueError, match="region_index"):
        pool_pi_to_regions(np.ones((6, 4)), None, 4)


def test_mix_identity_orthogonal_limits_and_orthogonality():
    R = 10
    Q = _orthogonal(R, seed=3)
    I_hat = mix_identity_orthogonal(Q, 0.0)
    Q_hat = mix_identity_orthogonal(Q, 1.0)
    np.testing.assert_allclose(I_hat.T @ I_hat, np.eye(R), atol=1e-8)
    np.testing.assert_allclose(Q_hat.T @ Q_hat, np.eye(R), atol=1e-8)
    np.testing.assert_allclose(np.abs(I_hat), np.eye(R), atol=1e-6)
    np.testing.assert_allclose(np.abs(Q_hat), np.abs(Q), atol=1e-6)
    mid = mix_identity_orthogonal(Q, 0.5)
    np.testing.assert_allclose(mid.T @ mid, np.eye(R), atol=1e-8)


def test_lambda_tau_gates():
    tau0 = 0.1
    assert _lambda_from_tau(None, tau0) == 1.0
    assert _lambda_from_tau(0.0, tau0) == pytest.approx(1.0)
    assert _lambda_from_tau(1e6, tau0) == pytest.approx(0.0, abs=1e-12)
    mid = _lambda_from_tau(tau0, tau0)
    assert mid == pytest.approx(0.5)


def test_local_shrinkage_matches_task1_formula():
    C = _sym(np.random.default_rng(1).normal(size=(6, 6)))
    Q = _orthogonal(6, seed=2)
    np.testing.assert_allclose(_local_shrinkage(C, Q, 0.0), C)
    np.testing.assert_allclose(_local_shrinkage(C, Q, 1.0), Q.T @ C @ Q)


def test_transform_mode_defaults_and_ablation():
    assert OursFull.transform_mode == "posterior_shrink"
    assert OursFull.tau0 == 0.1
    assert OursAblated.transform_mode == "point_procrustes"
    assert OursAblated.gauge_features is False
    full = OursFull()
    abl = OursAblated()
    assert full.transform_mode == "posterior_shrink"
    assert full.tau0 == 0.1
    assert abl.transform_mode == "point_procrustes"
    assert abl.gauge_features is False


def test_docstring_honesty_point_procrustes_does_not_claim_bbt_drive():
    abl_doc = (OursAblated.__doc__ or "").lower()
    full_doc = (OursFull.__doc__ or "").lower()
    assert "point_procrustes" in abl_doc
    assert "not" in abl_doc and ("bb" in abl_doc or "b b" in abl_doc or "hierarchy" in abl_doc or "hierarchical" in abl_doc)
    # Ablated doc must state hierarchy/ posterior is not used in the map
    assert any(phrase in abl_doc for phrase in (
        "hierarchy is not",
        "not used in the map",
        "not** used",
        "do not drive",
        "not** posterior",
        "posterior samples may still load",
    ))
    # Full doc describes posterior_shrink honestly and does not claim unmeasured wins
    assert "posterior" in full_doc
    assert "posterior_shrink" in full_doc
    assert "point_procrustes" in full_doc
    # Must not claim BB^T drives the *point_procrustes* mode
    # Find the point_procrustes section and ensure it references C_pop, not B B^T as driver
    idx = full_doc.find("point_procrustes")
    assert idx >= 0
    section = full_doc[idx : idx + 400]
    assert "c_pop" in section or "population mean" in section
    assert "bb^t" not in section and "b b^t" not in section


# --------------------------------------------------------------------------- posterior path
def test_posterior_path_uses_pi_means_and_skips_emd(tmp_path: Path, monkeypatch):
    import ot

    S, R, K, r = 3, 8, 8, 4
    mats = _random_connectomes(S, R, seed=5)
    run_dir = _write_posterior_artifacts(tmp_path / "run_post", S=S, R=R, K=K, r=r, seed=6)
    # Low τ → λ≈1 → transform is active Procrustes, not identity
    taus = [np.full(R, 0.01) for _ in range(S)]
    run_dir = _write_posterior_artifacts(
        tmp_path / "run_post2", S=S, R=R, K=K, r=r, taus=taus, seed=6
    )

    boom = _ExplodingEMD()
    monkeypatch.setattr(ot, "emd", boom)

    model = OursFull()
    model.fit(mats, {"run.seed": 0, "model.K": K, "model.r": r}, extra={"run_dir": run_dir})
    assert model.meta["source"] == "artifacts"
    assert model.meta["transform"] == "posterior_shrink_tau_gated"
    assert model.meta["reference_geometry"] == "C_bar_BBt"
    assert len(model.pi_means) == S
    assert boom.calls == 0, "posterior fit must not re-run ot.emd when means are loaded"

    out = model.transform(mats[0])
    assert boom.calls == 0
    assert out.shape == (R, R)
    np.testing.assert_allclose(out, out.T, atol=1e-8)
    np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-10)
    assert np.isfinite(out).all()
    assert not np.allclose(out, mats[0])

    aligned = model.transform_all(mats)
    assert boom.calls == 0
    assert aligned.shape == mats.shape


def test_posterior_path_pools_vertex_pi_with_region_index(tmp_path: Path, monkeypatch):
    import ot

    S, R, K, r = 2, 6, 6, 3
    V = 12  # vertex-level posterior
    mats = _random_connectomes(S, R, seed=7)
    region_index = np.arange(V) % R
    taus = [np.full(V, 0.02) for _ in range(S)]
    run_dir = _write_posterior_artifacts(
        tmp_path / "run_v",
        S=S,
        R=R,
        K=K,
        r=r,
        pi_shapes=[(V, K), (V, K)],
        taus=taus,
        seed=8,
    )
    boom = _ExplodingEMD()
    monkeypatch.setattr(ot, "emd", boom)

    model = OursFull()
    model.fit(
        mats,
        {"run.seed": 0},
        extra={"run_dir": run_dir, "region_index": region_index},
    )
    assert model.meta["transform"] == "posterior_shrink_tau_gated"
    assert boom.calls == 0
    assert model.pairwise_gamma(0, 1).shape == (R, R)
    out = model.transform(mats[0])
    assert out.shape == (R, R)
    assert not np.allclose(out, mats[0])


def test_tau_infinite_shrinks_to_identity(tmp_path: Path):
    S, R, K, r = 2, 6, 6, 3
    mats = _random_connectomes(S, R, seed=9)
    taus = [np.full(R, 1e6) for _ in range(S)]
    run_dir = _write_posterior_artifacts(
        tmp_path / "run_hi", S=S, R=R, K=K, r=r, taus=taus, seed=10
    )
    model = OursFull()
    model.fit(mats, {}, extra={"run_dir": run_dir})
    assert model.meta["transform"] == "posterior_shrink_tau_gated"
    assert all(lam < 1e-6 for lam in model.lambdas)
    out = model.transform(mats[0])
    np.testing.assert_allclose(out, mats[0], atol=1e-8)


def test_tau_zero_full_procrustes_qt_c_q(tmp_path: Path):
    S, R, K, r = 2, 6, 6, 3
    mats = _random_connectomes(S, R, seed=11)
    taus = [np.zeros(R) for _ in range(S)]
    run_dir = _write_posterior_artifacts(
        tmp_path / "run_lo", S=S, R=R, K=K, r=r, taus=taus, seed=12
    )
    model = OursFull()
    model.fit(mats, {}, extra={"run_dir": run_dir})
    assert model.meta["transform"] == "posterior_shrink_tau_gated"
    assert all(lam == pytest.approx(1.0) for lam in model.lambdas)
    Q = model._orths[0]
    expected = _sym(Q.T @ mats[0] @ Q)
    out = model.transform(mats[0])
    np.testing.assert_allclose(out, expected, atol=1e-8)
    assert not np.allclose(out, mats[0])


def test_adaptive_lambda_varies_with_subject_tau(tmp_path: Path):
    """Low-τ subjects align strongly; high-τ subjects stay near identity (adaptive strength)."""
    S, R, K, r = 2, 5, 5, 3
    mats = _random_connectomes(S, R, seed=13)
    taus = [np.full(R, 0.001), np.full(R, 5.0)]
    run_dir = _write_posterior_artifacts(
        tmp_path / "run_mix", S=S, R=R, K=K, r=r, taus=taus, seed=14
    )
    model = OursFull()
    model.fit(mats, {}, extra={"run_dir": run_dir})
    lam0, lam1 = model.lambdas
    assert lam0 > 0.9
    assert lam1 < 0.01
    out0 = model.transform(mats[0])
    out1 = model.transform(mats[1])
    Q0, Q1 = model._orths[0], model._orths[1]
    # subject 0 near full Procrustes
    np.testing.assert_allclose(out0, _sym(Q0.T @ mats[0] @ Q0), atol=0.05)
    # subject 1 near identity
    np.testing.assert_allclose(out1, mats[1], atol=0.05)


def test_pairwise_gamma_shape_and_symmetry_structure(tmp_path: Path):
    S, R, K, r = 3, 7, 7, 3
    mats = _random_connectomes(S, R, seed=15)
    run_dir = _write_posterior_artifacts(tmp_path / "run_g", S=S, R=R, K=K, r=r, seed=16)
    model = OursFull()
    model.fit(mats, {}, extra={"run_dir": run_dir})
    G = model.pairwise_gamma(0, 1)
    assert isinstance(G, np.ndarray)
    assert G.shape == (R, R)
    assert np.isfinite(G).all()
    # Γ_ij should be nonnegative mass-like transfer
    assert (G >= -1e-12).all()
    with pytest.raises(IndexError):
        model.pairwise_gamma(0, 99)


def test_k_neq_r_posterior_uses_rectangular_coupling(tmp_path: Path, monkeypatch):
    import ot

    S, R, K, r = 2, 10, 4, 2
    mats = _random_connectomes(S, R, seed=17)
    taus = [np.full(R, 0.02) for _ in range(S)]
    run_dir = _write_posterior_artifacts(
        tmp_path / "run_kr", S=S, R=R, K=K, r=r, taus=taus, seed=18
    )
    boom = _ExplodingEMD()
    monkeypatch.setattr(ot, "emd", boom)
    model = OursFull()
    model.fit(mats, {}, extra={"run_dir": run_dir})
    assert model.meta["transform"] == "posterior_shrink_tau_gated"
    assert boom.calls == 0
    out = model.transform(mats[0])
    assert out.shape == (R, R)
    assert np.isfinite(out).all()
    G = model.pairwise_gamma(0, 1)
    assert G.shape == (R, R)


def test_point_procrustes_meta_and_ablated_drops_hierarchy(tmp_path: Path):
    S, R, K, r = 2, 6, 6, 3
    mats = _random_connectomes(S, R, seed=19)
    taus = [np.full(R, 0.001) for _ in range(S)]  # would give λ≈1 if posterior used
    run_dir = _write_posterior_artifacts(
        tmp_path / "run_ab", S=S, R=R, K=K, r=r, taus=taus, seed=20
    )
    abl = OursAblated()
    abl.fit(mats, {}, extra={"run_dir": run_dir})
    assert abl.meta["transform"] == "point_procrustes_C_pop"
    assert abl.meta["transform_mode"] == "point_procrustes"
    assert abl.meta["gauge_features"] is False
    assert abl.meta["reference_geometry"] == "C_pop"
    # Even with low τ posteriors present, ablation uses full point Procrustes (λ=1)
    assert all(lam == 1.0 for lam in abl.lambdas)
    out = abl.transform(mats[0])
    Q = abl._orths[0]
    np.testing.assert_allclose(out, _sym(Q.T @ mats[0] @ Q), atol=1e-8)

    full = OursFull()
    full.fit(mats, {}, extra={"run_dir": run_dir})
    assert full.meta["transform"] == "posterior_shrink_tau_gated"
    assert full.meta != abl.meta
    out_full = full.transform(mats[0])
    # Hierarchy changes the map vs ablation when τ is low (λ≈1) but Q differs (C_bar vs C_pop)
    # At minimum, meta path differs; transforms need not be identical references.
    assert full.meta["reference_geometry"] != abl.meta["reference_geometry"]


def test_missing_posterior_falls_back_to_region_emd_procrustes(tmp_path: Path):
    S, R, K, r = 2, 6, 4, 2
    mats = _random_connectomes(S, R, seed=21)
    art = tmp_path / "run_tonly" / "artifacts"
    art.mkdir(parents=True)
    rng = np.random.default_rng(22)
    B = rng.normal(size=(K, r))
    np.savez(
        art / "template.npz",
        B=B,
        nu=np.full(K, 1.0 / K),
        eps=np.ones(S),
        subject_ids=np.array(["001", "002"]),
    )
    model = OursFull()
    model.fit(mats, {}, extra={"run_dir": tmp_path / "run_tonly"})
    assert model.meta["transform"] == "region_emd_procrustes"
    assert model.meta["reference_geometry"] == "C_pop"
    out = model.transform(mats[0])
    assert out.shape == (R, R)
    assert np.isfinite(out).all()
    assert not np.allclose(out, mats[0])


def test_shrinkage_import_path_used_when_available(tmp_path: Path):
    try:
        from trajot.eval.uncertainty import shrinkage_transform as task1
    except Exception:
        pytest.skip("Task-1 shrinkage_transform not importable")
    C = _sym(np.random.default_rng(23).normal(size=(5, 5)))
    Q = _orthogonal(5, seed=24)
    from trajot.baselines.ours import _apply_shrinkage

    np.testing.assert_allclose(_apply_shrinkage(C, Q, 0.3), task1(C, Q, 0.3))
    np.testing.assert_allclose(_apply_shrinkage(C, Q, 0.3), _local_shrinkage(C, Q, 0.3))
