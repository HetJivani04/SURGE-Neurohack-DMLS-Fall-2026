from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from trajot.config import Config
from trajot.inference.synthetic import make_synthetic_npz, read_truth
from trajot.inference.train import (
    TemplateParams, farthest_point_sampling, init_template, load_train_data, posterior_samples, prepare_subject,
    save_artifacts, train,
)
from trajot.io import contract
from trajot.runlog.parallel import pick_device

ROOT = Path(__file__).resolve().parents[2]
CPU = pick_device(prefer_mps=False)


def make_cfg(V: int, epochs: int, gauge: bool = False, band: float = 0.0, r: int = 8, p: int = 48, blocks: int = 2,
             entropy_weight: float = 1.0) -> Config:
    return Config({"run": {"seed": 0}, "model": {
        "K": V, "d": 8, "r": r, "m_draws": 4, "batch_subjects": 8, "sinkhorn": {"L": 30, "eps": [0.1, 0.01]},
        "beta": {"warmup_frac": 0.3, "synthetic": 50.0}, "sigma_f2": 1.0, "gauge_features": gauge,
        "entropy": {"weight": entropy_weight},
        "prior": {"sigma_B": 1.0, "sigma_F": 1.0, "F0": 0.0, "a_eps": 3.0, "b_eps": 0.4, "B_max_row_norm": 1.3},
        "encoder": {"p": p, "n_blocks": blocks, "heads": 4, "m_eigvecs": 4, "lambda_init": 1.0},
        "band": {"seconds": [10.0, 100.0], "bandwidth": 0.05, "weight": band}, "train": {"epochs": epochs, "lr": 0.003}}})


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    return make_synthetic_npz(tmp_path_factory.mktemp("planted"), n_subjects=8, V=32, R=10, T=64, d=8, seed=1)


def correspondence(result, truth) -> tuple[float, float]:
    """Pairwise vertex correspondence of derived couplings Gamma_AB = pi_A diag(nu)^-1 pi_B^T against the planted one."""
    perm, ids = truth["perm"], [str(s) for s in truth["subject_ids"]]
    posterior_mean = [pi.mean(0) for pi in result.pi_samples]
    nu = result.params.nu
    accuracy, mass = [], []
    for a, sid_a in enumerate(result.subject_ids):
        for b, sid_b in enumerate(result.subject_ids):
            if a == b:
                continue
            gamma = posterior_mean[a] @ np.diag(1.0 / nu) @ posterior_mean[b].T
            target = np.argsort(perm[ids.index(sid_b)])[perm[ids.index(sid_a)]]  # vertex of B that is A's node
            accuracy.append((gamma.argmax(1) == target).mean())
            mass.append(gamma[np.arange(len(target)), target].sum() / gamma.sum())
    return float(np.mean(accuracy)), float(np.mean(mass))


# ---- template initialization ---------------------------------------------------------------------------
def test_init_template_shapes_dtypes_and_determinism() -> None:
    rng = np.random.default_rng(0)
    C = rng.normal(size=(6, 20, 20))
    C = C + C.transpose(0, 2, 1)
    t = init_template(C, K=12, r=5, seed=3, n_features=9)
    assert isinstance(t, TemplateParams)
    assert t.B.shape == (12, 5) and t.B.dtype == np.float32 and t.F_bar.shape == (12, 9) and t.F_bar.dtype == np.float32
    assert t.nu.shape == (12,) and t.nu.dtype == np.float64 and np.allclose(t.nu, 1 / 12)
    assert t.eps.shape == (6,) and t.eps.dtype == np.float64 and (t.eps > 0).all()
    again = init_template(C, K=12, r=5, seed=3, n_features=9)
    assert np.array_equal(t.B, again.B) and np.array_equal(t.eps, again.eps)
    assert not np.array_equal(t.B, init_template(C, K=12, r=5, seed=4, n_features=9).B)

    bigger = init_template(C, K=50, r=5, seed=0)  # K > R: rows are reused
    assert bigger.B.shape == (50, 5) and bigger.F_bar.shape == (50, 5)


def test_init_template_follows_the_spectral_embedding_of_the_group_average() -> None:
    rng = np.random.default_rng(1)
    loadings = rng.normal(size=(30, 3))
    C = np.tile(loadings @ loadings.T, (4, 1, 1)) + 0.01 * rng.normal(size=(4, 30, 30))
    t = init_template(C, K=30, r=3, seed=0)
    gram = t.B @ t.B.T  # a permutation of the embedding rows (plus noise): same Gram spectrum
    assert np.allclose(np.sort(np.linalg.eigvalsh(gram))[-3:], np.sort(np.linalg.eigvalsh(np.mean(C, 0)))[-3:], rtol=0.15)


def test_farthest_point_sampling_picks_spread_out_distinct_points() -> None:
    points = np.random.default_rng(0).normal(size=(100, 3))
    chosen = farthest_point_sampling(points, 10)
    assert len(set(chosen.tolist())) == 10 and chosen[0] == 0
    random = np.random.default_rng(1).choice(100, 10, replace=False)

    def spread(idx):
        d = np.linalg.norm(points[idx][:, None] - points[idx][None], axis=-1)
        return d[np.triu_indices(10, 1)].min()

    assert spread(chosen) > spread(random)
    with pytest.raises(ValueError):
        farthest_point_sampling(points, 101)


# ---- data ------------------------------------------------------------------------------------------------
def test_prepare_subject_builds_the_model_inputs_from_a_contract_file(dataset: Path) -> None:
    cfg = make_cfg(32, 1, gauge=True)
    d = contract.read_subject_run(contract.subject_run_path(dataset, "001", "1"))
    d["timeseries"][3] = 0.0  # a vertex outside the field of view
    nodes = d["coords"][:32].astype(np.float64)
    sub = prepare_subject(d, cfg, nodes, 100.0)

    V = 31
    assert sub.valid.sum() == V and not sub.valid[3]
    assert sub.A.shape == (V, 8) and sub.mu.shape == (V,) and np.isclose(sub.mu.sum(), 1.0)
    assert sub.Y.shape == (V, 8 + 2) and sub.v.shape == (V, 8) and sub.M0.shape == (V, 32)
    assert sub.u.shape == (V, 8 + 2 + 4 + 2) and sub.u.dtype == np.float32
    assert (sub.M0 >= 0).all() and sub.M0.max() < 5
    assert sub.anchor == "euclidean"

    X = d["timeseries"][sub.valid].astype(np.float64)
    corr = np.corrcoef(X)
    full = prepare_subject(d, make_cfg(32, 1, r=60), nodes, 100.0)  # rank up to T - 1 = 63 > 60 loses little
    assert np.linalg.norm(full.A @ full.A.T - corr) / np.linalg.norm(corr) < 0.25
    assert np.allclose(np.diag(sub.A @ sub.A.T), (sub.A**2).sum(1)) and (sub.A**2).sum(1).max() <= 1.0 + 1e-9
    assert prepare_subject(d, make_cfg(32, 1, gauge=False), nodes, 100.0).v is None

    node_dist = np.linspace(0.1, 0.9, 32 * 32).reshape(32, 32)
    geo = prepare_subject(d, cfg, nodes, 100.0, node_dist=node_dist)
    assert geo.anchor == "geodesic"
    assert np.allclose(geo.M0, node_dist[geo.valid])


def test_load_train_data_trains_on_run_1_and_holds_run_2_out(dataset: Path) -> None:
    data = load_train_data(dataset, make_cfg(32, 1))
    assert len(data.train) == len(data.holdout) == 8 and data.beta_target == 50.0
    assert data.connectomes.shape == (8, 10, 10) and data.node_coords.shape == (32, 3)
    assert data.anchor == "euclidean"  # synthetic contract files have no faces
    assert all(sub.anchor == "euclidean" for sub in data.train)
    run1 = contract.read_subject_run(contract.subject_run_path(dataset, "001", "1"))["embedding"].astype(np.float64)
    run2 = contract.read_subject_run(contract.subject_run_path(dataset, "001", "2"))["embedding"].astype(np.float64)
    assert np.allclose(data.train[0].Y[:, :8], run1) and np.allclose(data.holdout[0].Y[:, :8], run2)
    assert not np.allclose(run1, run2)

    subset = load_train_data(dataset, make_cfg(32, 1), subjects=["002", "003"], beta_target=7.5)
    assert [s.subject_id for s in subset.train] == ["002", "003"] and subset.beta_target == 7.5
    with pytest.raises(ValueError, match="no subjects"):
        load_train_data(dataset, make_cfg(32, 1), subjects=["999"])


def test_load_train_data_prefers_geodesic_anchor_when_faces_are_available(dataset: Path) -> None:
    V = 32
    faces = np.array([[i, i + 1, i + 2] for i in range(V - 2)], dtype=np.int64)
    deriv = dataset / "derivatives" / "trajot"
    np.savez(deriv / "template_geometry.npz", faces=faces)
    data = load_train_data(dataset, make_cfg(32, 1))
    assert data.anchor == "geodesic"
    assert all(sub.anchor == "geodesic" for sub in data.train)
    assert (data.train[0].M0 >= 0).all() and data.train[0].M0.max() <= 1.0 + 1e-9
    (deriv / "template_geometry.npz").unlink()


# ---- the fit ---------------------------------------------------------------------------------------------
def test_a_smoke_fit_completes_on_cpu_and_returns_the_documented_result(dataset: Path, capsys) -> None:
    data = load_train_data(dataset, make_cfg(32, 3))
    data.holdout = None  # training must never touch run 2
    result = train(make_cfg(32, 3), data, CPU)

    S, V, K, M = 8, 32, 32, 4
    assert result.subject_ids == [f"{s:03d}" for s in range(1, 9)] and len(result.loss_trace) == 3
    assert result.params.B.shape == (K, 8) and result.params.B.dtype == np.float32
    assert result.params.F_bar.shape == (K, 10) and result.params.eps.shape == (S,) and (result.params.eps > 0).all()
    assert np.linalg.norm(result.params.B, axis=1).max() <= 1.3 + 1e-4  # the scale constraint
    assert len(result.pi_samples) == len(result.tau_phi) == S
    assert result.anchor == "euclidean" and result.entropy_weight == 1.0
    for pi, tau, sub in zip(result.pi_samples, result.tau_phi, data.train):
        assert pi.shape == (M, V, K) and pi.dtype == np.float64 and tau.shape == (V,) and (tau > 0).all()
        assert np.allclose(pi.sum((1, 2)), 1.0, atol=1e-3)  # probability couplings
        assert np.abs(pi.sum(1) - 1 / K).max() < 1e-9  # the last Sinkhorn half-step makes the column marginals exact
        assert (np.abs(pi.sum(2) - sub.mu) / sub.mu).mean() < 0.15  # rows: close on average at eps = 0.01, L = 30
    assert all(np.isfinite(e["total"]) and np.isfinite(e["entropy"]) for e in result.loss_trace)
    assert any(e["entropy"] != 0.0 for e in result.loss_trace)  # entropy is wired, not hardcoded zero
    assert result.loss_trace[-1]["beta"] == pytest.approx(50.0) and result.loss_trace[-1]["eps"] > 0.01
    assert set(result.loss_trace[0]) >= {"gw", "feature", "prior", "entropy", "log_prior", "total", "beta", "eps", "tau_mean"}

    out = capsys.readouterr().out
    assert "entropy WIRED" in out and "shannon" in out.lower() and out.count("epoch ") == 3
    assert result.entropy_estimator == "shannon_pi+hutchinson_slq"


def test_entropy_is_nonzero_and_finite_on_a_short_synthetic_fit(dataset: Path) -> None:
    cfg = make_cfg(32, 2)
    result = train(cfg, load_train_data(dataset, cfg, subjects=["001", "002", "003"]), CPU)
    entropies = [e["entropy"] for e in result.loss_trace]
    assert all(np.isfinite(entropies))
    assert any(abs(e) > 0.0 for e in entropies)



@pytest.mark.parametrize("gauge", [False, True], ids=["ablation-no-gauge", "gauge-features"])
def test_the_posterior_recovers_a_planted_permutation(dataset: Path, gauge: bool) -> None:
    """Acceptance: identification from posterior means beats chance by a wide margin, and the derived coupling puts
    its mass on the planted correspondence."""
    cfg = make_cfg(32, 80, gauge=gauge)
    data = load_train_data(dataset, cfg)
    result = train(cfg, data, CPU)
    accuracy, mass = correspondence(result, read_truth(dataset))
    chance = 1 / 32
    assert accuracy > 0.6 and accuracy > 15 * chance
    assert mass > 0.6 and mass > 15 * chance


def test_the_trained_posterior_meets_the_1e_minus_4_marginal_bound_given_enough_iterations(dataset: Path) -> None:
    """At eps = 0.01, 30 Sinkhorn iterations leave the row marginals only approximately satisfied (see the smoke
    fit); the 1e-4 bound is reached with more iterations from the same encoder and noise."""
    cfg = make_cfg(32, 3)
    data = load_train_data(dataset, cfg)
    result = train(cfg, data, CPU)
    for sub in data.train[:3]:
        pi, _ = posterior_samples(result.encoder, sub, result.params.nu, 0.01, 2, 1000, CPU, np.random.default_rng(0))
        assert np.abs(pi.sum(2) - sub.mu).max() < 1e-4 and np.abs(pi.sum(1) - 1 / 32).max() < 1e-4


def test_the_trained_encoder_scores_the_held_out_run(dataset: Path) -> None:
    cfg = make_cfg(32, 2)
    data = load_train_data(dataset, cfg)
    result = train(cfg, data, CPU)
    pi, tau = posterior_samples(result.encoder, data.holdout[0], result.params.nu, 0.01, 4, 30, CPU)
    assert pi.shape == (4, 32, 32) and tau.shape == (32,) and np.allclose(pi.sum((1, 2)), 1.0, atol=1e-3)


def test_the_band_prior_enters_the_prior_energy_only_when_weighted(dataset: Path) -> None:
    data = load_train_data(dataset, make_cfg(32, 1))
    off = train(make_cfg(32, 1, band=0.0), data, CPU).loss_trace[-1]["prior"]
    on = train(make_cfg(32, 1, band=1.0), data, CPU).loss_trace[-1]["prior"]
    assert on > off


def test_save_artifacts_writes_the_posterior_tau_and_template(dataset: Path, tmp_path: Path) -> None:
    cfg = make_cfg(32, 1)
    result = train(cfg, load_train_data(dataset, cfg, subjects=["001", "002"]), CPU)
    save_artifacts(tmp_path, result)
    out = tmp_path / "artifacts"
    with np.load(out / "posterior_samples.npz") as samples:
        assert sorted(samples.files) == ["sub-001", "sub-002"] and samples["sub-001"].shape == (4, 32, 32)
    with np.load(out / "tau_phi.npz") as tau:
        assert tau["sub-001"].shape == (32,)
    with np.load(out / "template.npz") as template:
        assert template["B"].shape == (32, 8) and template["eps"].shape == (2,) and list(template["subject_ids"]) == ["001", "002"]
        assert str(template["anchor"]) == "euclidean"
        assert float(template["entropy_weight"]) == pytest.approx(1.0)
        assert str(template["entropy_estimator"]) == "shannon_pi+hutchinson_slq"
    assert json.loads((out / "loss_trace.json").read_text())[0]["epoch"] == 1


# ---- conventions the issue asks to confirm by grep -------------------------------------------------------
def test_no_reinforce_score_function_or_detach_anywhere_in_model_and_inference() -> None:
    files = [*(ROOT / "src" / "trajot" / "model").glob("*.py"), *(ROOT / "src" / "trajot" / "inference").glob("*.py"),
             ROOT / "scripts" / "fit.py"]
    pattern = re.compile(r"REINFORCE|score[- _]function|\.detach\(\)", re.IGNORECASE)
    assert [f.name for f in files if pattern.search(f.read_text())] == []
    assert len(files) > 10


def test_gauge_and_band_docstrings_do_not_claim_temporal_dynamics() -> None:
    from trajot.geometry import diffusion
    from trajot.model import prior as prior_mod
    from trajot.inference import train as train_mod

    texts = [diffusion.gauge_velocity.__doc__ or "", diffusion.gauge_features.__doc__ or "",
             prior_mod.band_prior.__doc__ or "", train_mod.__doc__ or ""]
    banned = ("temporal velocity", "kinematic", "captures dynamics", "temporal/kinematic")
    for text in texts:
        lowered = text.lower()
        for phrase in banned:
            assert phrase not in lowered
    assert "spatial" in (diffusion.gauge_velocity.__doc__ or "").lower()
    assert "band prior" in (prior_mod.band_prior.__doc__ or "").lower()

