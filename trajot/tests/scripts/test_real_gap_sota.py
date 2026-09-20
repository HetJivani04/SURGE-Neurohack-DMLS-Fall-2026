from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_real_gap_sota.py"


def _load_mod():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("run_real_gap_sota", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_real_gap_sota"] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_mod()


def test_score_transforms_primary_same_map_metrics():
    rng = np.random.default_rng(0)
    S, R = 6, 8
    run1 = rng.normal(size=(S, R, R))
    run1 = 0.5 * (run1 + run1.transpose(0, 2, 1))
    run2 = run1 + 0.05 * rng.normal(size=run1.shape)
    run2 = 0.5 * (run2 + run2.transpose(0, 2, 1))
    subjects = [f"{i:03d}" for i in range(S)]
    pairs = [(i, i) for i in range(S)] + [(0, 1)]
    # identity map on both runs
    row = mod.score_transforms("noalign", run1.copy(), run2.copy(), run1, run2, subjects, pairs)
    assert row["status"] == "ok"
    assert row["heldout_protocol_wrong"] is True
    assert np.isfinite(row["reliability_raw"])
    assert row["reliability_after"] == pytest.approx(row["reliability_raw"])
    assert row["gain_after"] == pytest.approx(0.0)
    assert row["ident_after"] == pytest.approx(row["ident_raw"])
    assert row["collapsed"] is False
    assert row["tau_phi_mean"] is None


def test_score_transforms_flags_identity_collapse():
    rng = np.random.default_rng(1)
    S, R = 8, 6
    run1 = rng.normal(size=(S, R, R))
    run1 = 0.5 * (run1 + run1.transpose(0, 2, 1))
    run2 = run1 + 0.01 * rng.normal(size=run1.shape)
    # collapse: map every subject/run to the same template
    template = run1.mean(axis=0)
    aligned1 = np.stack([template.copy() for _ in range(S)])
    aligned2 = np.stack([template.copy() for _ in range(S)])
    subjects = [f"{i:03d}" for i in range(S)]
    pairs = [(i, i) for i in range(S)]
    row = mod.score_transforms("conn_srm", aligned1, aligned2, run1, run2, subjects, pairs)
    assert row["ident_collapsed"] is True
    assert row["collapsed"] is True


def test_verdict_primary_sota_bars():
    rows = [
        {
            "method": "noalign",
            "status": "ok",
            "reliability_raw": 0.64,
            "reliability_after": 0.64,
            "reliability_delta": 0.0,
            "gain_after": 0.0,
            "ident_after": 0.96,
            "ident_raw": 0.96,
            "collapsed": False,
            "tau_phi_mean": None,
        },
        {
            "method": "brainsync",
            "status": "ok",
            "reliability_raw": 0.64,
            "reliability_after": 0.642,
            "reliability_delta": 0.002,
            "gain_after": 0.0007,
            "ident_after": 0.96,
            "ident_raw": 0.96,
            "collapsed": False,
            "tau_phi_mean": None,
        },
        {
            "method": "fugw",
            "status": "ok",
            "reliability_raw": 0.64,
            "reliability_after": 0.62,
            "reliability_delta": -0.02,
            "gain_after": -0.005,
            "ident_after": 0.92,
            "ident_raw": 0.96,
            "collapsed": True,
            "tau_phi_mean": None,
        },
        {
            "method": "conn_srm",
            "status": "ok",
            "reliability_raw": 0.64,
            "reliability_after": 0.85,
            "reliability_delta": 0.21,
            "gain_after": 0.39,
            "ident_after": 0.08,
            "ident_raw": 0.96,
            "collapsed": True,
            "tau_phi_mean": None,
        },
        {
            "method": "ours_full_point_procrustes",
            "status": "ok",
            "reliability_raw": 0.64,
            "reliability_after": 0.644,
            "reliability_delta": 0.004,
            "gain_after": 0.018,
            "ident_after": 0.857,
            "ident_raw": 0.96,
            "collapsed": False,
            "tau_phi_mean": 0.009,
            "lambda_mean": 1.0,
        },
        {
            "method": "ours_full_posterior_shrink_entropy",
            "status": "ok",
            "reliability_raw": 0.64,
            "reliability_after": 0.685,
            "reliability_delta": 0.045,
            "gain_after": 0.02,
            "ident_after": 0.98,
            "ident_raw": 0.96,
            "collapsed": False,
            "tau_phi_mean": 0.009,
            "lambda_mean": 0.55,
            "lambda_min": 0.1,
            "lambda_max": 0.95,
            "lambda_source": "row_entropy",
            "n_undetermined_subjects": 12,
        },
    ]
    group = {"n_subjects": 12, "n_eff": 7.4, "n_eff_min": 2.5, "ci_ratio": 1.28}
    v = mod._verdict(rows, group)
    assert v["metric_protocol"] == "same_map_both_runs_v1"
    assert v["heldout_protocol_wrong_not_headlined"] is True
    assert v["beat_noalign_on_gain_after"] is True
    assert v["beat_brainsync_on_gain_after"] is True
    assert v["beat_fugw_on_gain_after"] is True  # fugw collapsed
    assert v["sota_beat_brainsync_fugw_noalign"] is True
    assert v["group_neff_lt_S"] is True
    assert v["best_ours_row"] == "ours_full_posterior_shrink_entropy"
    md = mod.render_markdown(
        {
            "data_root": "/tmp/x",
            "n_subjects": 49,
            "beta": 29.189,
            "n_pairs": 500,
            "pairs_seed": 2026,
            "ours_artifacts": "/tmp/a",
            "transform_paths": {},
            "methods": rows,
            "group": group,
            "verdict": v,
        }
    )
    assert "corrected protocol" in md or "PRIMARY" in md
    assert "gain_after" in md
    assert "protocol-wrong" in md
