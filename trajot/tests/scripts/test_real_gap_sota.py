from __future__ import annotations

import json
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


def test_score_transforms_shapes_and_heldout():
    rng = np.random.default_rng(0)
    S, R = 6, 8
    run1 = rng.normal(size=(S, R, R))
    run1 = 0.5 * (run1 + run1.transpose(0, 2, 1))
    np.einsum("sii->i", run1)[:] = 0
    run2 = run1 + 0.01 * rng.normal(size=run1.shape)
    run2 = 0.5 * (run2 + run2.transpose(0, 2, 1))
    aligned1 = run1.copy()
    aligned2 = run2.copy()
    subjects = [f"{i:03d}" for i in range(S)]
    pairs = [(i, i) for i in range(S)] + [(0, 1), (2, 3)]
    row = mod.score_transforms("noalign", aligned1, aligned2, run1, run2, subjects, pairs)
    assert row["method"] == "noalign"
    assert row["status"] == "ok"
    assert np.isfinite(row["heldout_score_module"])
    assert np.isfinite(row["heldout_score_aligned"])
    assert np.isfinite(row["scanrescan_corr_raw"])
    assert np.isfinite(row["alignment_gain"])
    assert row["tau_phi_mean"] is None
    # identity transform: aligned residual vs run2 equals raw scan-rescan residual
    assert row["heldout_score_module"] == pytest.approx(row["heldout_score_aligned"])


def test_verdict_notes_include_honest_fail():
    rows = [
        {
            "method": "noalign",
            "status": "ok",
            "heldout_score_module": -1.0,
            "scanrescan_corr_after": 0.9,
            "ident_accuracy": 0.96,
            "alignment_gain": 0.0,
            "tau_phi_mean": None,
        },
        {
            "method": "ours_full_posterior_shrink",
            "status": "ok",
            "heldout_score_module": -2.0,
            "scanrescan_corr_after": 0.5,
            "ident_accuracy": 0.8,
            "alignment_gain": 0.01,
            "tau_phi_mean": 0.08,
        },
    ]
    v = mod._verdict(rows, {"n_subjects": 10, "n_eff": 7.2, "ci_ratio": 1.2})
    assert v["sota_heldout_beats_noalign"] is False
    assert any("HONEST" in n for n in v["notes"])
    assert v["group_neff_lt_S"] is True
    assert "point estimates" in v["literature_gap"]


def test_render_markdown_has_table():
    payload = {
        "table": "REAL_n49_gap_sota",
        "data_root": "/tmp/x",
        "n_subjects": 49,
        "beta": 29.189,
        "n_pairs": 500,
        "pairs_seed": 2026,
        "ours_artifacts": "/tmp/a",
        "transform_paths": {"noalign": "identity"},
        "methods": [
            {
                "method": "noalign",
                "heldout_score_module": -1.0,
                "heldout_error_module": 1.0,
                "alignment_gain": 0.0,
                "ident_accuracy": 0.96,
                "scanrescan_corr_raw": 0.4,
                "scanrescan_corr_after": 0.4,
                "tau_phi_mean": None,
            }
        ],
        "group": {"error": "none yet"},
        "verdict": {
            "sota_heldout_beats_noalign": False,
            "sota_heldout_beats_fugw": False,
            "ours_alignment_gain_positive": False,
            "group_neff_lt_S": False,
            "notes": ["test note"],
            "literature_gap": "point estimates only",
        },
    }
    md = mod.render_markdown(payload)
    assert "REAL N=49" in md
    assert "noalign" in md
    assert "Honest verdict" in md
