from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def compare_module():
    return _load_module(ROOT / "scripts" / "compare.py", "compare_copy")


def _write_metrics(runs: Path, run_id: str, payload: dict) -> None:
    d = runs / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")


def test_compare_writes_csv_with_both_methods_and_nulls_empty(tmp_path, compare_module):
    runs = tmp_path / "runs"
    _write_metrics(
        runs,
        "00_noalign_abc",
        {
            "experiment": "00_noalign",
            "run_id": "00_noalign_abc",
            "n_subjects": 8,
            "n_pairs": 6,
            "pairs_seed": 2026,
            "permutations_B": 16,
            "methods": {
                "noalign": {
                    "ident_accuracy": 0.75,
                    "ident_ci": [0.5, 0.95],
                    "perm_p": 0.02,
                    "null_max": 0.4,
                    "alignment_gain": 0.0,
                    "nonidentifiable_pairs": 2,
                    "per_pair_uncertainty": None,
                    "per_pair_flags": [True, True, False, False, False, False],
                },
            },
            "beta": None,
            "notes": "synthetic",
        },
    )
    _write_metrics(
        runs,
        "01_brainsync_def",
        {
            "experiment": "01_brainsync",
            "run_id": "01_brainsync_def",
            "n_subjects": 8,
            "n_pairs": 6,
            "pairs_seed": 2026,
            "permutations_B": 16,
            "methods": {
                "brainsync": {
                    "ident_accuracy": 0.9,
                    "ident_ci": [0.7, 1.0],
                    "perm_p": 0.001,
                    "null_max": None,
                    "alignment_gain": 0.15,
                    "nonidentifiable_pairs": 0,
                    "per_pair_uncertainty": 0.12,
                    "per_pair_flags": [False] * 6,
                },
            },
            "beta": 50.0,
            "notes": "",
        },
    )

    out = tmp_path / "results" / "tables" / "comparison.csv"
    rc = compare_module.main(["--runs", str(runs), "--out", str(out), "--quiet"])
    assert rc == 0
    assert out.is_file()

    text = out.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 3  # header + two methods
    header = lines[0].split(",")
    assert "method" in header
    assert "null_max" in header
    assert "per_pair_uncertainty" in header

    body = "\n".join(lines[1:])
    assert "noalign" in body
    assert "brainsync" in body
    # nulls stay empty, not 0
    assert ",," in body or body.count(",,") >= 1
    # brainsync null_max is null -> empty cell; noalign per_pair_uncertainty is null -> empty
    for line in lines[1:]:
        cells = line.split(",")
        # method col index
        mi = header.index("method")
        nm = header.index("null_max")
        ppu = header.index("per_pair_uncertainty")
        if cells[mi] == "brainsync":
            assert cells[nm] == ""
            assert cells[ppu] != ""  # filled uncertainty
        if cells[mi] == "noalign":
            assert cells[ppu] == ""  # unfilled stays empty
            assert cells[nm] != ""  # filled null_max


def test_compare_empty_runs_dir(tmp_path, compare_module, capsys):
    out = tmp_path / "comparison.csv"
    rc = compare_module.main(["--runs", str(tmp_path / "missing"), "--out", str(out)])
    assert rc == 0
    assert out.is_file()
    assert out.read_text().strip().splitlines()[0].startswith("experiment")


def test_compare_help(compare_module):
    with pytest.raises(SystemExit) as exc:
        compare_module.main(["--help"])
    assert exc.value.code == 0
