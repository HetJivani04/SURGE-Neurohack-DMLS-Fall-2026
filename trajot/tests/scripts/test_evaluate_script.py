from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

from trajot.eval.metrics import METHOD_KEYS, TOP_KEYS

ROOT = Path(__file__).resolve().parents[2]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evaluate_synthetic_runs_noalign_and_fake(tmp_path):
    shutil.copytree(ROOT / "configs", tmp_path / "configs", ignore=shutil.ignore_patterns("paths.yaml"))
    (tmp_path / "configs" / "paths.yaml").write_text(
        f"data_root: {tmp_path / 'data'}\ncontract_version: '1'\n"
    )

    out_path = tmp_path / "metrics.json"
    module = _load_module(ROOT / "scripts" / "evaluate.py", "evaluate_copy")
    rc = module.main(
        [
            "--config",
            str(tmp_path / "configs" / "experiments" / "00_noalign.yaml"),
            "--methods",
            "noalign,fake",
            "--synthetic",
            "--output",
            str(out_path),
            "--n-jobs",
            "1",
            "--seed",
            "11",
        ]
    )
    assert rc == 0

    payload = json.loads(out_path.read_text())
    assert set(payload) == TOP_KEYS
    assert payload["notes"] == "synthetic"
    assert set(payload["methods"]) == {"noalign", "fake"}
    assert payload["n_subjects"] > 0

    for stats in payload["methods"].values():
        assert set(stats) == METHOD_KEYS
        assert 0.0 <= stats["ident_accuracy"] <= 1.0
        assert 0.0 <= stats["perm_p"] <= 1.0
        assert isinstance(stats["nonidentifiable_pairs"], int)
        assert len(stats["per_pair_flags"]) == payload["n_subjects"]
