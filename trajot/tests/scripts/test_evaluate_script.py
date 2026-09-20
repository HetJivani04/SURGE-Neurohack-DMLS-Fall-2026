from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Plan Interfaces / D7 METHOD_KEYS. Preferred when Agent C has landed null_max.
PLAN_METHOD_KEYS = {
    "ident_accuracy",
    "ident_ci",
    "perm_p",
    "null_max",
    "alignment_gain",
    "nonidentifiable_pairs",
    "per_pair_uncertainty",
    "per_pair_flags",
}


def _expected_method_keys() -> set[str]:
    try:
        from trajot.eval.metrics import METHOD_KEYS

        if "null_max" in set(METHOD_KEYS):
            return set(METHOD_KEYS)
    except Exception:
        pass
    return set(PLAN_METHOD_KEYS)


def _expected_top_keys() -> set[str]:
    try:
        from trajot.eval.metrics import TOP_KEYS

        return set(TOP_KEYS)
    except Exception:
        return {
            "experiment",
            "run_id",
            "n_subjects",
            "n_pairs",
            "pairs_seed",
            "permutations_B",
            "methods",
            "beta",
            "notes",
        }


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
    assert set(payload) == _expected_top_keys()
    assert payload["notes"] == "synthetic"
    assert set(payload["methods"]) == {"noalign", "fake"}
    assert payload["n_subjects"] > 0

    keys = _expected_method_keys()
    for name, stats in payload["methods"].items():
        assert set(stats) == keys
        assert 0.0 <= stats["ident_accuracy"] <= 1.0
        if stats["perm_p"] is not None:
            assert 0.0 <= stats["perm_p"] <= 1.0
        assert "null_max" in stats
        assert isinstance(stats["nonidentifiable_pairs"], int)
        flags = stats["per_pair_flags"]
        assert flags is None or isinstance(flags, list)
        if flags is not None:
            assert len(flags) > 0
        # Baselines emit null for the two model-only columns (W4 validate_metrics contract).
        assert stats["per_pair_flags"] is None, name
        assert stats["per_pair_uncertainty"] is None, name


def test_evaluate_method_helper_schema_valid(tmp_path):
    import numpy as np
    from trajot.config import Config

    shutil.copytree(ROOT / "configs", tmp_path / "configs", ignore=shutil.ignore_patterns("paths.yaml"))
    (tmp_path / "configs" / "paths.yaml").write_text(
        f"data_root: {tmp_path / 'data'}\ncontract_version: '1'\n"
    )
    module = _load_module(ROOT / "scripts" / "evaluate.py", "evaluate_helper")

    raw = {
        "experiment": "00_noalign",
        "data": {"root": str(tmp_path / "data"), "contract_version": "1"},
        "run": {"seed": 3, "n_jobs": 1, "debug": True, "synthetic": True},
        "eval": {"pairs": {"n": 6, "seed": 7}, "permutations": {"B": 20}},
    }
    cfg = Config(raw)
    S, R = 6, 12
    rng = np.random.default_rng(0)
    run1 = np.stack([_sym(rng, R) for _ in range(S)])
    run2 = run1 + 0.05 * np.stack([_sym(rng, R) for _ in range(S)])
    subjects = [f"{i:03d}" for i in range(S)]

    stats = module.evaluate_method("noalign", None, run1, run2, cfg, subjects)
    stats = module._strip_meta(stats)
    assert set(stats) == _expected_method_keys()
    assert 0.0 <= stats["ident_accuracy"] <= 1.0
    assert isinstance(stats["null_max"], (float, type(None)))
    assert stats["per_pair_flags"] is None
    assert stats["per_pair_uncertainty"] is None
    assert module._canonical_method_key("full") == "ours_full"
    assert module._canonical_method_key("ablated") == "ours_ablated"
    assert module._canonical_method_key("ours_full") == "ours_full"


def test_evaluate_all_writes_canonical_ours_keys_not_aliases(tmp_path):
    import numpy as np
    from trajot.config import Config

    shutil.copytree(ROOT / "configs", tmp_path / "configs", ignore=shutil.ignore_patterns("paths.yaml"))
    (tmp_path / "configs" / "paths.yaml").write_text(
        f"data_root: {tmp_path / 'data'}\ncontract_version: '1'\n"
    )
    module = _load_module(ROOT / "scripts" / "evaluate.py", "evaluate_keys")
    raw = {
        "experiment": "10_ours_full",
        "data": {"root": str(tmp_path / "data"), "contract_version": "1"},
        "run": {"seed": 1, "n_jobs": 1, "debug": True, "synthetic": True},
        "eval": {"pairs": {"n": 4, "seed": 2}, "permutations": {"B": 8}},
    }
    cfg = Config(raw)
    S, R = 5, 8
    rng = np.random.default_rng(0)
    run1 = np.stack([_sym(rng, R) for _ in range(S)])
    run2 = run1 + 0.02 * np.stack([_sym(rng, R) for _ in range(S)])
    subjects = [f"{i:03d}" for i in range(S)]
    out = module._evaluate_all(["full", "ours_full", "noalign"], run1, run2, cfg, subjects)
    assert "ours_full" in out and "full" not in out
    assert "noalign" in out
    assert out["noalign"]["per_pair_flags"] is None
    assert out["noalign"]["per_pair_uncertainty"] is None


def _sym(rng, R: int) -> "np.ndarray":
    import numpy as np

    A = rng.normal(size=(R, R))
    C = 0.5 * (A + A.T)
    np.fill_diagonal(C, 0.0)
    return C
