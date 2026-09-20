from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TOP_KEYS = {
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

METHOD_KEYS = {
    "ident_accuracy",
    "ident_ci",
    "perm_p",
    "alignment_gain",
    "nonidentifiable_pairs",
    "per_pair_uncertainty",
    "per_pair_flags",
}


def validate_metrics_payload(payload: dict[str, Any]) -> None:
    if set(payload) != TOP_KEYS:
        extra = set(payload) - TOP_KEYS
        missing = TOP_KEYS - set(payload)
        raise ValueError(f"metrics top-level keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")

    methods = payload["methods"]
    if not isinstance(methods, dict):
        raise ValueError("methods must be a dict")

    for method, stats in methods.items():
        if set(stats) != METHOD_KEYS:
            extra = set(stats) - METHOD_KEYS
            missing = METHOD_KEYS - set(stats)
            raise ValueError(
                f"method {method!r} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )


def write_metrics(path: Path, payload: dict[str, Any]) -> Path:
    validate_metrics_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
