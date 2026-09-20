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
    "null_max",
    "alignment_gain",
    "nonidentifiable_pairs",
    "per_pair_uncertainty",
    "per_pair_flags",
}

# Evaluator diagnostics (D7 extension): make identity fallback visible in metrics.json.
# Required METHOD_KEYS remain mandatory; these are optional extras.
DIAGNOSTIC_METHOD_KEYS = {
    "fit_error",
    "transforms_applied",
    "transform_diagnostics",
    "status",
}
ALLOWED_METHOD_KEYS = METHOD_KEYS | DIAGNOSTIC_METHOD_KEYS

# Columns baselines often cannot fill; Agent D serializes these as JSON null.
OPTIONAL_METHOD_KEYS = {
    "null_max",
    "per_pair_uncertainty",
    "ident_ci",
    "perm_p",
    "alignment_gain",
    "nonidentifiable_pairs",
    "per_pair_flags",
    "ident_accuracy",
}


def method_metrics_template() -> dict[str, Any]:
    """All method keys present with JSON-null values.

    Use when a baseline cannot fill a column; never invent zeros.
    """
    return {key: None for key in sorted(METHOD_KEYS)}


def validate_metrics_payload(payload: dict[str, Any]) -> None:
    if set(payload) != TOP_KEYS:
        extra = set(payload) - TOP_KEYS
        missing = TOP_KEYS - set(payload)
        raise ValueError(f"metrics top-level keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")

    methods = payload["methods"]
    if not isinstance(methods, dict):
        raise ValueError("methods must be a dict")

    for method, stats in methods.items():
        missing = METHOD_KEYS - set(stats)
        extra = set(stats) - ALLOWED_METHOD_KEYS
        if missing or extra:
            raise ValueError(
                f"method {method!r} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for key, value in stats.items():
            if value is None:
                continue
            if key == "ident_accuracy" and not isinstance(value, (int, float)):
                raise ValueError(f"method {method!r} ident_accuracy must be float or None")
            if key == "perm_p" and not isinstance(value, (int, float)):
                raise ValueError(f"method {method!r} perm_p must be float or None")
            if key == "null_max" and not isinstance(value, (int, float)):
                raise ValueError(f"method {method!r} null_max must be float or None")
            if key == "alignment_gain" and not isinstance(value, (int, float)):
                raise ValueError(f"method {method!r} alignment_gain must be float or None")
            if key == "per_pair_uncertainty" and not isinstance(value, (int, float)):
                raise ValueError(f"method {method!r} per_pair_uncertainty must be float or None")


def write_metrics(path: Path, payload: dict[str, Any]) -> Path:
    validate_metrics_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
