from __future__ import annotations

from .alignment_gain import alignment_gain, pair_gain
from .controls import (
    ControlResult,
    banded_coupling,
    random_permutation_control,
    shuffle_time_control,
)
from .folds import (
    FoldAssignment,
    identification_subjects,
    make_folds,
    make_two_run_splits,
    sample_pairs,
)
from .identifiability import (
    nonidentifiable_count,
    nonidentifiable_pairs,
    posterior_width_flag,
)
from .identification import (
    IdentificationResult,
    accuracy_ci,
    flatten_features,
    identification_accuracy,
    identify_subjects,
    pearson_scores,
)
from .metrics import (
    METHOD_KEYS,
    TOP_KEYS,
    method_metrics_template,
    validate_metrics_payload,
    write_metrics,
)
from .permutation import (
    PermutationResult,
    null_max,
    permutation_null,
    permutation_p,
    sign_flip_permutation,
)

__all__ = [
    "alignment_gain",
    "pair_gain",
    "identify_subjects",
    "identification_accuracy",
    "flatten_features",
    "pearson_scores",
    "accuracy_ci",
    "nonidentifiable_pairs",
    "nonidentifiable_count",
    "posterior_width_flag",
    "permutation_p",
    "permutation_null",
    "sign_flip_permutation",
    "null_max",
    "random_permutation_control",
    "banded_coupling",
    "shuffle_time_control",
    "make_folds",
    "identification_subjects",
    "make_two_run_splits",
    "sample_pairs",
    "FoldAssignment",
    "validate_metrics_payload",
    "write_metrics",
    "method_metrics_template",
    "IdentificationResult",
    "PermutationResult",
    "ControlResult",
    "TOP_KEYS",
    "METHOD_KEYS",
]
