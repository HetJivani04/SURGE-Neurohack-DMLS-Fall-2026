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
    OPTIONAL_UNCERTAINTY_KEYS,
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
from .uncertainty import (
    UNCERTAINTY_KEYS,
    barycentric_map_variance,
    heldout_predictive_score,
    mean_row_entropy,
    nonident_auroc,
    posterior_coverage,
    scale_template_frobenius,
    shrinkage_transform,
    subject_mean_tau,
    subject_row_entropies,
    temperature_calibrated_coverage,
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
    "OPTIONAL_UNCERTAINTY_KEYS",
    "UNCERTAINTY_KEYS",
    "shrinkage_transform",
    "posterior_coverage",
    "nonident_auroc",
    "heldout_predictive_score",
    "subject_mean_tau",
    "scale_template_frobenius",
    "mean_row_entropy",
    "subject_row_entropies",
    "barycentric_map_variance",
    "temperature_calibrated_coverage",
]
