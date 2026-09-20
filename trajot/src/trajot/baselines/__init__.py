from __future__ import annotations

from .ablated import AblatedModel
from .base import (
    BASELINE_REGISTRY,
    Baseline,
    BaselineResult,
    FakeBaseline,
    cfg_get,
    get_baseline,
    register_baseline,
    upper_triangle_features,
)
from .brainsync import BrainSync, brainsync_Q, to_connectome_transform
from .conn_srm import ConnSRM
from .fugw import FUGW
from .harness import align_features, run_baseline
from .noalign import NoAlign
from .ours import (
    OursAblated,
    OursFull,
    calibrated_tau0,
    mix_identity_orthogonal,
    pool_pi_to_regions,
    shrink_pi_hierarchical,
)

__all__ = [
    "Baseline",
    "BaselineResult",
    "BASELINE_REGISTRY",
    "register_baseline",
    "get_baseline",
    "FakeBaseline",
    "NoAlign",
    "BrainSync",
    "brainsync_Q",
    "to_connectome_transform",
    "FUGW",
    "ConnSRM",
    "AblatedModel",
    "OursFull",
    "OursAblated",
    "calibrated_tau0",
    "mix_identity_orthogonal",
    "pool_pi_to_regions",
    "shrink_pi_hierarchical",
    "run_baseline",
    "align_features",
    "cfg_get",
    "upper_triangle_features",
]
