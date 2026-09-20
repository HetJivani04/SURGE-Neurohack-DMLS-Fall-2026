from __future__ import annotations

import numpy as np


def nonidentifiable_pairs(correct_mask: np.ndarray) -> int:
    mask = np.asarray(correct_mask, dtype=bool)
    return int(mask.size - np.count_nonzero(mask))
