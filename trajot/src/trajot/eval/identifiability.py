from __future__ import annotations

import numpy as np


def nonidentifiable_count(
    per_pair_gains: np.ndarray,
    null_dist: np.ndarray,
    alpha: float = 0.05,
) -> tuple[int, np.ndarray]:
    """Count pairs whose gain is not significantly above the method's null.

    A pair is non-identifiable when its per-pair alignment gain is not
    greater than the one-sided α-level critical value of the method's own
    permutation null (shuffled subject pairing). Equivalently, the gain
    falls inside the null bulk at ``alpha``.

    Returns ``(count, flags)`` where ``flags[k]`` is True iff pair ``k`` is
    non-identifiable.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    gains = np.asarray(per_pair_gains, dtype=np.float64).ravel()
    null = np.asarray(null_dist, dtype=np.float64).ravel()
    if null.size == 0:
        raise ValueError("null_dist must be non-empty")
    if gains.size == 0:
        return 0, np.zeros(0, dtype=bool)

    # One-sided upper-tail critical value: significant gain exceeds this.
    # "Not significantly above null" => flag when gain <= quantile(null, 1-alpha).
    threshold = float(np.quantile(null, 1.0 - alpha))
    flags = gains <= threshold
    return int(np.count_nonzero(flags)), flags.astype(bool)


def nonidentifiable_pairs(
    per_pair_gains: np.ndarray,
    null_dist: np.ndarray | None = None,
    alpha: float = 0.05,
) -> int:
    """Count of non-identifiable pairs.

    With ``null_dist``: Track B statistic (gains vs permutation null).
    Without ``null_dist``: backward-compat path that counts misses in a
    boolean ``correct_mask``.
    """
    if null_dist is None:
        mask = np.asarray(per_pair_gains, dtype=bool)
        return int(mask.size - np.count_nonzero(mask))
    count, _ = nonidentifiable_count(per_pair_gains, null_dist, alpha=alpha)
    return count


def posterior_width_flag(
    tau_phi: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Flag subjects whose mean posterior width exceeds ``threshold``.

    Model-only W4 column. True => high uncertainty / non-identifiable.
    Accepts ``(S,)`` or ``(S, V)`` (or ``(S, V, ...)``) tau arrays.
    """
    tau = np.asarray(tau_phi, dtype=np.float64)
    if tau.ndim == 0:
        raise ValueError("tau_phi must be at least 1D")
    if tau.ndim == 1:
        means = tau
    else:
        means = tau.reshape(tau.shape[0], -1).mean(axis=1)
    return (means > float(threshold)).astype(bool)
