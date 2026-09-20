"""Figures for the write-up: the identifiability figure (posterior widths) and the permutation nulls.

Both draw on a bare ``matplotlib.figure.Figure`` (no pyplot, so no backend or global state) and save a PNG.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def _widths_figure(tau_phi: np.ndarray) -> Figure:
    """Top: per-subject median and interquartile range of the width over vertices. Bottom: the (S, V) width map.

    Subjects are sorted by median width, narrowest first, in both panels.
    """
    order = np.argsort(np.median(tau_phi, axis=1))
    ranked = tau_phi[order]
    q1, median, q3 = np.percentile(ranked, [25, 50, 75], axis=1)
    fig = Figure(figsize=(9, 6), layout="constrained")
    top, bottom = fig.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]})
    rank = np.arange(len(ranked))
    top.fill_between(rank, q1, q3, alpha=0.3, label="interquartile range over vertices")
    top.plot(rank, median, marker=".", label="median over vertices")
    top.set_xlabel("subject (sorted by median width)")
    top.set_ylabel("posterior width")
    top.legend()
    image = bottom.imshow(ranked, aspect="auto", interpolation="nearest", origin="lower")
    bottom.set_xlabel("vertex")
    bottom.set_ylabel("subject (sorted by median width)")
    fig.colorbar(image, ax=bottom, label="posterior width")
    return fig


def plot_posterior_widths(tau_phi: np.ndarray, out_path: Path) -> Path:
    """Per-subject posterior width across the cortex, the identifiability figure. ``tau_phi`` is (S, V) float64."""
    tau = np.asarray(tau_phi, dtype=np.float64)
    if tau.ndim != 2 or tau.size == 0:
        raise ValueError(f"tau_phi must be a non-empty (S, V) array, got shape {tau.shape}")
    if not np.isfinite(tau).all() or (tau < 0).any():
        raise ValueError("tau_phi must be finite and non-negative")
    return _save(_widths_figure(tau), out_path)


def _null_figure(nulls: dict[str, np.ndarray], observed: dict[str, float]) -> Figure:
    """One panel per method: its null as a histogram, the observed statistic as a vertical line."""
    ncols = min(3, len(nulls))
    nrows = math.ceil(len(nulls) / ncols)
    fig = Figure(figsize=(4.2 * ncols, 3.2 * nrows), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
    for ax, (name, null) in zip(axes, nulls.items()):
        ax.hist(null, bins=40, color="0.6")
        ax.axvline(observed[name], color="C3", label="observed")
        ax.set_title(name)
        ax.set_xlabel("permutation null statistic")
        ax.set_ylabel("count")
        ax.legend()
    for ax in axes[len(nulls):]:
        ax.set_visible(False)
    return fig


def plot_null_distributions(nulls: dict[str, np.ndarray], observed: dict[str, float], out_path: Path) -> Path:
    """Each method's permutation null with its observed statistic marked. ``nulls`` maps method to a 1-D array."""
    if not nulls:
        raise ValueError("nulls is empty")
    if set(nulls) != set(observed):
        raise ValueError(f"nulls and observed must name the same methods, got {sorted(nulls)} and {sorted(observed)}")
    clean = {}
    for name, null in nulls.items():
        array = np.asarray(null, dtype=np.float64)
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"null of {name!r} must be a non-empty finite 1-D array")
        if not math.isfinite(observed[name]):
            raise ValueError(f"observed statistic of {name!r} must be finite")
        clean[name] = array
    return _save(_null_figure(clean, {name: float(observed[name]) for name in clean}), out_path)


def _save(fig: Figure, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png", dpi=150)
    return out_path
