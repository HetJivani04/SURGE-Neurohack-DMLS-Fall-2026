from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

from trajot.report.figures import _null_figure, _widths_figure, plot_null_distributions, plot_posterior_widths

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def widths(rng, S=12, V=300):
    return rng.gamma(2.0, 0.1, size=(S, 1)) * rng.uniform(0.5, 1.5, size=(S, V))


def is_png_with_ink(path: Path) -> bool:
    if path.read_bytes()[:8] != PNG_MAGIC:
        return False
    image = mpimg.imread(path)
    return image.ndim == 3 and min(image.shape[:2]) > 100 and float(image[..., :3].min()) < 0.5


# ---- plot_posterior_widths -------------------------------------------------------------------------------
def test_posterior_widths_writes_a_png_and_returns_its_path(tmp_path: Path) -> None:
    tau = widths(np.random.default_rng(0))
    out = tmp_path / "figures" / "nested" / "widths.png"
    assert plot_posterior_widths(tau, out) == out and is_png_with_ink(out)
    assert plot_posterior_widths(tau, str(tmp_path / "again.png")) == tmp_path / "again.png"


def test_posterior_widths_does_not_change_its_input_and_depends_on_it(tmp_path: Path) -> None:
    tau = widths(np.random.default_rng(1))
    before = tau.copy()
    a = plot_posterior_widths(tau, tmp_path / "a.png")
    np.testing.assert_array_equal(tau, before)
    b = plot_posterior_widths(tau, tmp_path / "b.png")
    c = plot_posterior_widths(2.0 * tau, tmp_path / "c.png")
    assert a.read_bytes() == b.read_bytes() and a.read_bytes() != c.read_bytes()


def test_the_widths_figure_shows_per_subject_summaries_sorted_by_width_and_the_map() -> None:
    rng = np.random.default_rng(2)
    tau = widths(rng, S=9, V=120)
    top, bottom = _widths_figure(tau).axes[:2]
    medians = np.median(tau, axis=1)
    line = top.lines[0]
    np.testing.assert_allclose(line.get_ydata(), np.sort(medians))  # subjects ordered by their median width
    image = bottom.images[0].get_array()
    assert image.shape == tau.shape
    np.testing.assert_allclose(np.asarray(image), tau[np.argsort(medians)])


def test_posterior_widths_works_at_the_dataset_size(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    assert is_png_with_ink(plot_posterior_widths(rng.gamma(2.0, 0.05, size=(120, 5124)), tmp_path / "full.png"))


@pytest.mark.parametrize("bad", [np.ones(5), np.ones((2, 3, 4)), np.empty((0, 4)), np.empty((4, 0)), np.full((3, 4), np.nan),
                                 np.full((3, 4), np.inf), -np.ones((3, 4))])
def test_posterior_widths_rejects_malformed_input(bad, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plot_posterior_widths(bad, tmp_path / "x.png")
    assert not (tmp_path / "x.png").exists()


# ---- plot_null_distributions -----------------------------------------------------------------------------
def nulls_for(names, rng):
    return {n: rng.normal(0.1, 0.02, size=2000) for n in names}, {n: 0.1 + 0.03 * (i + 1) for i, n in enumerate(names)}


@pytest.mark.parametrize("n_methods", [1, 2, 5, 6])
def test_null_distributions_writes_a_png_for_any_number_of_methods(n_methods: int, tmp_path: Path) -> None:
    names = ["No alignment", "BrainSync", "FUGW", "connectivity-SRM", "Ours (ablated)", "Ours (full)"][:n_methods]
    nulls, observed = nulls_for(names, np.random.default_rng(4))
    out = tmp_path / "nulls" / f"{n_methods}.png"
    assert plot_null_distributions(nulls, observed, out) == out and is_png_with_ink(out)


def test_each_method_gets_its_own_panel_with_the_observed_statistic_marked() -> None:
    names = ["A", "B", "C"]
    nulls, observed = nulls_for(names, np.random.default_rng(5))
    fig = _null_figure(nulls, observed)
    panels = [ax for ax in fig.axes if ax.get_visible() and ax.patches]
    assert [ax.get_title() for ax in panels] == names
    for ax, name in zip(panels, names):
        assert sum(p.get_height() for p in ax.patches) == len(nulls[name])  # the histogram counts every null value
        marks = [line for line in ax.lines if len(set(np.asarray(line.get_xdata()).tolist())) == 1]
        assert [float(m.get_xdata()[0]) for m in marks] == [observed[name]]


def test_null_distributions_output_depends_on_the_observed_value(tmp_path: Path) -> None:
    nulls, observed = nulls_for(["A", "B"], np.random.default_rng(6))
    a = plot_null_distributions(nulls, observed, tmp_path / "a.png")
    b = plot_null_distributions(nulls, dict(observed, A=observed["A"] + 0.02), tmp_path / "b.png")
    assert a.read_bytes() != b.read_bytes()


def test_null_distributions_rejects_malformed_input(tmp_path: Path) -> None:
    good, observed = nulls_for(["A", "B"], np.random.default_rng(7))
    cases = [({}, {}), (good, {"A": 0.1}), (good, dict(observed, C=0.3)), (good, dict(observed, A=float("nan"))),
             (dict(good, A=np.array([])), observed), (dict(good, A=np.ones((3, 3))), observed),
             (dict(good, A=np.array([0.1, np.nan])), observed)]
    for nulls, obs in cases:
        with pytest.raises(ValueError):
            plot_null_distributions(nulls, obs, tmp_path / "x.png")
    assert not (tmp_path / "x.png").exists()
