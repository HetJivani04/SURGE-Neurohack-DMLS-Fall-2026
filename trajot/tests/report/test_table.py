from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path

import pytest

from fake_runs import EXPERIMENTS, method_stats, metrics_payload, write_full_registry, write_run
from trajot.report.table import (
    COLUMN_TITLES, EMPTY_CELL, MODEL_ROWS, ROW_METHODS, TABLE_COLUMNS, TABLE_EXPERIMENTS, TABLE_ROWS, MetricsError,
    collect_rows, render_table, render_table_meta, validate_metrics,
)

GOOD = metrics_payload({"noalign": method_stats(), "full": method_stats(model=True)})


def payload(**changes):
    out = json.loads(json.dumps(GOOD))
    out.update(changes)
    return out


# ---- the fixed shape -------------------------------------------------------------------------------------
def test_the_table_is_six_fixed_rows_and_five_fixed_columns() -> None:
    assert TABLE_ROWS == ["No alignment", "BrainSync", "FUGW", "connectivity-SRM", "Ours (ablated)", "Ours (full)"]
    assert TABLE_COLUMNS == ["method", "ident_accuracy", "perm_p", "per_pair_uncertainty", "nonidentifiable_pairs"]
    assert EMPTY_CELL == "—" and len(COLUMN_TITLES) == 5


def test_every_row_has_one_metrics_key_and_one_experiment() -> None:
    assert list(ROW_METHODS) == TABLE_ROWS and len(set(ROW_METHODS.values())) == 6
    assert list(EXPERIMENTS) == TABLE_EXPERIMENTS and MODEL_ROWS == ["Ours (ablated)", "Ours (full)"]
    assert [EXPERIMENTS[e][0] for e in TABLE_EXPERIMENTS] == list(ROW_METHODS.values())


# ---- validate_metrics ------------------------------------------------------------------------------------
def test_a_well_formed_payload_is_accepted() -> None:
    validate_metrics(GOOD, "run-a")


@pytest.mark.parametrize("key", ["methods", "n_pairs", "pairs_seed", "permutations_B", "beta"])
def test_a_missing_top_level_key_is_rejected_and_named(key: str) -> None:
    bad = payload()
    del bad[key]
    with pytest.raises(MetricsError) as err:
        validate_metrics(bad, "run-x")
    assert "run-x" in str(err.value) and key in str(err.value)


@pytest.mark.parametrize("key", ["ident_accuracy", "ident_ci", "perm_p", "null_max", "alignment_gain",
                                 "nonidentifiable_pairs", "per_pair_uncertainty", "per_pair_flags"])
@pytest.mark.parametrize("method", ["noalign", "full"])
def test_a_missing_per_method_key_is_rejected_and_named(method: str, key: str) -> None:
    bad = payload()
    del bad["methods"][method][key]
    with pytest.raises(MetricsError) as err:
        validate_metrics(bad, "run-y")
    assert "run-y" in str(err.value) and key in str(err.value) and method in str(err.value)


@pytest.mark.parametrize("bad_methods", [{}, [], None, "x"])
def test_methods_must_be_a_non_empty_dict(bad_methods) -> None:
    with pytest.raises(MetricsError, match="methods"):
        validate_metrics(payload(methods=bad_methods), "r")


def test_a_payload_that_is_not_an_object_is_rejected() -> None:
    with pytest.raises(MetricsError, match="run-q"):
        validate_metrics([GOOD], "run-q")


@pytest.mark.parametrize("key,value", [
    ("ident_accuracy", 1.2), ("ident_accuracy", -0.1), ("ident_accuracy", "0.5"), ("ident_accuracy", True),
    ("ident_accuracy", float("nan")), ("ident_ci", [0.1]), ("ident_ci", [0.1, "x"]), ("ident_ci", 0.5),
    ("ident_ci", [0.1, 0.2, 0.3]), ("perm_p", 0.0), ("perm_p", 1.5), ("perm_p", -0.2), ("perm_p", "x"),
    ("null_max", "a"), ("null_max", None), ("alignment_gain", "high"), ("alignment_gain", float("inf")),
    ("nonidentifiable_pairs", 1.5), ("nonidentifiable_pairs", -1), ("nonidentifiable_pairs", True),
    ("nonidentifiable_pairs", None),
])
def test_malformed_values_are_rejected_and_named(key: str, value) -> None:
    bad = payload()
    bad["methods"]["noalign"][key] = value
    with pytest.raises(MetricsError) as err:
        validate_metrics(bad, "run-z")
    assert key in str(err.value) and "run-z" in str(err.value)


def test_the_stated_ranges_are_inclusive_where_the_issue_says_so() -> None:
    for acc in (0.0, 1.0):
        ok = payload()
        ok["methods"]["noalign"]["ident_accuracy"] = acc
        validate_metrics(ok, "r")
    ok = payload()
    ok["methods"]["noalign"]["perm_p"] = 1.0  # (0, 1]: one is allowed, zero is not
    ok["methods"]["noalign"]["alignment_gain"] = -0.2  # a gain may be negative
    validate_metrics(ok, "r")


@pytest.mark.parametrize("key", ["per_pair_uncertainty", "per_pair_flags"])
@pytest.mark.parametrize("zero", [0, 0.0, [], False, "", {}, 0.31, [True, False]])
def test_a_baseline_reporting_anything_but_null_is_rejected(key: str, zero) -> None:
    bad = payload()
    bad["methods"]["noalign"][key] = zero
    with pytest.raises(MetricsError) as err:
        validate_metrics(bad, "run-b")
    assert key in str(err.value) and "null" in str(err.value) and "run-b" in str(err.value)


@pytest.mark.parametrize("baseline", ["noalign", "brainsync", "fugw", "conn_srm", "fake", "anything_else"])
def test_every_non_model_method_must_report_null_for_the_model_only_columns(baseline: str) -> None:
    with pytest.raises(MetricsError, match="null"):
        validate_metrics(payload(methods={baseline: method_stats(model=True)}), "r")
    validate_metrics(payload(methods={baseline: method_stats()}), "r")


@pytest.mark.parametrize("model", ["ablated", "full"])
@pytest.mark.parametrize("key", ["per_pair_uncertainty", "per_pair_flags"])
def test_a_model_row_must_populate_the_model_only_columns(model: str, key: str) -> None:
    validate_metrics(payload(methods={model: method_stats(model=True)}), "r")
    with pytest.raises(MetricsError) as err:
        validate_metrics(payload(methods={model: method_stats(model=True, **{key: None})}), "run-m")
    assert key in str(err.value) and "run-m" in str(err.value)


@pytest.mark.parametrize("key,value", [("per_pair_uncertainty", "wide"), ("per_pair_uncertainty", float("nan")),
                                       ("per_pair_flags", [1, 0]), ("per_pair_flags", "yes"), ("per_pair_flags", [])])
def test_model_only_columns_must_be_well_formed_when_present(key: str, value) -> None:
    with pytest.raises(MetricsError, match=key):
        validate_metrics(payload(methods={"full": method_stats(model=True, **{key: value})}), "r")


@pytest.mark.parametrize("key,value", [("n_pairs", None), ("n_pairs", 0), ("n_pairs", -5), ("n_pairs", 12.5),
                                       ("n_pairs", "500"), ("n_pairs", True), ("pairs_seed", None), ("pairs_seed", "x"),
                                       ("pairs_seed", 1.5), ("permutations_B", None), ("permutations_B", 0)])
def test_an_undeclared_subsample_is_rejected(key: str, value) -> None:
    with pytest.raises(MetricsError) as err:
        validate_metrics(payload(**{key: value}), "run-u")
    assert key in str(err.value) and "run-u" in str(err.value)


def test_beta_may_be_null_for_baseline_only_runs_but_must_be_a_number_otherwise() -> None:
    validate_metrics(payload(beta=None), "r")
    for bad in ("28", True, float("nan")):
        with pytest.raises(MetricsError, match="beta"):
            validate_metrics(payload(beta=bad), "r")


def test_validation_is_not_confused_by_extra_keys() -> None:
    extra = payload(anything=1)
    extra["methods"]["noalign"]["extra"] = [True]
    validate_metrics(extra, "r")


# ---- collect_rows ----------------------------------------------------------------------------------------
def test_collect_rows_returns_one_dict_per_table_row_in_table_order(tmp_path: Path) -> None:
    rows = collect_rows(write_full_registry(tmp_path))
    assert [r["method"] for r in rows] == TABLE_ROWS and not any(r["missing"] for r in rows)
    assert rows[0]["run_id"].startswith("00_noalign__") and rows[5]["run_id"].startswith("10_ours_full__")
    assert rows[4]["per_pair_uncertainty"] == 0.31 and rows[0]["per_pair_uncertainty"] is None
    assert rows[1]["ident_accuracy"] == pytest.approx(0.55) and rows[1]["experiment"] == "01_brainsync"
    assert {r["n_pairs"] for r in rows} == {500} and {r["pairs_seed"] for r in rows} == {2026}
    assert {r["permutations_B"] for r in rows} == {10000} and rows[5]["beta"] == 28.4


def test_the_latest_successful_run_per_experiment_wins(tmp_path: Path) -> None:
    for run_id, start, status, acc in (("old", "2026-09-20T08:00:00Z", "ok", 0.30), ("new", "2026-09-20T09:00:00Z", "ok", 0.60),
                                       ("newest_failed", "2026-09-20T10:00:00Z", "failed", 0.99),
                                       ("newest_running", "2026-09-20T11:00:00Z", "running", 0.99)):
        write_run(tmp_path, run_id, "01_brainsync", metrics_payload({"brainsync": method_stats(accuracy=acc)}),
                  status=status, start_utc=start)
    row = collect_rows(tmp_path / "index.csv", experiments=["01_brainsync"])[1]
    assert row["run_id"] == "new" and row["ident_accuracy"] == 0.60


def test_explicit_run_ids_override_the_latest_rule(tmp_path: Path) -> None:
    for run_id, start, acc in (("old", "2026-09-20T08:00:00Z", 0.30), ("new", "2026-09-20T09:00:00Z", 0.60)):
        write_run(tmp_path, run_id, "01_brainsync", metrics_payload({"brainsync": method_stats(accuracy=acc)}), start_utc=start)
    assert collect_rows(tmp_path / "index.csv", run_ids=["old"])[1]["ident_accuracy"] == 0.30
    with pytest.raises(ValueError, match="nope"):
        collect_rows(tmp_path / "index.csv", run_ids=["nope"])


def test_an_explicit_run_that_did_not_succeed_is_refused(tmp_path: Path) -> None:
    write_run(tmp_path, "bad", "01_brainsync", metrics_payload({"brainsync": method_stats()}), status="failed")
    with pytest.raises(ValueError, match="failed"):
        collect_rows(tmp_path / "index.csv", run_ids=["bad"])


def test_experiments_select_which_rows_are_filled(tmp_path: Path) -> None:
    index = write_full_registry(tmp_path)
    rows = collect_rows(index, experiments=["01_brainsync", "10_ours_full", "03_conn_srm"])
    assert [r["method"] for r in rows if not r["missing"]] == ["BrainSync", "connectivity-SRM", "Ours (full)"]
    assert [r["method"] for r in rows if r["missing"]] == ["No alignment", "FUGW", "Ours (ablated)"]
    with pytest.raises(ValueError, match="unknown experiment 'bogus'"):
        collect_rows(index, experiments=["bogus"])


def test_a_multi_method_run_fills_several_rows(tmp_path: Path) -> None:
    """W3's evaluate script reports many methods in one metrics.json."""
    methods = {k: method_stats(model=m, accuracy=0.4 + 0.1 * i) for i, (k, m) in enumerate(
        [("noalign", False), ("brainsync", False), ("fugw", False), ("conn_srm", False), ("ablated", True)])}
    write_run(tmp_path, "eval-run", "00_noalign", metrics_payload(methods, experiment="00_noalign"))
    rows = collect_rows(tmp_path / "index.csv")
    assert [r["method"] for r in rows if not r["missing"]] == TABLE_ROWS[:5] and rows[5]["missing"]
    assert {r["run_id"] for r in rows if not r["missing"]} == {"eval-run"}


def test_fake_runs_are_ignored_unless_asked_for(tmp_path: Path) -> None:
    only_fake = write_full_registry(tmp_path / "fake", fake=True)
    assert all(r["missing"] for r in collect_rows(only_fake))
    included = collect_rows(only_fake, include_fake=True)
    assert not any(r["missing"] for r in included)
    real = write_full_registry(tmp_path / "mixed")
    write_run(tmp_path / "mixed", "fake-newer", "00_noalign", metrics_payload({"noalign": method_stats(accuracy=0.99)}),
              start_utc="2026-09-21T00:00:00Z", fake=True)
    assert collect_rows(real)[0]["ident_accuracy"] == pytest.approx(0.5)  # a newer fake run never shadows a real one
    assert collect_rows(real, include_fake=True)[0]["ident_accuracy"] == pytest.approx(0.99)
    assert collect_rows(real, run_ids=["fake-newer"])[0]["ident_accuracy"] == pytest.approx(0.99)


def test_experiments_outside_the_table_are_ignored_by_default(tmp_path: Path) -> None:
    """The registry also holds preprocess and fit runs; they carry no comparison metrics."""
    index = write_full_registry(tmp_path)
    write_run(tmp_path, "preprocess__x", "preprocess", None)
    assert not any(r["missing"] for r in collect_rows(index))


def test_an_invalid_metrics_json_raises_naming_the_run_and_the_key(tmp_path: Path) -> None:
    write_run(tmp_path, "run-with-zero", "01_brainsync", metrics_payload({"brainsync": method_stats(per_pair_uncertainty=0.0)}))
    with pytest.raises(MetricsError) as err:
        collect_rows(tmp_path / "index.csv")
    assert "run-with-zero" in str(err.value) and "per_pair_uncertainty" in str(err.value)


def test_an_empty_methods_run_from_the_fit_script_is_rejected_by_name(tmp_path: Path) -> None:
    write_run(tmp_path, "fit-run", "10_ours_full", metrics_payload({}))
    with pytest.raises(MetricsError, match="fit-run.*methods"):
        collect_rows(tmp_path / "index.csv")


def test_a_registered_run_without_a_usable_metrics_file_is_reported_by_name(tmp_path: Path) -> None:
    write_run(tmp_path, "no-metrics", "01_brainsync", None)
    with pytest.raises(MetricsError, match="no-metrics"):
        collect_rows(tmp_path / "index.csv")
    (tmp_path / "corrupt").mkdir()
    (tmp_path / "corrupt" / "metrics.json").write_text("{not json")
    write_run(tmp_path, "corrupt", "02_fugw", None)
    with pytest.raises(MetricsError, match="corrupt.*JSON"):
        collect_rows(tmp_path / "index.csv", experiments=["02_fugw"])


def test_a_missing_registry_gives_an_all_missing_table(tmp_path: Path) -> None:
    rows = collect_rows(tmp_path / "runs" / "index.csv")
    assert [r["method"] for r in rows] == TABLE_ROWS and all(r["missing"] and r["run_id"] is None for r in rows)


# ---- render_table ----------------------------------------------------------------------------------------
def parse_markdown(text: str) -> list[list[str]]:
    return [[c.strip() for c in line.strip().strip("|").split("|")] for line in text.strip().splitlines()]


def test_the_rendered_table_is_exactly_six_rows_by_five_columns_for_any_input(tmp_path: Path) -> None:
    full = collect_rows(write_full_registry(tmp_path))
    rng = random.Random(0)
    inputs = [[], full, full[:2], full[3:], list(reversed(full)), full + full, [{"method": "Not a method", "missing": False}],
              [{"method": "FUGW", "missing": False}], [{"method": "Ours (full)"}]]
    inputs += [rng.sample(full, rng.randint(0, 6)) for _ in range(30)]
    for rows in inputs:
        cells = parse_markdown(render_table(rows, "markdown"))
        assert cells[0] == COLUMN_TITLES and len(cells) == 2 + 6 and all(len(line) == 5 for line in cells)
        assert [line[0].replace("*", "").replace(" (missing)", "") for line in cells[2:]] == TABLE_ROWS
        parsed = list(csv.reader(io.StringIO(render_table(rows, "csv"))))
        assert parsed[0] == TABLE_COLUMNS and len(parsed) == 1 + 6 and all(len(line) == 5 for line in parsed)
        assert [line[0].replace(" (missing)", "") for line in parsed[1:]] == TABLE_ROWS


def test_baseline_cells_in_the_two_right_hand_columns_are_an_em_dash_never_zero(tmp_path: Path) -> None:
    rows = parse_markdown(render_table(collect_rows(write_full_registry(tmp_path))))[2:]
    for baseline in rows[:4]:
        assert baseline[3] == EMPTY_CELL and baseline[4] == EMPTY_CELL
    for model in rows[4:]:
        assert model[3] == "0.310" and model[4] == "12"
    assert rows[1][1] == "0.55 [0.50, 0.60]" and rows[1][2] == "0.0001"


def test_a_baseline_shows_a_dash_even_though_its_run_carries_a_track_b_count(tmp_path: Path) -> None:
    """Every baseline reports a Track B count in metrics.json, but the *flagged* column is a model-only quantity."""
    rows = collect_rows(write_full_registry(tmp_path))
    assert rows[0]["nonidentifiable_pairs"] == 12
    assert parse_markdown(render_table(rows))[2][4] == EMPTY_CELL


def test_a_missing_run_renders_dashes_and_a_missing_marker() -> None:
    for line in parse_markdown(render_table([]))[2:]:
        assert line[0].endswith("(missing)") and line[1:] == [EMPTY_CELL] * 4
    for line in list(csv.reader(io.StringIO(render_table([], "csv"))))[1:]:
        assert line[0].endswith("(missing)") and line[1:] == [EMPTY_CELL] * 4


def test_model_rows_are_bold_in_markdown_when_present(tmp_path: Path) -> None:
    lines = parse_markdown(render_table(collect_rows(write_full_registry(tmp_path))))
    assert [lines[i][0] for i in (2, 6, 7)] == ["No alignment", "**Ours (ablated)**", "**Ours (full)**"]


def test_unknown_formats_are_rejected() -> None:
    with pytest.raises(ValueError, match="fmt"):
        render_table([], "latex")


def test_render_table_meta_states_the_declared_subsample() -> None:
    text = render_table_meta("two-run split", 2026, 500, 28.4)
    for expected in ("500", "2026", "two-run split", "28.4", "beta"):
        assert expected in text
    assert "draw" in text.lower() and "default_rng" in text
    assert "not reported" in render_table_meta("f", 2026, 500, None)
    assert "not declared" in render_table_meta("f", None, None, None)
