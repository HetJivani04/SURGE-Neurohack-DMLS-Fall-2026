from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd
import pytest

from fake_runs import dataset_like_manifest, manifest, method_stats, metrics_payload, write_full_registry, write_run
from trajot.report.sensitivity import SENSITIVITY_COLUMNS, SENSITIVITY_LABEL, SENSITIVITY_SUFFIX, long_run_subset, scan_length_metrics
from trajot.report.table import MetricsError, TABLE_EXPERIMENTS, collect_rows


# ---- long_run_subset -------------------------------------------------------------------------------------
def test_the_long_run_subset_is_the_15_5_6_subjects_of_the_dataset() -> None:
    frame, expected = dataset_like_manifest()
    subset = long_run_subset(frame)
    assert subset == expected and len(subset) == 26
    counts = frame[frame.subject_id.isin(subset)].n_volumes.value_counts().to_dict()
    assert counts == {360: 15, 480: 5, 724: 6}


def test_the_threshold_is_inclusive_and_a_subject_qualifies_through_any_of_its_runs() -> None:
    frame = manifest([("a", "1", 300), ("b", "1", 299), ("c", "1", 132), ("c", "2", 360)])
    assert long_run_subset(frame) == ["a", "c"]
    assert long_run_subset(frame, min_volumes=301) == ["c"]
    assert long_run_subset(frame, min_volumes=299) == ["a", "b", "c"]


def test_the_long_run_subset_matches_a_brute_force_selection_for_random_manifests() -> None:
    rng = random.Random(0)
    for _ in range(40):
        runs = [(f"{s:03d}", str(r), rng.choice([130, 132, 240, 300, 360, 724]))
                for s in range(rng.randint(1, 30)) for r in range(1, rng.randint(1, 2) + 1)]
        threshold = rng.choice([1, 132, 300, 361, 1000])
        expected = sorted({s for s, _, n in runs if n >= threshold})
        result = long_run_subset(manifest(runs), threshold)
        assert result == expected and all(isinstance(s, str) for s in result)


def test_the_long_run_subset_of_nothing_is_empty_and_bad_input_is_rejected() -> None:
    assert long_run_subset(manifest([("a", "1", 132)])) == []
    assert long_run_subset(manifest([("a", "1", 132)])[:0]) == []
    with pytest.raises(ValueError, match="n_volumes"):
        long_run_subset(pd.DataFrame({"subject_id": ["a"]}))
    with pytest.raises(ValueError, match="subject_id"):
        long_run_subset(pd.DataFrame({"n_volumes": [400]}))
    for bad in (0, -5, 3.5, True):
        with pytest.raises(ValueError, match="min_volumes"):
            long_run_subset(manifest([("a", "1", 400)]), bad)


# ---- scan_length_metrics ---------------------------------------------------------------------------------
SUBSET = [f"{i:03d}" for i in range(26)]


def sensitivity_run(runs_dir: Path, experiment: str, key: str, model: bool, run_id: str, n_subjects: int = 26,
                    accuracy: float = 0.7, start_utc: str = "2026-09-20T12:00:00Z", **kw) -> None:
    write_run(runs_dir, run_id, experiment + SENSITIVITY_SUFFIX,
              metrics_payload({key: method_stats(model, accuracy)}, experiment=experiment + SENSITIVITY_SUFFIX,
                              run_id=run_id, n_subjects=n_subjects), start_utc=start_utc, **kw)


def test_the_columns_and_the_flag_mark_the_output_as_a_sensitivity_analysis(tmp_path: Path) -> None:
    sensitivity_run(tmp_path, "01_brainsync", "brainsync", False, "s-bs", accuracy=0.61)
    sensitivity_run(tmp_path, "10_ours_full", "ours_full", True, "s-full", accuracy=0.72)
    frame = scan_length_metrics(tmp_path / "index.csv", SUBSET)
    assert list(frame.columns) == SENSITIVITY_COLUMNS
    assert (frame["analysis"] == SENSITIVITY_LABEL).all() and "sensitivity" in SENSITIVITY_LABEL
    assert list(frame["method"]) == ["BrainSync", "Ours (full)"] and list(frame["run_id"]) == ["s-bs", "s-full"]
    row = frame.iloc[0]
    assert (row.n_subjects, row.ident_accuracy, row.ident_ci_low, row.ident_ci_high) == (26, 0.61, pytest.approx(0.56), pytest.approx(0.66))
    assert row.perm_p == 0.0001 and row.nonidentifiable_pairs == 12 and math.isnan(row.per_pair_uncertainty)
    assert frame.iloc[1].per_pair_uncertainty == 0.31


def test_no_sensitivity_run_gives_an_empty_frame_with_the_columns_not_an_error(tmp_path: Path) -> None:
    for index in (write_full_registry(tmp_path), tmp_path / "nothing" / "index.csv"):
        frame = scan_length_metrics(index, SUBSET)
        assert frame.empty and list(frame.columns) == SENSITIVITY_COLUMNS


def test_sensitivity_runs_never_reach_the_headline_table(tmp_path: Path) -> None:
    index = write_full_registry(tmp_path)
    before = collect_rows(index)
    sensitivity_run(tmp_path, "01_brainsync", "brainsync", False, "s-bs", accuracy=0.99, start_utc="2030-01-01T00:00:00Z")
    assert collect_rows(index) == before
    assert not any(name.endswith(SENSITIVITY_SUFFIX) for name in TABLE_EXPERIMENTS)


def test_the_latest_successful_sensitivity_run_per_experiment_is_used(tmp_path: Path) -> None:
    sensitivity_run(tmp_path, "01_brainsync", "brainsync", False, "old", accuracy=0.3, start_utc="2026-09-20T08:00:00Z")
    sensitivity_run(tmp_path, "01_brainsync", "brainsync", False, "new", accuracy=0.6, start_utc="2026-09-20T09:00:00Z")
    sensitivity_run(tmp_path, "01_brainsync", "brainsync", False, "bad", accuracy=0.9, start_utc="2026-09-20T10:00:00Z", status="failed")
    frame = scan_length_metrics(tmp_path / "index.csv", SUBSET)
    assert list(frame.run_id) == ["new"] and list(frame.ident_accuracy) == [0.6]


def test_experiments_select_which_headline_experiments_are_repeated(tmp_path: Path) -> None:
    sensitivity_run(tmp_path, "01_brainsync", "brainsync", False, "s-bs")
    sensitivity_run(tmp_path, "10_ours_full", "ours_full", True, "s-full")
    assert list(scan_length_metrics(tmp_path / "index.csv", SUBSET, experiments=["10_ours_full"]).run_id) == ["s-full"]
    assert scan_length_metrics(tmp_path / "index.csv", SUBSET, experiments=["02_fugw"]).empty


def test_a_run_evaluated_on_a_different_number_of_subjects_is_rejected_by_name(tmp_path: Path) -> None:
    sensitivity_run(tmp_path, "01_brainsync", "brainsync", False, "whole-cohort", n_subjects=83)
    with pytest.raises(MetricsError) as err:
        scan_length_metrics(tmp_path / "index.csv", SUBSET)
    assert "whole-cohort" in str(err.value) and "n_subjects" in str(err.value) and "83" in str(err.value) and "26" in str(err.value)


def test_an_invalid_sensitivity_payload_is_rejected_by_name(tmp_path: Path) -> None:
    write_run(tmp_path, "s-bad", "01_brainsync" + SENSITIVITY_SUFFIX,
              metrics_payload({"brainsync": method_stats(per_pair_uncertainty=0.0)}, n_subjects=26))
    with pytest.raises(MetricsError, match="s-bad.*per_pair_uncertainty"):
        scan_length_metrics(tmp_path / "index.csv", SUBSET)


def test_fake_sensitivity_runs_are_left_out_unless_asked_for(tmp_path: Path) -> None:
    sensitivity_run(tmp_path, "01_brainsync", "brainsync", False, "s-fake", fake=True)
    assert scan_length_metrics(tmp_path / "index.csv", SUBSET).empty
    assert list(scan_length_metrics(tmp_path / "index.csv", SUBSET, include_fake=True).run_id) == ["s-fake"]


def test_an_empty_subset_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        scan_length_metrics(Path("does-not-matter") / "index.csv", [])
