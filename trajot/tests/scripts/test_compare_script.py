from __future__ import annotations

import ast
import builtins
import csv
import importlib.util
import io
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from fake_runs import EXPERIMENTS, dataset_like_manifest, method_stats, metrics_payload, write_full_registry, write_run
from trajot.report.sensitivity import SENSITIVITY_LABEL, SENSITIVITY_SUFFIX
from trajot.report.table import EMPTY_CELL, ROW_METHODS, TABLE_COLUMNS, TABLE_EXPERIMENTS, TABLE_ROWS

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script("compare")


@pytest.fixture
def runs_dir(tmp_path, script, monkeypatch) -> Path:
    monkeypatch.setattr(script, "RUNS_DIR", tmp_path / "runs")
    return tmp_path / "runs"


@pytest.fixture
def project(tmp_path, script, runs_dir, monkeypatch) -> Path:
    """Adds a configs/ with a paths.yaml and the dataset manifest, for --sensitivity."""
    shutil.copytree(ROOT / "configs", tmp_path / "configs", ignore=shutil.ignore_patterns("paths.yaml"))
    (tmp_path / "configs" / "paths.yaml").write_text(f"data_root: {tmp_path / 'data'}\ncontract_version: '1.0.0'\n")
    monkeypatch.setattr(script, "DEFAULT_CONFIG", tmp_path / "configs" / "experiments" / "10_ours_full.yaml")
    (tmp_path / "data" / "derivatives" / "trajot").mkdir(parents=True)
    dataset_like_manifest()[0].to_parquet(tmp_path / "data" / "derivatives" / "trajot" / "manifest.parquet")
    return tmp_path


def pipe_rows(text: str) -> list[list[str]]:
    return [[c.strip() for c in line.strip().strip("|").split("|")] for line in text.splitlines() if line.startswith("|")]


def run(script, capsys, *argv: str) -> tuple[int, str, str]:
    code = script.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ---- the acceptance criteria -----------------------------------------------------------------------------
def test_experiments_all_prints_the_six_by_five_table_built_from_the_registry_alone(script, runs_dir, capsys) -> None:
    write_full_registry(runs_dir)
    code, out, err = run(script, capsys, "--experiments", "all")
    cells = pipe_rows(out)
    assert code == 0 and err == "" and len(cells) == 2 + 6 and all(len(row) == 5 for row in cells)
    assert [row[0].replace("*", "") for row in cells[2:]] == TABLE_ROWS
    assert all(row[3] == row[4] == EMPTY_CELL for row in cells[2:6])  # baselines: empty on purpose
    assert all(row[3] not in (EMPTY_CELL, "0") and row[4] not in (EMPTY_CELL, "0") for row in cells[6:])  # both model rows populated
    assert cells[3][1] == "0.55 [0.50, 0.60]"


def test_the_declared_subsample_seed_fold_scheme_and_beta_accompany_the_table(script, runs_dir, capsys) -> None:
    write_full_registry(runs_dir)
    _, out, _ = run(script, capsys)
    for expected in ("500 ordered subject pairs, seed 2026", "default_rng", script.FOLD_SCHEME, "beta: 28.4"):
        assert expected in out


def test_it_works_when_only_the_fake_runs_are_present(script, runs_dir, capsys) -> None:
    write_full_registry(runs_dir, fake=True)
    code, out, _ = run(script, capsys, "--experiments", "all")
    rows = pipe_rows(out)[2:]
    assert code == 0 and len(rows) == 6 and all(row[0].endswith("(missing)") and row[1:] == [EMPTY_CELL] * 4 for row in rows)
    code, out, _ = run(script, capsys, "--experiments", "all", "--include-fake")
    rows = pipe_rows(out)[2:]
    assert code == 0 and not any("(missing)" in row[0] for row in rows) and rows[5][3] == "0.310"


def test_it_works_with_no_registry_at_all(script, runs_dir, capsys) -> None:
    code, out, _ = run(script, capsys)
    assert code == 0 and len(pipe_rows(out)) == 8 and out.count("(missing)") >= 6
    assert "not declared" in out and "not reported" in out


# ---- selecting runs --------------------------------------------------------------------------------------
def test_experiments_and_explicit_runs_select_what_is_compared(script, runs_dir, capsys) -> None:
    write_full_registry(runs_dir)
    write_run(runs_dir, "later-brainsync", "01_brainsync", metrics_payload({"brainsync": method_stats(accuracy=0.9)}),
              start_utc="2026-09-21T00:00:00Z")
    _, out, _ = run(script, capsys, "--experiments", "01_brainsync,10_ours_full")
    rows = pipe_rows(out)[2:]
    assert [row[0].endswith("(missing)") for row in rows] == [True, False, True, True, True, False]
    assert rows[1][1].startswith("0.90")  # the latest successful run
    first = next(p.name for p in runs_dir.iterdir() if p.name.startswith("01_brainsync__"))
    _, out, _ = run(script, capsys, "--runs", first)
    assert pipe_rows(out)[3][1].startswith("0.55")
    _, out, _ = run(script, capsys, "--experiments", "02_fugw", "--runs", first)  # --runs overrides --experiments
    assert not pipe_rows(out)[3][0].endswith("(missing)") and pipe_rows(out)[4][0].endswith("(missing)")


def test_the_source_run_of_every_row_is_listed_so_a_row_can_be_traced(script, runs_dir, capsys) -> None:
    write_full_registry(runs_dir)
    _, out, _ = run(script, capsys, "--experiments", "01_brainsync")
    assert "- BrainSync: 01_brainsync__00000001__20260920T100000Z" in out and "- FUGW: (missing)" in out


# ---- formats and files -----------------------------------------------------------------------------------
def test_csv_output_is_six_rows_by_five_columns_with_the_metadata_as_comments(script, runs_dir, capsys) -> None:
    write_full_registry(runs_dir)
    code, out, _ = run(script, capsys, "--format", "csv")
    body = [line for line in out.splitlines() if line and not line.startswith("#")]
    parsed = list(csv.reader(io.StringIO("\n".join(body))))
    assert code == 0 and parsed[0] == TABLE_COLUMNS and len(parsed) == 7 and all(len(row) == 5 for row in parsed)
    assert parsed[1][3] == EMPTY_CELL and parsed[5][3] == "0.310"
    assert any(line.startswith("# ") and "500" in line and "2026" in line for line in out.splitlines())


def test_out_writes_the_same_text_it_prints_creating_folders(script, runs_dir, tmp_path, capsys) -> None:
    write_full_registry(runs_dir)
    target = tmp_path / "reports" / "sub" / "table.md"
    code, out, _ = run(script, capsys, "--out", str(target))
    assert code == 0 and target.read_text() == out


# ---- failures name the run and the key -------------------------------------------------------------------
def test_an_invalid_run_fails_naming_the_run_and_the_key_and_prints_no_table(script, runs_dir, capsys) -> None:
    write_run(runs_dir, "zero-baseline", "01_brainsync", metrics_payload({"brainsync": method_stats(per_pair_flags=[])}))
    code, out, err = run(script, capsys, "--experiments", "all")
    assert code == 1 and out == "" and "zero-baseline" in err and "per_pair_flags" in err and "null" in err


def test_a_run_missing_a_required_key_fails_naming_them(script, runs_dir, capsys) -> None:
    stats = method_stats()
    del stats["null_max"]
    write_run(runs_dir, "no-null-max", "01_brainsync", metrics_payload({"brainsync": stats}))
    code, out, err = run(script, capsys)
    assert code == 1 and out == "" and "no-null-max" in err and "null_max" in err


def test_unknown_experiments_and_runs_are_reported(script, runs_dir, capsys) -> None:
    write_full_registry(runs_dir)
    code, out, err = run(script, capsys, "--experiments", "bogus")
    assert code == 1 and out == "" and "unknown experiment 'bogus'" in err
    code, out, err = run(script, capsys, "--runs", "nope")
    assert code == 1 and out == "" and "nope" in err


def test_runs_that_declare_different_subsamples_are_not_put_in_one_table(script, runs_dir, capsys) -> None:
    write_run(runs_dir, "a", "00_noalign", metrics_payload({"noalign": method_stats()}))
    write_run(runs_dir, "b", "01_brainsync", metrics_payload({"brainsync": method_stats()}, n_pairs=300))
    code, out, err = run(script, capsys)
    assert code == 1 and out == "" and "n_pairs" in err and "500" in err and "300" in err and "a" in err and "b" in err


def test_a_bad_metric_or_format_is_a_usage_error(script, runs_dir) -> None:
    for bad in (["--metric", "bogus"], ["--format", "latex"]):
        with pytest.raises(SystemExit) as err:
            script.main(bad)
        assert err.value.code == 2


# ---- --sensitivity and --metric --------------------------------------------------------------------------
def add_sensitivity_runs(runs_dir: Path) -> None:
    for i, (experiment, (key, model)) in enumerate(EXPERIMENTS.items()):
        run_id = f"{experiment}{SENSITIVITY_SUFFIX}__{i}"
        write_run(runs_dir, run_id, experiment + SENSITIVITY_SUFFIX,
                  metrics_payload({key: method_stats(model, 0.6 + 0.01 * i)}, experiment=experiment + SENSITIVITY_SUFFIX,
                                  run_id=run_id, n_subjects=26), start_utc="2026-09-20T12:00:00Z")


def test_sensitivity_appends_the_flagged_long_run_table_after_the_headline_table(script, project, runs_dir, capsys) -> None:
    write_full_registry(runs_dir)
    add_sensitivity_runs(runs_dir)
    code, plain, _ = run(script, capsys)
    code_s, out, err = run(script, capsys, "--sensitivity")
    assert code == code_s == 0 and err == "" and out.startswith(plain.rstrip("\n"))  # the headline table is untouched
    tail = out[len(plain.rstrip("\n")):]
    assert "Scan-length sensitivity" in tail and "not a headline number" in tail and "26" in tail
    rows = pipe_rows(tail)
    assert [row[0] for row in rows[2:]] == TABLE_ROWS and rows[0][:4] == ["method", "experiment", "run_id", "n_subjects"]
    assert "00_noalign_long__0" in tail and all("(missing)" not in row[0] for row in rows)


def test_sensitivity_with_no_long_run_evaluations_says_so(script, project, runs_dir, capsys) -> None:
    write_full_registry(runs_dir)
    code, out, _ = run(script, capsys, "--sensitivity")
    assert code == 0 and "Scan-length sensitivity" in out and "no runs" in out and SENSITIVITY_SUFFIX in out
    assert len(pipe_rows(out)) == 8  # only the headline table


def test_metric_restricts_the_sensitivity_columns_and_leaves_the_headline_table_alone(script, project, runs_dir, capsys) -> None:
    write_full_registry(runs_dir)
    add_sensitivity_runs(runs_dir)
    _, plain, _ = run(script, capsys)
    _, out, _ = run(script, capsys, "--sensitivity", "--metric", "perm_p")
    assert out.startswith(plain.rstrip("\n"))
    header = pipe_rows(out[len(plain.rstrip("\n")):])[0]
    assert header == ["method", "experiment", "run_id", "n_subjects", "perm_p"]
    _, out, _ = run(script, capsys, "--metric", "perm_p")  # accepted without --sensitivity; the five columns stay fixed
    assert out == plain
    _, out, _ = run(script, capsys, "--sensitivity", "--metric", "ident_accuracy")
    assert pipe_rows(out[len(plain.rstrip("\n")):])[0][4:] == ["ident_accuracy", "ident_ci_low", "ident_ci_high"]


def test_sensitivity_in_csv_carries_the_flag_on_every_row(script, project, runs_dir, capsys) -> None:
    write_full_registry(runs_dir)
    add_sensitivity_runs(runs_dir)
    _, out, _ = run(script, capsys, "--sensitivity", "--format", "csv")
    lines = [line for line in out.splitlines() if line and not line.startswith("#")]
    frame_rows = list(csv.DictReader(io.StringIO("\n".join(lines[7:]))))
    assert len(frame_rows) == 6 and {row["analysis"] for row in frame_rows} == {SENSITIVITY_LABEL}


def test_sensitivity_needs_the_dataset_manifest(script, project, runs_dir, capsys) -> None:
    write_full_registry(runs_dir)
    (project / "data" / "derivatives" / "trajot" / "manifest.parquet").unlink()
    code, out, err = run(script, capsys, "--sensitivity")
    assert code == 1 and out == "" and "manifest" in err.lower()


# ---- it never opens a run's internals and never re-runs anything ------------------------------------------
def test_only_the_registry_and_each_runs_metrics_json_are_opened(script, runs_dir, capsys, monkeypatch) -> None:
    write_full_registry(runs_dir)
    for run_dir in [p for p in runs_dir.iterdir() if p.is_dir()]:  # internals that must stay untouched, even if corrupt
        (run_dir / "manifest.json").write_text("{not json")
        (run_dir / "log.txt").write_text("\x00\xff")
        (run_dir / "artifacts").mkdir()
        (run_dir / "artifacts" / "params.pt").write_bytes(b"\x00")
    seen: set[str] = set()
    real_open, real_path_open = builtins.open, Path.open
    monkeypatch.setattr(builtins, "open", lambda file, *a, **k: (seen.add(str(file)), real_open(file, *a, **k))[1])
    monkeypatch.setattr(Path, "open", lambda self, *a, **k: (seen.add(str(self)), real_path_open(self, *a, **k))[1])

    def refuse(*args, **kwargs):
        raise AssertionError("compare.py started a process")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(os, "system", refuse)
    code, _, _ = run(script, capsys, "--experiments", "all")
    inside = {Path(p).relative_to(runs_dir).as_posix() for p in seen if str(runs_dir) in p}
    assert code == 0 and inside == {"index.csv", *(f"{p.name}/metrics.json" for p in runs_dir.iterdir() if p.is_dir())}


def process_starters(path: Path) -> list[str]:
    """Imports of process-starting modules and calls of os.system / os.popen / os.exec* / os.spawn* in a source file."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        found += [m for m in modules if m.split(".")[0] in {"subprocess", "runpy", "multiprocessing"}]
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
            if node.attr in {"system", "popen"} or node.attr.startswith(("exec", "spawn")):
                found.append(f"os.{node.attr}")
    return found


def test_the_comparison_code_cannot_start_a_run() -> None:
    sources = [ROOT / "scripts" / "compare.py", *sorted((ROOT / "src" / "trajot" / "report").glob("*.py"))]
    assert len(sources) >= 5 and {p.name for p in sources} >= {"compare.py", "table.py", "group.py", "sensitivity.py", "figures.py"}
    assert {p.name: process_starters(p) for p in sources if process_starters(p)} == {}


# ---- the constants agree with the scripts that produce the runs ------------------------------------------
def test_the_table_experiments_are_the_experiments_of_run_experiment() -> None:
    assert set(TABLE_EXPERIMENTS) == set(load_script("run_experiment").EXPERIMENT_NAMES)


def test_the_method_keys_of_the_rows_are_the_names_w3_evaluates() -> None:
    from trajot.baselines import BASELINE_REGISTRY

    assert set(list(ROW_METHODS.values())[:5]) <= set(BASELINE_REGISTRY)  # the model row "full" has no W3 entry yet


def test_the_fold_scheme_describes_w3s_two_run_split(script) -> None:
    from trajot.eval.folds import make_two_run_splits

    query, gallery = make_two_run_splits(["b", "a", "c"])
    assert query == gallery == ["a", "b", "c"]  # the property the scheme text states
    assert "run 1" in script.FOLD_SCHEME and "run 2" in script.FOLD_SCHEME
