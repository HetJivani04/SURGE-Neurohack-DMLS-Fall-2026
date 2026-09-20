"""The dependency is one-way: trajot.report imports from other packages and no module of W0-W3 imports it."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "src" / "trajot" / "report"
W0_W3_PACKAGES = ("io", "geometry", "model", "inference", "baselines", "eval", "runlog")
W0_W3_SCRIPTS = ("preprocess", "fit", "evaluate", "run_experiment")


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            modules.add(base)
            modules |= {f"{base}.{alias.name}" for alias in node.names}  # `from trajot import report`
    return modules


def imports_report(path: Path) -> bool:
    return any(m == "trajot.report" or m.startswith("trajot.report.") for m in imported_modules(path))


def test_no_module_of_w0_to_w3_imports_the_report_package() -> None:
    sources = [p for pkg in W0_W3_PACKAGES for p in (ROOT / "src" / "trajot" / pkg).rglob("*.py")]
    sources += [ROOT / "src" / "trajot" / "config.py", ROOT / "src" / "trajot" / "__init__.py"]
    sources += [ROOT / "scripts" / f"{name}.py" for name in W0_W3_SCRIPTS]
    sources += [p for p in (ROOT / "scripts").glob("merge_*.py")]
    assert len(sources) > 25  # the scan really covered the packages
    assert [str(p.relative_to(ROOT)) for p in sources if imports_report(p)] == []


def test_the_scan_would_catch_an_import(tmp_path: Path) -> None:
    for text in ("import trajot.report.table", "from trajot.report import table", "from trajot import report",
                 "from trajot.report.group import meta_analysis"):
        bad = tmp_path / "bad.py"
        bad.write_text(text + "\n")
        assert imports_report(bad), text
    ok = tmp_path / "ok.py"
    ok.write_text("import trajot.reports_elsewhere\nfrom trajot import runlog\n")
    assert not imports_report(ok)


def test_the_report_package_imports_no_report_internals_of_other_workstreams_state() -> None:
    """It only reads: nothing in it imports a training or model entry point."""
    heavy = {"trajot.model", "trajot.inference", "trajot.geometry"}
    for path in REPORT.glob("*.py"):
        offenders = {m for m in imported_modules(path) if any(m == h or m.startswith(h + ".") for h in heavy)}
        assert offenders == set(), f"{path.name} imports {offenders}"
