"""The write-up keeps the claim discipline, states its status honestly, and carries the table exactly as compare.py rendered it."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = (ROOT / "reports" / "RESULTS.md").read_text(encoding="utf-8")
TABLE = (ROOT / "reports" / "results_table.md").read_text(encoding="utf-8")


def has(text: str, *needles: str) -> bool:
    return all(n.lower() in text.lower() for n in needles)


def _fold(text: str) -> str:
    """Case- and accent-insensitive form, so the ASCII needle 'Memoli' matches the document's 'Mémoli'."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


# ---- what the write-up must contain (W4 issue, items 19 and 20) -------------------------------------------
@pytest.mark.parametrize("needles", [
    ("the results table",), ("declared pairs", "500", "seed", "2026", "10,000"),
    ("paired bootstrap", "10,000", "mcnemar"), ("identifiability", "withdrawn"),
    ("beta", "calibrat", "83 strict two-run"), ("posterior row-entropy", "coverage_is_bayes=false"),
    ("scan-length sensitivity", "not run"), ("what failed", "detached entropy jacobian"),
    ("python", ".central_venv"), ("ground-truth correspondence",),
    ("random-effects", "one-sample t-test"),
], ids=lambda needles: needles[0])
def test_the_write_up_has_every_required_section(needles) -> None:
    assert has(RESULTS, *needles), needles


def test_the_table_and_its_metadata_are_exactly_what_compare_rendered() -> None:
    assert TABLE.strip() in RESULTS
    assert has(TABLE, "Declared pair subsample", "Draw procedure", "Fold scheme", "beta", "Source runs")
    assert len([line for line in TABLE.splitlines() if line.startswith("|")]) == 2 + 6  # header, rule, six rows


def test_the_write_up_states_its_status_and_does_not_invent_wins() -> None:
    assert has(RESULTS, "Phase 2 REAL N=83", "does not invent")
    assert has(RESULTS, "scan-length sensitivity", "not run")  # long-run sensitivity remains open (long runs are one-run subjects)
    assert has(RESULTS, "10,000", "B=200")
    # Frozen comparison table cells are reported numbers, not empty placeholders.
    cells = re.findall(r"^\|[^|]+\|([^|]+)\|", TABLE, flags=re.M)[2:]
    assert any(cell.strip() not in ("—", "") for cell in cells)


# ---- the claim-discipline appendix ------------------------------------------------------------------------
def test_it_never_writes_unique_minimizer_and_states_finitely_many_optima() -> None:
    assert "unique minimizer" not in RESULTS.lower() and "unique minimiser" not in RESULTS.lower()
    assert "finitely many optima" in RESULTS


@pytest.mark.parametrize("attribution", ["FUGW", "Thual", "ULOT", "Mazelet", "Mallasto", "Keller", "Memoli", "Demetci", "OTTER"])
def test_inherited_components_are_attributed_explicitly(attribution: str) -> None:
    assert _fold(attribution) in _fold(RESULTS)


def test_absences_are_phrased_as_search_results() -> None:
    assert "no such work was found" in RESULTS
    assert not re.search(r"\b(no one|nobody|never been done|first to)\b", RESULTS, flags=re.I)


def test_the_band_prior_is_not_called_a_dynamics_model() -> None:
    assert "The band prior is a band prior and not a dynamics model" in RESULTS


def test_no_behavioural_prediction_is_reported_and_marek_is_the_reason() -> None:
    assert has(RESULTS, "No behavioural or cognitive prediction", "Marek et al. 2022", "|r| = 0.01", "N = 3,928")
