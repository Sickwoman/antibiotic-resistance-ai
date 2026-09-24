"""Tests for the Version 0.7 table generator (no data, no model: string handling only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generalisation_tables import refuse_missing_cells, verdict  # noqa: E402

from src.utils import ConfigError  # noqa: E402


def test_a_gap_counts_only_when_its_interval_excludes_zero():
    """The pre-registered reading: an interval covering 0 is 'not demonstrated', never 'no difference'."""
    assert verdict(0.01, 0.09) == "shown"          # wholly above 0
    assert verdict(-0.09, -0.01) == "shown"        # wholly below 0
    assert verdict(-0.05, 0.09) == "not demonstrated"
    assert verdict(0.0, 0.09) == "not demonstrated"       # touching 0 is not excluding it
    assert verdict(-0.09, 0.0) == "not demonstrated"


def test_a_table_with_a_missing_cell_is_refused():
    """A raw `nan` means a value is absent from the reports, not that it is undefined.

    This is the guard that caught a real defect: the saved-model rows carried no training sites, so the
    published table printed "nan" where a site name belongs.
    """
    good = "| Experiment | Trained on |\n|---|---|\n| external | DRIAMS-A |\n"
    assert refuse_missing_cells(good) == good

    for missing in ("nan", "NaN", "None", "<NA>"):
        bad = f"| Experiment | Trained on |\n|---|---|\n| external | {missing} |\n"
        with pytest.raises(ConfigError, match="missing cell"):
            refuse_missing_cells(bad)


def test_a_dash_is_an_allowed_value():
    """`number()` renders a genuinely undefined number as a dash, which must stay publishable."""
    text = "| Comparison | Spearman |\n|---|---|\n| reported regions | – |\n"
    assert refuse_missing_cells(text) == text


def test_prose_containing_the_word_is_not_mistaken_for_a_cell():
    """Only table rows are inspected: a sentence is not a row, even if it mentions the word."""
    text = "A model fitted on nan would be nonsense.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    assert refuse_missing_cells(text) == text
