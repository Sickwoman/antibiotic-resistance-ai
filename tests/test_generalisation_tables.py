"""Tests for the Version 0.7 table generator (no data, no model: string handling only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generalisation_tables import refuse_missing_cells, seed_section, verdict  # noqa: E402

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


# ------------------------------------------------------------------------------------------------
# Issue #22: the seed-variation table must aggregate exactly the rows it says it does
# ------------------------------------------------------------------------------------------------

def _seed_rows(split: str, site: str, model: str, seeds: list[int], aurocs: list[float]) -> list[dict]:
    return [{"split": split, "site": site, "model": model, "seed": s, "roc_auc": a}
            for s, a in zip(seeds, aurocs, strict=True)]


def _parse(table: str) -> list[dict[str, str]]:
    lines = [ln for ln in table.splitlines() if ln.startswith("|")]
    if not lines:                                   # md_table([]) renders nothing at all
        return []
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    return [dict(zip(header, (c.strip() for c in ln.strip("|").split("|")), strict=True)) for ln in lines[2:]]


def test_a_saved_model_row_is_not_counted_as_a_seed():
    """The defect in #22, reproduced with the real V0.7 DRIAMS-B values.

    The five refit seeds and the saved-model reference share split and site, and the saved model's seed (42)
    is the same as the first refit seed. Grouping by split and site alone put all six rows in one group; the
    distinct-seed count still read 5 because 42 appears twice, while the mean, lowest and highest were taken
    over six rows. So the table printed "Seeds 5" beside statistics computed on six.
    """
    refit = [0.814660, 0.823685, 0.815981, 0.809597, 0.823465]
    test = pd.DataFrame(
        _seed_rows("external", "DRIAMS-B", "tuned_lightgbm", [42, 43, 44, 45, 46], refit)
        + _seed_rows("external", "DRIAMS-B", "v0.4_tuned_lightgbm", [42], [0.808607])   # the saved model
    )
    rows = _parse(seed_section(test))
    assert len(rows) == 1, f"expected one seed-variation row, got {rows}"
    row = rows[0]
    assert row["Seeds"] == "5"
    assert row["Mean AUROC"] == f"{sum(refit) / 5:.3f}" == "0.817", "the mean must cover the five seeds only"
    assert row["Lowest"] == f"{min(refit):.3f}" == "0.810", "the saved model's 0.809 must not become the minimum"
    assert row["Highest"] == f"{max(refit):.3f}"


def test_the_printed_seed_count_always_equals_the_rows_aggregated():
    """The general property behind #22: a seed that appears twice in one group is a data error, not a seed.

    Raising is deliberate. The generator already refuses a table with a missing cell rather than print a
    wrong number, and a duplicated seed is the same kind of fault: it would make the count and the statistics
    describe different sets of rows.
    """
    test = pd.DataFrame(_seed_rows("temporal", "DRIAMS-A", "tuned_lightgbm", [42, 42, 43], [0.70, 0.71, 0.72]))
    with pytest.raises(ConfigError, match="appears more than once"):
        seed_section(test)


def test_a_single_seed_group_is_still_left_out():
    """Unchanged behaviour: one seed has no spread, so it gets no row - which is also where the saved model,
    now in a group of its own, ends up."""
    test = pd.DataFrame(_seed_rows("external", "DRIAMS-D", "v0.4_tuned_lightgbm", [42], [0.704142]))
    assert _parse(seed_section(test)) == []


def test_groups_without_a_saved_model_are_unchanged():
    """temporal and external_ab never had a saved-model row, and #22 found they already agreed everywhere."""
    temporal = [0.728418, 0.735, 0.727, 0.740, 0.733]
    test = pd.DataFrame(_seed_rows("temporal", "DRIAMS-A", "tuned_lightgbm", [42, 43, 44, 45, 46], temporal))
    row = _parse(seed_section(test))[0]
    assert row["Seeds"] == "5" and row["Mean AUROC"] == f"{sum(temporal) / 5:.3f}"

