"""Replicate selection, exclusion counting and the 5 % pause (amendment A6), on synthetic structures only."""

from __future__ import annotations

import pytest

from src.bruker import BrukerReadError
from src.marisma_cohort import (
    NO_PASSING_REPLICATE,
    NO_REPLICATE_FOLDER,
    SpectrumFolder,
    exclusion_summary,
    folder_order,
    parse_spectrum_member,
    pause_check,
    select_first_passing,
)


def test_integer_names_are_ordered_numerically():
    assert folder_order(["10", "2", "1"]) == ["1", "2", "10"]


def test_mixed_names_are_ordered_by_code_point():
    assert folder_order(["0_A2", "0_A10", "0_A1", "1"]) == ["0_A1", "0_A10", "0_A2", "1"]
    assert folder_order(["b", "B", "a"]) == ["B", "a", "b"]          # code point: upper case first


def test_non_ascii_digits_do_not_count_as_integers():
    assert folder_order(["２", "1"]) == ["1", "２"]                    # full-width two: code point order


def reader(outcomes):
    """A stand-in reader: outcomes maps a replicate path to a spectrum value or to a failure reason."""
    def read(path):
        value = outcomes[path]
        if value in {"empty", "no_calibration", "range_not_covered"}:
            raise BrukerReadError(value)
        return value
    return read


def test_the_first_passing_replicate_in_registered_order_is_selected():
    replicates = {"2": {"1": "b2t1"}, "10": {"1": "b10t1"}, "1": {"2": "b1t2", "1": "b1t1", "10": "b1t10"}}
    outcomes = {"b1t1": "empty", "b1t2": "no_calibration", "b1t10": "S", "b2t1": "T", "b10t1": "U"}
    s = select_first_passing(replicates, reader(outcomes))
    assert (s.status, s.spectrum, s.biological, s.technical) == ("selected", "S", "1", "10")
    assert s.failures == ("empty", "no_calibration")


def test_later_biological_replicates_are_tried_when_the_first_has_none_passing():
    replicates = {"1": {"1": "x"}, "2": {"1": "y"}}
    s = select_first_passing(replicates, reader({"x": "empty", "y": "Y"}))
    assert (s.status, s.spectrum, s.biological) == ("selected", "Y", "2")


def test_an_isolate_without_a_passing_replicate_is_excluded_with_its_reasons():
    s = select_first_passing({"1": {"1": "x", "2": "y"}}, reader({"x": "empty", "y": "range_not_covered"}))
    assert (s.status, s.spectrum, s.failures) == (NO_PASSING_REPLICATE, None, ("empty", "range_not_covered"))


@pytest.mark.parametrize("replicates", [{}, {"1": {}}])
def test_an_isolate_without_replicate_folders_is_counted_separately(replicates):
    s = select_first_passing(replicates, reader({}))
    assert s.status == NO_REPLICATE_FOLDER and s.failures == ()


def test_selection_is_deterministic_whatever_the_insertion_order():
    a = {"1": {"1": "x", "2": "y"}, "0": {"5": "z"}}
    b = {"0": {"5": "z"}, "1": {"2": "y", "1": "x"}}
    read = reader({"x": "X", "y": "Y", "z": "empty"})
    assert select_first_passing(a, read) == select_first_passing(b, read)


def test_summary_and_pause_threshold():
    sel = ([select_first_passing({"1": {"1": "ok"}}, reader({"ok": "S"}))] * 19
           + [select_first_passing({"1": {"1": "bad"}}, reader({"bad": "empty"}))]
           + [select_first_passing({}, reader({}))] * 3)
    summary = exclusion_summary(sel)
    assert summary["by_status"] == {"no_passing_replicate": 1, "no_replicate_folder": 3, "selected": 19}
    assert summary["reasons_among_excluded_isolates"] == {"empty": 1}
    p = pause_check(sel)
    assert (p["numerator"], p["denominator"]) == (1, 20)       # isolates without folders are not in the denominator
    assert p["share"] == pytest.approx(0.05) and p["pause"] is False   # exactly 5 % does not exceed 5 %
    sel2 = sel + [select_first_passing({"1": {"1": "bad"}}, reader({"bad": "empty"}))]
    assert pause_check(sel2)["pause"] is True                  # 2 / 21 > 5 %


def test_a_fid_member_is_placed_in_the_layout():
    name = "MARISMa/2021/Escherichia/Coli/ab12cd34/0_A10/1/1SLin/fid"
    assert parse_spectrum_member(name) == SpectrumFolder(
        "2021", "Escherichia", "Coli", "ab12cd34", "0_A10", "1", "MARISMa/2021/Escherichia/Coli/ab12cd34/0_A10/1/1SLin")


@pytest.mark.parametrize("name", [
    "MARISMa/2021/Escherichia/Coli/ab12cd34/0_A10/1/1SLin/fid (2)",     # a copy, not a spectrum
    "MARISMa/2021/Escherichia/Coli/ab12cd34/0_A10/1/1SLin/acqu",
    "MARISMa/2021/Escherichia/Coli/ab12cd34/0_A10/1/1Ref/fid",          # not the linear spectrum folder
    "MARISMa/2021/Escherichia/Coli/ab12cd34/0_A10/1SLin/fid",           # a level missing
    "Other/2021/Escherichia/Coli/ab12cd34/0_A10/1/1SLin/fid",
    "MARISMa/2021/Escherichia//ab12cd34/0_A10/1/1SLin/fid",
])
def test_members_outside_the_layout_are_not_spectra(name):
    assert parse_spectrum_member(name) is None


def test_layout_folders_come_from_directory_components_only():
    from src.marisma_cohort import layout_folders
    names = ["MARISMa/2021/Escherichia/Coli/iso1/0_A1/1/1SLin/fid",
             "MARISMa/2021/Escherichia/Coli/iso1/0_A1/1/1SLin/acqu",
             "MARISMa/2021/Escherichia/Coli/iso2/0_B3/1/",            # a replicate folder without a spectrum
             "MARISMa/2021/Escherichia/Coli/iso3/",                   # an isolate folder without replicates
             "MARISMa/2021/Escherichia/Coli/iso3/.DS_Store",          # a file, not a replicate folder
             "MARISMa/2021/Escherichia/Coli/.DS_Store"]
    isolates, replicates = layout_folders(names)
    assert isolates == {("2021", "Escherichia", "Coli", i) for i in ("iso1", "iso2", "iso3")}
    assert {(r.isolate, r.biological, r.technical, r.folder.rsplit("/", 1)[-1]) for r in replicates} == {
        ("iso1", "0_A1", "1", "1SLin"), ("iso2", "0_B3", "1", "1SLin")}
