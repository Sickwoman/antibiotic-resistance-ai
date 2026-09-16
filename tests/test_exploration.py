"""Tests for the pre-registered pair-selection rules and table helpers (synthetic data only)."""

from __future__ import annotations

import pandas as pd

from src.data_loader import ColumnSplit
from src.exploration import SiteData, class_counts, inventory_table, pair_candidates, target_rows

SPECIES = "Escherichia coli"
ABX = ["Ciprofloxacin", "Ceftriaxone"]


def make_site(site: str, rows: list[tuple[str, str, str, str]], binned: bool = True) -> SiteData:
    """rows = (year, species, ciprofloxacin_label, ceftriaxone_label); '' means missing."""
    records = []
    for i, (year, species, cip, cro) in enumerate(rows):
        records.append({"code": f"{site}-{i}", "species": species, "Ciprofloxacin": cip or None,
                        "Ceftriaxone": cro or None, "driams_site": site, "driams_year": year})
    table = pd.DataFrame(records)
    split = ColumnSplit(metadata=["code", "species", "driams_site", "driams_year"], antibiotics=list(ABX))
    manifest = None
    if binned:
        manifest = pd.DataFrame({"folder": "binned_6000", "year": table["driams_year"],
                                 "filename": table["code"] + ".txt", "size_bytes": 1, "code": table["code"]})
    return SiteData(site, table, split, manifest)


def config(**overrides) -> dict:
    ps = {"development_site": "DRIAMS-A", "min_resistant_dev": 4, "min_susceptible_dev": 4,
          "min_minority_fraction": 0.1, "min_years_with_labels": 2, "temporal_test_year": "2018",
          "min_resistant_test_year": 1, "external_sites": ["DRIAMS-B", "DRIAMS-C"],
          "min_per_class_external": 2, "min_external_sites": 1}
    ps.update(overrides)
    return {"pair_selection": ps, "labels": {"intermediate_as": "resistant"},
            "target": {"species": SPECIES, "preferred_antibiotic": "Ciprofloxacin"}}


def dev_rows(cip_r: int, cip_s: int, cro_r: int, cro_s: int) -> list[tuple[str, str, str, str]]:
    rows = []
    for k in range(max(cip_r + cip_s, cro_r + cro_s)):
        year = "2017" if k % 2 == 0 else "2018"
        cip = "R" if k < cip_r else ("S" if k < cip_r + cip_s else "")
        cro = "R" if k < cro_r else ("S" if k < cro_r + cro_s else "")
        rows.append((year, SPECIES, cip, cro))
    rows.append(("2018", "Staphylococcus aureus", "R", "R"))  # other species must be ignored
    return rows


EXTERNAL_OK = [("2018", SPECIES, "R", "R"), ("2018", SPECIES, "R", "R"),
               ("2018", SPECIES, "S", "S"), ("2018", SPECIES, "S", "S")]


def test_class_counts_intermediate_handling():
    s = pd.Series(["R", "I", "S", None])
    assert class_counts(s, "resistant") == {"R": 1, "I": 1, "S": 1, "class1": 2, "class0": 1}
    assert class_counts(s, "susceptible")["class0"] == 2
    assert class_counts(s, "exclude")["class1"] == 1


def test_preferred_antibiotic_selected_when_eligible():
    sites = {"DRIAMS-A": make_site("DRIAMS-A", dev_rows(6, 6, 6, 6)),
             "DRIAMS-B": make_site("DRIAMS-B", EXTERNAL_OK),
             "DRIAMS-C": make_site("DRIAMS-C", EXTERNAL_OK)}
    table, decision = pair_candidates(sites, config())
    assert decision["selected_antibiotic"] == "Ciprofloxacin"
    assert decision["preferred_status"] == "eligible"
    cip = table.set_index("antibiotic").loc["Ciprofloxacin"]
    assert cip["dev_class1"] == 6 and cip["dev_class0"] == 6     # S. aureus row not counted
    assert cip["dev_years_with_both_classes"] == 2


def test_fallback_when_preferred_fails():
    sites = {"DRIAMS-A": make_site("DRIAMS-A", dev_rows(2, 10, 6, 6)),
             "DRIAMS-B": make_site("DRIAMS-B", EXTERNAL_OK)}
    table, decision = pair_candidates(sites, config())
    assert decision["preferred_status"] == "not eligible"
    assert "resistant 2 < 4" in decision["preferred_reasons"]
    assert decision["selected_antibiotic"] == "Ceftriaxone"


def test_pending_when_external_sites_missing():
    sites = {"DRIAMS-A": make_site("DRIAMS-A", dev_rows(6, 6, 6, 6))}
    _, decision = pair_candidates(sites, config())
    assert decision["preferred_status"] == "pending"
    assert decision["selected_antibiotic"] == "Ciprofloxacin"
    assert "DRIAMS-B" in decision["preferred_reasons"]


def test_no_decision_without_development_site():
    sites = {"DRIAMS-B": make_site("DRIAMS-B", EXTERNAL_OK)}
    table, decision = pair_candidates(sites, config())
    assert decision["selected_antibiotic"] is None
    assert set(table["status"]) == {"pending"}


def test_external_site_too_small_is_not_eligible():
    small = [("2018", SPECIES, "R", "R"), ("2018", SPECIES, "S", "S")]
    sites = {"DRIAMS-A": make_site("DRIAMS-A", dev_rows(6, 6, 6, 6)),
             "DRIAMS-B": make_site("DRIAMS-B", small),
             "DRIAMS-C": make_site("DRIAMS-C", small)}
    _, decision = pair_candidates(sites, config())
    assert decision["preferred_status"] == "not eligible"
    assert decision["selected_antibiotic"] is None


def test_target_rows_requires_binned_spectrum_and_dedups_codes():
    sd = make_site("DRIAMS-A", [("2018", SPECIES, "R", ""), ("2018", SPECIES, "S", "")])
    sd.manifest = sd.manifest.iloc[:1]                       # only the first code has a spectrum
    assert list(target_rows(sd, SPECIES)["code"]) == ["DRIAMS-A-0"]
    dup = pd.concat([sd.table, sd.table.iloc[:1]], ignore_index=True)
    sd.table = dup
    assert len(target_rows(sd, SPECIES, require_binned=False)) == 2


def test_inventory_counts_files_without_metadata():
    sd = make_site("DRIAMS-A", [("2018", SPECIES, "R", "")])
    extra = pd.DataFrame({"folder": ["binned_6000"], "year": ["2018"], "filename": ["orphan.txt"],
                          "size_bytes": [1], "code": ["orphan"]})
    sd.manifest = pd.concat([sd.manifest, extra], ignore_index=True)
    inv = inventory_table({"DRIAMS-A": sd}, ["binned_6000"])
    row = inv.iloc[0]
    assert row["id_rows"] == 1 and row["binned_6000_files"] == 2
    assert row["binned_6000_files_without_id_row"] == 1
    assert row["labelled_rows_without_binned"] == 0
