"""Tests for the pre-registered pair-selection rules and table helpers (synthetic data only)."""

from __future__ import annotations

import pandas as pd

from src.data_loader import ColumnSplit
from src.exploration import (
    SiteData,
    acquisition_vs_folder_table,
    class_counts,
    duplicate_table,
    group_concentration_table,
    inventory_table,
    pair_candidates,
    target_rows,
)

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


def _with_patients(sd: SiteData, patients: list[str], dates: list[str], stations: list[str]) -> SiteData:
    sd.table = sd.table.assign(patient_no=patients, case_no=["c" + p for p in patients],
                               order_no=["o" + p for p in patients], acquisition_date=dates, workstation=stations)
    return sd


def test_duplicate_table_uses_patient_groups():
    sd = make_site("DRIAMS-A", [("2018", SPECIES, "R", ""), ("2018", SPECIES, "S", ""),
                                ("2018", SPECIES, "S", ""), ("2018", "Staphylococcus aureus", "R", "")])
    sd = _with_patients(sd, ["p1", "p1", "p2", "p3"], ["2018-01-01"] * 4, ["Urine"] * 4)
    row = duplicate_table({"DRIAMS-A": sd}, SPECIES, ["patient_no", "case_no"]).iloc[0]
    assert row["group_column"] == "patient_no"
    assert row["groups"] == 3 and row["groups_with_more_than_one_row"] == 1
    assert row["target_groups"] == 2 and row["target_max_rows_per_group"] == 2
    assert row["target_orders_with_more_than_one_spectrum"] == 1


def test_group_concentration_never_outputs_ids():
    sd = make_site("DRIAMS-A", [("2018", SPECIES, "R", "")] * 3 + [("2018", SPECIES, "S", "")])
    sd = _with_patients(sd, ["secret-id", "secret-id", "secret-id", "other-id"],
                        ["2018-01-01", "2018-02-01", "2018-02-01", "2018-03-01"], ["Stool", "Urine", "Stool", "Blood"])
    table = group_concentration_table({"DRIAMS-A": sd}, SPECIES, ["patient_no"])
    first = table.iloc[0]
    assert first["spectra"] == 3 and first["distinct_acquisition_days"] == 2 and first["distinct_workstations"] == 2
    assert "secret-id" not in table.to_csv()


def test_acquisition_year_vs_folder():
    sd = make_site("DRIAMS-A", [("2018", SPECIES, "R", ""), ("2018", SPECIES, "S", "")])
    sd = _with_patients(sd, ["p1", "p2"], ["2017-12-30", "2018-01-02"], ["Urine", "Urine"])
    table = acquisition_vs_folder_table({"DRIAMS-A": sd})
    assert set(zip(table["acquisition_year"], table["rows"], strict=True)) == {("2017", 1), ("2018", 1)}
    assert acquisition_vs_folder_table({"DRIAMS-B": make_site("DRIAMS-B", EXTERNAL_OK)}).empty


def test_pair_candidates_reports_patient_counts():
    sd = make_site("DRIAMS-A", dev_rows(6, 6, 6, 6))
    n = len(sd.table)
    sd = _with_patients(sd, [f"p{i % 3}" for i in range(n)], ["2018-01-01"] * n, ["Urine"] * n)
    cfg = config() | {"driams": {"group_columns": ["patient_no"]}}
    table, _ = pair_candidates({"DRIAMS-A": sd, "DRIAMS-B": make_site("DRIAMS-B", EXTERNAL_OK)}, cfg)
    cip = table.set_index("antibiotic").loc["Ciprofloxacin"]
    assert cip["dev_groups_class1"] == 3 and cip["dev_groups_class0"] == 3


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
