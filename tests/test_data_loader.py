"""Tests for src.data_loader using small synthetic files (no real patient data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    MetadataError,
    SpectrumFormatError,
    discover_sites,
    discover_years,
    encode_labels,
    label_counts,
    load_id_table,
    load_site_tables,
    read_binned_spectrum,
    read_raw_spectrum,
    read_spectrum_table,
    split_metadata_antibiotic_columns,
)

AMBIGUOUS = ["S(2)", "R(1)", "R(2)", "L(1)", "I(1)", "I(1), S(1)", "R(1), I(1)", "R(1), S(1)", "R(1), I(1), S(1)"]
METADATA = ["id", "code", "species", "laboratory_species", "case_no", "acquisition_date", "workstation"]

ID_CSV = (
    "code,species,laboratory_species,Ciprofloxacin,Ceftriaxone,Notes\n"
    "c1,Escherichia coli,Escherichia coli,R,S,x\n"
    "c2,Escherichia coli,Escherichia coli,S,-,y\n"
    'c3,Escherichia coli,Escherichia coli,I,"R(1), S(1)",z\n'
    "c4,Klebsiella pneumoniae,Klebsiella pneumoniae,-,R,\n"
)

RAW_SPECTRUM = (
    "#  /some/instrument/path/fid\n"
    "#  c1\n"
    '"mass.myspec..1..." "intensity.myspec..1..."\n'
    + "".join(f"{2000 + i * 0.5} {100 + i}\n" for i in range(200))
)


@pytest.fixture()
def driams_root(tmp_path):
    root = tmp_path / "DRIAMS"
    id_dir = root / "DRIAMS-Z" / "id" / "2018"
    id_dir.mkdir(parents=True)
    (id_dir / "2018_clean.csv").write_text(ID_CSV, encoding="utf-8")
    (root / "DRIAMS-Z" / "raw" / "2018").mkdir(parents=True)
    (root / "not-a-site").mkdir()
    return root


# ---------------------------------------------------------------- discovery / metadata

def test_discovery(driams_root):
    assert discover_sites(driams_root) == ["DRIAMS-Z"]
    assert discover_years(driams_root, "DRIAMS-Z") == ["2018"]
    assert discover_sites(driams_root / "missing") == []


def test_load_id_table_marks_dash_as_missing(driams_root):
    df = load_id_table(driams_root, "DRIAMS-Z", "2018")
    assert len(df) == 4
    assert pd.isna(df.loc[1, "Ceftriaxone"])
    assert pd.isna(df.loc[3, "Ciprofloxacin"])
    assert set(df["driams_site"]) == {"DRIAMS-Z"}
    assert set(df["driams_year"]) == {"2018"}


def test_load_site_tables_concatenates_years(driams_root):
    extra = driams_root / "DRIAMS-Z" / "id" / "2019"
    extra.mkdir()
    (extra / "2019_clean.csv").write_text("code,species,Ciprofloxacin\nc9,Escherichia coli,S\n", encoding="utf-8")
    df = load_site_tables(driams_root, "DRIAMS-Z")
    assert len(df) == 5
    assert sorted(df["driams_year"].unique()) == ["2018", "2019"]


def test_strat_table_is_preferred_over_clean(driams_root):
    # DRIAMS-A ships both; strat has the same rows plus patient/case/date columns.
    strat = driams_root / "DRIAMS-Z" / "id" / "2018" / "2018_strat.csv"
    strat.write_text("code,species,patient_no,Ciprofloxacin\nc1,Escherichia coli,p1,R\n", encoding="utf-8")
    df = load_id_table(driams_root, "DRIAMS-Z", "2018")
    assert "patient_no" in df.columns and len(df) == 1
    assert set(df["driams_id_file"]) == {"2018_strat.csv"}
    clean_only = load_id_table(driams_root, "DRIAMS-Z", "2018", suffixes="clean")
    assert set(clean_only["driams_id_file"]) == {"2018_clean.csv"} and len(clean_only) == 4


def test_index_artifact_columns_are_dropped(driams_root):
    path = driams_root / "DRIAMS-Z" / "id" / "2018" / "2018_clean.csv"
    path.write_text("Unnamed: 0.1,Unnamed: 0,code,species,Ciprofloxacin\n0,0,c1,Escherichia coli,S\n", encoding="utf-8")
    df = load_id_table(driams_root, "DRIAMS-Z", "2018")
    assert not any(c.startswith("Unnamed") for c in df.columns)
    assert df.attrs["dropped_columns"] == ["Unnamed: 0.1", "Unnamed: 0"]
    site = load_site_tables(driams_root, "DRIAMS-Z")
    assert site.attrs["dropped_columns"] == {"DRIAMS-Z/2018": ["Unnamed: 0.1", "Unnamed: 0"]}


def test_missing_metadata_file_raises(driams_root):
    with pytest.raises(MetadataError, match="not found"):
        load_id_table(driams_root, "DRIAMS-Z", "2017")


def test_empty_metadata_file_raises(driams_root):
    path = driams_root / "DRIAMS-Z" / "id" / "2018" / "2018_clean.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(MetadataError, match="empty"):
        load_id_table(driams_root, "DRIAMS-Z", "2018")


def test_header_only_metadata_file_raises(driams_root):
    path = driams_root / "DRIAMS-Z" / "id" / "2018" / "2018_clean.csv"
    path.write_text("code,species\n", encoding="utf-8")
    with pytest.raises(MetadataError, match="no rows"):
        load_id_table(driams_root, "DRIAMS-Z", "2018")


def test_column_split_is_data_driven(driams_root):
    df = load_id_table(driams_root, "DRIAMS-Z", "2018")
    split = split_metadata_antibiotic_columns(df, METADATA, AMBIGUOUS)
    assert split.antibiotics == ["Ciprofloxacin", "Ceftriaxone"]
    assert split.unknown == ["Notes"]            # free text is never treated as a label
    assert "code" in split.metadata and "driams_year" in split.metadata


def test_column_split_reports_empty_columns():
    df = pd.DataFrame({"code": ["a", "b"], "Colistin": [np.nan, np.nan]})
    split = split_metadata_antibiotic_columns(df, METADATA, AMBIGUOUS)
    assert split.empty == ["Colistin"]


def test_column_split_separates_binary_markers():
    # DRIAMS-B stores ESBL / MRSA screens as 0/1 rather than R/I/S.
    df = pd.DataFrame({"code": ["a", "b", "c"], "ESBL": ["1", np.nan, "0"], "Ampicillin": ["R", "S", np.nan]})
    split = split_metadata_antibiotic_columns(df, METADATA, AMBIGUOUS)
    assert split.binary_markers == ["ESBL"]
    assert split.antibiotics == ["Ampicillin"]
    assert split.unknown == []


# ---------------------------------------------------------------- labels

def test_label_counts():
    s = pd.Series(["R", "S", "I", "R(1), S(1)", None, "S", "weird"])
    assert label_counts(s, AMBIGUOUS) == {"R": 1, "I": 1, "S": 2, "ambiguous": 1, "missing": 1, "other": 1}


@pytest.mark.parametrize("mode, expected_i", [("resistant", 1.0), ("susceptible", 0.0), ("exclude", np.nan)])
def test_encode_labels_intermediate_modes(mode, expected_i):
    s = pd.Series(["R", "S", "I", "R(1)", None], name="Ciprofloxacin")
    out = encode_labels(s, mode, AMBIGUOUS)
    assert out.iloc[0] == 1.0 and out.iloc[1] == 0.0
    assert (np.isnan(out.iloc[2]) and np.isnan(expected_i)) or out.iloc[2] == expected_i
    assert np.isnan(out.iloc[3]) and np.isnan(out.iloc[4])
    assert out.dtype == np.float64


def test_encode_labels_rejects_unknown_values():
    with pytest.raises(MetadataError, match="unrecognised"):
        encode_labels(pd.Series(["R", "resistant"], name="X"), "resistant", AMBIGUOUS)


def test_encode_labels_rejects_bad_mode():
    with pytest.raises(ValueError):
        encode_labels(pd.Series(["R"]), "maybe", AMBIGUOUS)


# ---------------------------------------------------------------- spectra

def test_read_raw_spectrum(tmp_path):
    path = tmp_path / "c1.txt"
    path.write_text(RAW_SPECTRUM, encoding="utf-8")
    values = read_raw_spectrum(path)
    assert values.shape == (200, 2)
    assert values[0, 0] == pytest.approx(2000.0)
    assert values[-1, 1] == pytest.approx(299.0)


def test_empty_spectrum_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(SpectrumFormatError, match="empty"):
        read_spectrum_table(path)


def test_comment_only_spectrum_file(tmp_path):
    path = tmp_path / "comments.txt"
    path.write_text("# nothing here\n# still nothing\n", encoding="utf-8")
    with pytest.raises(SpectrumFormatError):
        read_spectrum_table(path)


def test_wrong_extension(tmp_path):
    path = tmp_path / "spectrum.exe"
    path.write_bytes(b"MZ\x90\x00")
    with pytest.raises(SpectrumFormatError, match="Unsupported file type"):
        read_spectrum_table(path)


def test_missing_spectrum_file(tmp_path):
    with pytest.raises(SpectrumFormatError, match="not found"):
        read_spectrum_table(tmp_path / "nope.txt")


def test_corrupted_binary_spectrum(tmp_path):
    path = tmp_path / "corrupt.txt"
    path.write_bytes(bytes(range(256)) * 20)
    with pytest.raises(SpectrumFormatError):
        read_spectrum_table(path)


def test_non_numeric_rows(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("mz intensity\n2000 10\n2001 abc\n", encoding="utf-8")
    with pytest.raises(SpectrumFormatError, match="non-numeric"):
        read_spectrum_table(path)


def test_wrong_number_of_columns(tmp_path):
    path = tmp_path / "three.txt"
    path.write_text("1 2 3\n4 5 6\n", encoding="utf-8")
    with pytest.raises(SpectrumFormatError, match="expected 2 columns"):
        read_spectrum_table(path)


def test_missing_value_in_spectrum(tmp_path):
    path = tmp_path / "nan.txt"
    path.write_text("2000 10\n2001 nan\n", encoding="utf-8")
    with pytest.raises(SpectrumFormatError):
        read_spectrum_table(path)


def _binned_text(n: int, start: int = 0) -> str:
    return "bin_index binned_intensity\n" + "".join(f"{start + i} {i * 1e-6}\n" for i in range(n))


def test_read_binned_spectrum(tmp_path):
    path = tmp_path / "b.txt"
    path.write_text(_binned_text(6000), encoding="utf-8")
    vec = read_binned_spectrum(path)
    assert vec.shape == (6000,)
    assert vec[1] == pytest.approx(1e-6)


def test_binned_wrong_number_of_features(tmp_path):
    path = tmp_path / "short.txt"
    path.write_text(_binned_text(5999), encoding="utf-8")
    with pytest.raises(SpectrumFormatError, match="expected 6000 bins"):
        read_binned_spectrum(path)


def test_binned_wrong_indices(tmp_path):
    path = tmp_path / "shifted.txt"
    path.write_text(_binned_text(6000, start=1), encoding="utf-8")
    with pytest.raises(SpectrumFormatError, match="bin indices"):
        read_binned_spectrum(path)


def test_raw_spectrum_validation(tmp_path):
    too_short = tmp_path / "short.txt"
    too_short.write_text("2000 1\n2001 2\n", encoding="utf-8")
    with pytest.raises(SpectrumFormatError, match="only 2 points"):
        read_raw_spectrum(too_short)

    unsorted = tmp_path / "unsorted.txt"
    unsorted.write_text("".join(f"{3000 - i} 5\n" for i in range(150)), encoding="utf-8")
    with pytest.raises(SpectrumFormatError, match="strictly increasing"):
        read_raw_spectrum(unsorted)

    negative = tmp_path / "negative.txt"
    negative.write_text("".join(f"{2000 + i} {-1 if i == 5 else 3}\n" for i in range(150)), encoding="utf-8")
    with pytest.raises(SpectrumFormatError, match="negative"):
        read_raw_spectrum(negative)
