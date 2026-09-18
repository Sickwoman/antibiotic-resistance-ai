"""Tests for the Version 0.2 dataset builder, using a small synthetic DRIAMS folder."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataError
from src.dataset import (
    CohortSpec,
    DatasetError,
    assign_group_ids,
    build_dataset,
    load_dataset,
    resolve_relpath,
    spectrum_relpath,
)
from src.preprocessing import PreprocessingConfig, preprocess_file
from src.splits import SplitError, build_splits
from src.utils import load_config
from tests.test_preprocessing import synthetic_spectrum, write_spectrum


def pid(label: str) -> str:
    """32-character hex patient hash, like DRIAMS-A patient_no."""
    return hashlib.md5(label.encode()).hexdigest()


P1, P2, P3, P4 = pid("patient-1"), pid("patient-2"), pid("patient-3"), pid("patient-4")

# code, species, ciprofloxacin, workstation, patient, date, spectrum kind
SITE_Y_2017 = [
    ("y01", "Escherichia coli", "R", "Urine", P1, "2017-03-01", "ok"),
    ("y02", "Escherichia coli", "S", "Blood", P1, "2017-03-05", "ok"),
    ("y03", "Escherichia coli", "I", "Urine", P2, "2017-05-01", "ok"),
    ("y04", "Escherichia coli", "-", "Urine", P2, "2017-05-02", "ok"),
    ("y05", "Escherichia coli", "R(1), S(1)", "Urine", P2, "2017-05-03", "ok"),
    ("y06", "Escherichia coli", "X", "Urine", P2, "2017-05-04", "ok"),
    ("y07", "Escherichia coli", "S", "HospitalHygiene", P4, "2017-06-01", "ok"),
    ("y08", "Escherichia coli", "S", "Urine", P4, "2017-06-02", "missing"),
    ("y09", "Escherichia coli", "R", "Urine", P4, "2017-06-03", "garbage"),
    ("y10", "Escherichia coli", "S", "Urine", P4, "2017-06-04", "zeros"),
    ("y11", "Escherichia coli", "S", "Urine", P4, "2017-06-05", "copy_of_y02"),
    ("y12", "Klebsiella pneumoniae", "R", "Urine", P4, "2017-06-06", "ok"),
    ("y13", "MIX!Escherichia coli", "R", "Urine", P4, "2017-06-07", "ok"),
    ("y14", "Escherichia coli", "S", "Varia", P4, "2017-07-01", "ok"),
    ("y18", "Escherichia coli", "S", "Urine", P4, "2017-07-02", "ok"),        # published binned file disagrees
    ("y19", "Escherichia coli", "R", "Urine", P4, "2017-07-03", "ok"),        # repeated with another label
    ("y20", "Escherichia coli", "-", "Urine", P4, "2017-07-04", "ok"),        # repeated; only 2018 copy labelled
]
SITE_Y_2018 = [
    ("y01", "Escherichia coli", "R", "Urine", P3, "2018-01-10", "ok"),       # repeated code, same label
    ("y15", "Escherichia coli", "R", "Urine", P3, "2017-12-30", "ok"),       # 2018 folder, acquired in 2017
    ("y16", "Escherichia coli", "S", "Blood", P3, "2018-02-01", "ok"),
    ("y17", "Escherichia coli", "S", "Urine", P1, "2018-03-01", "ok"),       # same hash as 2017, other year
    ("y19", "Escherichia coli", "S", "Urine", P3, "2018-03-02", "ok"),
    ("y20", "Escherichia coli", "S", "Urine", P3, "2018-03-03", "ok"),
]
SITE_Z = [("z01", "Escherichia coli", "R"), ("z02", "Escherichia coli", "S"), ("z03", "Escherichia coli", "S")]
PATIENT_HASHES = {P1, P2, P3, P4}


def _seed(year: str, index: int) -> int:
    return int(year) * 100 + index            # unique per row, so only the intended copy is a duplicate


Y02_SEED = _seed("2017", 1)


def _write_raw(folder: Path, code: str, kind: str, seed: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if kind == "missing":
        return
    if kind == "garbage":
        (folder / f"{code}.txt").write_bytes(bytes(range(256)) * 10)
        return
    mz, inten = synthetic_spectrum(n=1500, seed=Y02_SEED if kind == "copy_of_y02" else seed)
    if kind == "zeros":
        inten = np.zeros_like(inten)
    write_spectrum(folder / f"{code}.txt", mz, inten)


@pytest.fixture()
def driams(tmp_path):
    root = tmp_path / "DRIAMS"
    for year, rows in (("2017", SITE_Y_2017), ("2018", SITE_Y_2018)):
        id_dir = root / "DRIAMS-Y" / "id" / year
        id_dir.mkdir(parents=True)
        pd.DataFrame([{"code": c, "species": s, "laboratory_species": s, "Ciprofloxacin": cip,
                       "acquisition_date": d, "acquisition_time": "10:00:00", "workstation": ws,
                       "patient_no": p, "case_no": pid(p + "case"), "order_no": pid(p + "order")}
                      for c, s, cip, ws, p, d, _ in rows]).to_csv(id_dir / f"{year}_strat.csv", index=False)
        for k, (c, _, _, _, _, _, kind) in enumerate(rows):
            if not (year == "2018" and c == "y01"):
                _write_raw(root / "DRIAMS-Y" / "raw" / year, c, kind, seed=_seed(year, k))
    z_id = root / "DRIAMS-Z" / "id" / "2018"
    z_id.mkdir(parents=True)
    pd.DataFrame([{"species": s, "code": c, "combined_code": None, "Ciprofloxacin": cip, "ESBL": "1"}
                  for c, s, cip in SITE_Z]).to_csv(z_id / "2018_clean.csv", index=False)
    for k, (c, _, _) in enumerate(SITE_Z):
        _write_raw(root / "DRIAMS-Z" / "raw" / "2018", c, "ok", seed=_seed("9999", k))

    # published binned_6000 references: correct for y01, wrong for y18, absent for everything else
    binned = root / "DRIAMS-Y" / "binned_6000" / "2017"
    binned.mkdir(parents=True)
    correct, _ = preprocess_file(root / "DRIAMS-Y" / "raw" / "2017" / "y01.txt", PreprocessingConfig(dtype="float64"))
    wrong = np.random.default_rng(5).random(6000)
    for code, values in (("y01", correct), ("y18", wrong)):
        (binned / f"{code}.txt").write_text(
            "bin_index binned_intensity\n" + "".join(f"{j} {float(v)!r}\n" for j, v in enumerate(values)),
            encoding="utf-8")

    config = copy.deepcopy(load_config())
    config["paths"]["driams_root"] = str(root)
    config["dataset"]["output_dir"] = str(tmp_path / "processed")
    config["dataset"]["sites"] = ["DRIAMS-Y", "DRIAMS-Z"]
    config["dataset"]["chunk_size"] = 4
    return config, tmp_path / "processed"


def _build(config, **kwargs):
    spec = CohortSpec.from_config(config, **kwargs)
    summary = build_dataset(config, spec)
    X, meta, saved = load_dataset(Path(config["dataset"]["output_dir"]) / spec.name)
    exclusions = pd.read_csv(Path(config["dataset"]["output_dir"]) / spec.name / "exclusions.csv", dtype=str)
    return summary, X, meta, exclusions


def reasons(exclusions: pd.DataFrame) -> dict[str, str]:
    return dict(zip(exclusions["code"] + "@" + exclusions["year_folder"], exclusions["reason"], strict=True))


def test_primary_dataset_contents(driams):
    config, _ = driams
    summary, X, meta, exclusions = _build(config)
    assert X.dtype == np.float32 and X.shape == (len(meta), 6000)
    assert set(meta["code"] + "@" + meta["year_folder"]) == {
        "y01@2017", "y02@2017", "y03@2017", "y14@2017", "y15@2018", "y16@2018", "y17@2018", "y20@2018",
        "z01@2018", "z02@2018", "z03@2018"}
    labels = dict(zip(meta["code"] + "@" + meta["year_folder"], meta["label"], strict=True))
    assert labels["y01@2017"] == 1 and labels["y02@2017"] == 0 and labels["y03@2017"] == 1   # I -> resistant
    assert reasons(exclusions) == {
        "y01@2018": "duplicate_record", "y04@2017": "no_ast_result", "y05@2017": "ambiguous_ast_result",
        "y06@2017": "unsupported_ast_result", "y07@2017": "excluded_workstation",
        "y08@2017": "no_spectrum_file", "y09@2017": "malformed_spectrum", "y10@2017": "unusable_spectrum",
        "y11@2017": "duplicate_spectrum", "y18@2017": "differs_from_driams_binned",
        "y19@2017": "conflicting_duplicate_record", "y19@2018": "conflicting_duplicate_record",
        "y20@2017": "no_ast_result"}
    # every target-species row is either used or excluded; other species are not counted
    target_rows = sum(v["target_species_rows"] for v in summary["per_site"].values())
    assert target_rows == len(meta) + len(exclusions) == 24
    check = summary["verification_against_driams_binned"]
    assert check["enabled"] and check["verified"] == 1 and check["failed"] == 1
    assert check["no_reference_file"] == 11 and check["max_relative_difference_of_verified"] < 1e-6
    assert summary["resistant"] == int((meta["label"] == 1).sum()) == 4
    assert summary["excluded_workstations_by_site"] == {"DRIAMS-Y": {"HospitalHygiene": 1}}
    assert summary["per_site"]["DRIAMS-Y"]["other_spellings_not_included"] == {"MIX!Escherichia coli": 1}
    assert summary["per_site"]["DRIAMS-Y"]["ast_values"] == {"I": 1, "R": 2, "S": 5}
    assert summary["per_site"]["DRIAMS-Z"]["samples_without_patient_id"] == 3
    assert summary["x_shape"] == [11, 6000]
    assert any(n.startswith("Sites without patient IDs (DRIAMS-Z)") for n in summary["notes"])
    assert any(n.startswith("Sites without a workstation column (DRIAMS-Z)") for n in summary["notes"])


def test_rows_of_x_match_their_spectra(driams):
    config, _ = driams
    from src.preprocessing import PreprocessingConfig, preprocess_file

    _, X, meta, _ = _build(config)
    pcfg = PreprocessingConfig.from_config(config)
    root = config["paths"]["driams_root"]
    for i in (0, 4, len(meta) - 1):
        expected, _ = preprocess_file(resolve_relpath(root, meta.loc[i, "spectrum_relpath"]), pcfg)
        assert np.array_equal(X[i], expected)


def test_acquisition_dates_and_groups(driams):
    config, _ = driams
    _, _, meta, _ = _build(config)
    by_key = meta.set_index(meta["code"] + "@" + meta["year_folder"])
    assert by_key.loc["y15@2018", "acquisition_date"] == "2017-12-30"
    assert by_key.loc["y15@2018", "year_folder"] == "2018"
    assert pd.isna(by_key.loc["z01@2018", "acquisition_date"])
    g = by_key["group_id"]
    assert g["y01@2017"] == g["y02@2017"]                  # same patient, same year
    assert g["y15@2018"] == g["y16@2018"]
    assert g["y17@2018"] != g["y01@2017"]                  # same hash in another year = different group
    assert len({g["z01@2018"], g["z02@2018"], g["z03@2018"]}) == 3
    assert set(by_key.loc[by_key["site"] == "DRIAMS-Z", "group_source"]) == {"sample"}
    assert set(by_key.loc[by_key["site"] == "DRIAMS-Y", "group_source"]) == {"patient_no"}


def test_no_patient_identifiers_in_outputs(driams):
    config, out = driams
    _build(config)
    secrets = PATIENT_HASHES | {pid(p + "case") for p in PATIENT_HASHES} | {pid(p + "order") for p in PATIENT_HASHES}
    for path in (out / "ecoli_ciprofloxacin").iterdir():
        text = path.read_bytes().decode("latin-1")
        for secret in secrets:
            assert secret not in text, f"patient/case/order hash leaked into {path.name}"
    for name in ("metadata.csv", "exclusions.csv"):
        columns = set(pd.read_csv(out / "ecoli_ciprofloxacin" / name, nrows=0).columns)
        assert not columns & {"patient_no", "case_no", "order_no", "group_key"}


def test_intermediate_can_be_excluded(driams):
    config, _ = driams
    summary, _, meta, exclusions = _build(config, intermediate_as="exclude")
    assert summary["dataset"] == "ecoli_ciprofloxacin__intermediate-exclude"
    assert "y03" not in set(meta["code"])
    assert reasons(exclusions)["y03@2017"] == "intermediate_excluded"
    assert set(meta["ast_value"]) == {"R", "S"}


def test_intermediate_as_susceptible(driams):
    config, _ = driams
    _, _, meta, _ = _build(config, intermediate_as="susceptible")
    assert int(meta.loc[meta["code"] == "y03", "label"].iloc[0]) == 0


def test_hospital_hygiene_filter_can_be_disabled(driams):
    config, _ = driams
    config["dataset"]["exclude_workstations"] = []
    _, _, meta, _ = _build(config, name="with_hygiene")
    assert "y07" in set(meta["code"])


def test_verification_is_skipped_for_non_driams_settings(driams):
    config, _ = driams
    config["preprocessing"]["bin_width"] = 6.0
    summary, X, meta, _ = _build(config, name="coarse_bins")
    assert X.shape[1] == 3000
    check = summary["verification_against_driams_binned"]
    assert not check["enabled"] and "differ from DRIAMS" in check["skipped_reason"]
    assert "y18" in set(meta["code"])


def test_verification_can_be_disabled(driams):
    config, _ = driams
    config["dataset"]["verify_against_driams_binned"] = False
    summary, _, meta, _ = _build(config, name="unverified")
    assert summary["verification_against_driams_binned"]["enabled"] is False
    assert summary["verification_against_driams_binned"]["skipped_reason"] is None
    assert "y18" in set(meta["code"])


def test_required_acquisition_date_excludes_undated_sites(driams):
    config, _ = driams
    config["dataset"]["require_acquisition_date"] = True
    _, _, meta, exclusions = _build(config, name="dated_only")
    assert set(meta["site"]) == {"DRIAMS-Y"}
    assert set(exclusions.loc[exclusions["site"] == "DRIAMS-Z", "reason"]) == {"missing_acquisition_date"}


def test_build_is_reproducible(driams):
    config, out = driams
    s1, *_ = _build(config, name="run1")
    s2, *_ = _build(config, name="run2")
    assert (out / "run1" / "X.npy").read_bytes() == (out / "run2" / "X.npy").read_bytes()
    assert (out / "run1" / "metadata.csv").read_bytes() == (out / "run2" / "metadata.csv").read_bytes()
    assert (s1["row_fingerprint"], s1["x_sha256"]) == (s2["row_fingerprint"], s2["x_sha256"])
    assert not (out / "run1.building").exists()


def test_fingerprints_detect_files_changed_after_the_build(driams):
    config, out = driams
    summary = _build(config)[0]                           # keep no reference to the memory-mapped X
    folder = out / "ecoli_ciprofloxacin"
    assert len(summary["row_fingerprint"]) == 16 and len(summary["x_sha256"]) == 64
    load_dataset(folder, verify_x=True)
    gc.collect()                                          # release the memory map before editing X.npy

    x_path = folder / "X.npy"
    data = bytearray(x_path.read_bytes())
    data[-1] ^= 0xFF
    x_path.write_bytes(bytes(data))
    load_dataset(folder)                                  # the quick check does not read X
    with pytest.raises(DatasetError, match="X.npy does not match"):
        load_dataset(folder, verify_x=True)
    gc.collect()

    meta_path = folder / "metadata.csv"
    edited = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
    edited.loc[0, "label"] = "0" if edited.loc[0, "label"] == "1" else "1"   # same shape, other label
    edited.to_csv(meta_path, index=False)
    with pytest.raises(DatasetError, match="fingerprint"):
        load_dataset(folder)

    summary_path = folder / "summary.json"
    old = json.loads(summary_path.read_text(encoding="utf-8"))
    del old["row_fingerprint"]
    summary_path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(DatasetError, match="older version"):
        load_dataset(folder)


def test_sensitivity_dataset_reuses_the_primary_splits(driams):
    config, out = driams
    # the synthetic sites are DRIAMS-Y (dated, patient IDs) and DRIAMS-Z; random/within_year stay on the
    # absent DRIAMS-A and are skipped
    config["splits"]["temporal"].update(sites=["DRIAMS-Y"], validation_start="2017-06-15")
    config["splits"]["external"].update(train_sites=["DRIAMS-Y"], test_sites=["DRIAMS-Z"], validation_fraction=0.5)
    _, _, meta, _ = _build(config)
    _, _, variant, _ = _build(config, intermediate_as="exclude")
    variant_name = "ecoli_ciprofloxacin__intermediate-exclude"
    with pytest.raises(SplitError, match="Build it first"):
        build_splits(variant, config, variant_name, out)

    skipped: list[str] = []
    primary_splits = build_splits(meta, config, "ecoli_ciprofloxacin", out, skipped)
    assert set(primary_splits) == {"temporal", "external"} and len(skipped) == 2
    for name, split in primary_splits.items():
        split.save(out / "ecoli_ciprofloxacin" / "splits" / f"{name}.json", meta)

    derived = build_splits(variant, config, variant_name, out)
    assert list(derived) == list(primary_splits) == ["temporal", "external"]
    assert any(note.startswith("[ecoli_ciprofloxacin] Removed") for note in derived["temporal"].notes)
    key = meta["code"] + "@" + meta["year_folder"]
    variant_key = variant["code"] + "@" + variant["year_folder"]
    for name, split in derived.items():
        assert split.derived_from["dataset"] == "ecoli_ciprofloxacin"
        for part, idx in split.parts().items():
            expected = set(key.iloc[primary_splits[name].parts()[part]]) - {"y03@2017"}   # the I sample
            assert set(variant_key.iloc[idx]) == expected
    assert set(key.iloc[primary_splits["temporal"].train]) == {"y01@2017", "y02@2017", "y03@2017"}
    assert "y03@2017" in set(key.iloc[np.concatenate(list(primary_splits["external"].parts().values()))])


def test_rebuild_replaces_previous_output(driams):
    config, out = driams
    _build(config)
    _build(config)
    assert sorted(p.name for p in (out / "ecoli_ciprofloxacin").iterdir()) == [
        "X.npy", "exclusions.csv", "metadata.csv", "summary.json"]


def test_missing_site_is_an_error(driams):
    config, _ = driams
    with pytest.raises(DatasetError, match="not extracted"):
        build_dataset(config, CohortSpec.from_config(config, sites=["DRIAMS-Y", "DRIAMS-C"]))


def test_non_default_choices_never_reuse_the_primary_name(driams):
    config, _ = driams
    assert CohortSpec.from_config(config).name == "ecoli_ciprofloxacin"
    assert CohortSpec.from_config(config, sites=["DRIAMS-Z"]).name == "ecoli_ciprofloxacin__sites-Z"
    assert (CohortSpec.from_config(config, intermediate_as="exclude", sites=["DRIAMS-Y"]).name
            == "ecoli_ciprofloxacin__intermediate-exclude__sites-Y")
    assert CohortSpec.from_config(config, sites=["DRIAMS-Y", "DRIAMS-Z"]).name == "ecoli_ciprofloxacin"


def test_failed_build_releases_its_files(driams, monkeypatch):
    config, out = driams
    import src.dataset as dataset_module

    def crash(*args, **kwargs):
        raise RuntimeError("simulated crash while processing spectra")

    monkeypatch.setattr(dataset_module, "_process_spectra", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        build_dataset(config, CohortSpec.from_config(config))
    assert (out / "ecoli_ciprofloxacin.building").exists()
    monkeypatch.undo()
    _build(config)                                       # the leftover folder can be removed and rebuilt
    assert not (out / "ecoli_ciprofloxacin.building").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="open files only block deletion on Windows")
def test_locked_leftover_folder_gives_a_clear_error(driams):
    config, out = driams
    leftover = out / "ecoli_ciprofloxacin.building"
    leftover.mkdir(parents=True)
    handle = np.lib.format.open_memmap(leftover / "X_unfiltered.npy", mode="w+", dtype="float32", shape=(2, 2))
    try:
        with pytest.raises(DatasetError, match="still open"):
            build_dataset(config, CohortSpec.from_config(config))
    finally:
        handle._mmap.close()
        del handle


def test_load_dataset_detects_inconsistency(driams):
    config, out = driams
    _build(config)
    meta_path = out / "ecoli_ciprofloxacin" / "metadata.csv"
    meta = pd.read_csv(meta_path)
    meta.iloc[:-1].to_csv(meta_path, index=False)
    with pytest.raises(DatasetError, match="does not match"):
        load_dataset(out / "ecoli_ciprofloxacin")
    with pytest.raises(DatasetError, match="not a built dataset"):
        load_dataset(out / "nothing-here")


def test_relative_paths_are_portable(tmp_path):
    rel = spectrum_relpath("DRIAMS-A", "raw", "2018", "abc")
    assert rel == "DRIAMS-A/raw/2018/abc.txt" and "\\" not in rel
    resolved = resolve_relpath(tmp_path, rel)
    assert resolved.parts[-4:] == ("DRIAMS-A", "raw", "2018", "abc.txt")
    # the same relative path maps onto both Windows and POSIX roots
    parts = PurePosixPath(rel).parts
    assert PureWindowsPath("C:/DRIAMS").joinpath(*parts) == PureWindowsPath(r"C:\DRIAMS\DRIAMS-A\raw\2018\abc.txt")
    assert PurePosixPath("/data/DRIAMS").joinpath(*parts) == PurePosixPath("/data/DRIAMS/DRIAMS-A/raw/2018/abc.txt")


def test_assign_group_ids():
    ids, source = assign_group_ids(pd.Series(["a", None, "b", "a", None]))
    assert ids[0] == ids[3] and ids[0] != ids[2]
    assert len({ids[1], ids[4], ids[0], ids[2]}) == 4
    assert list(source) == ["patient_no", "sample", "patient_no", "patient_no", "sample"]


# --- review fix: a metadata row cannot point outside the DRIAMS folder --------------------------------------------

@pytest.mark.parametrize("relpath", ["../../etc/passwd", "DRIAMS-A/../../secret.txt", "/etc/passwd",
                                     "C:/Windows/system.ini", "DRIAMS-A/raw/2018/../../../x.txt", "",
                                     "DRIAMS-A/raw/2018/~root.txt", "DRIAMS-A/raw/2018/a b.txt"])
def test_a_spectrum_path_outside_the_root_is_refused(tmp_path, relpath):
    """Spectrum paths come from metadata files this project does not write, so they are checked."""
    with pytest.raises(DataError):
        resolve_relpath(tmp_path, relpath)


def test_real_spectrum_paths_still_resolve(tmp_path):
    relpath = spectrum_relpath("DRIAMS-A", "raw", "2018", "a1b2c3_3312")
    assert resolve_relpath(tmp_path, relpath) == tmp_path / "DRIAMS-A" / "raw" / "2018" / "a1b2c3_3312.txt"
    uuid_code = "0a1b2c3d-4e5f-6789-abcd-ef0123456789_3313"       # DRIAMS-D style
    assert resolve_relpath(tmp_path, spectrum_relpath("DRIAMS-D", "binned_6000", "2018", uuid_code)).name.endswith(
        f"{uuid_code}.txt")
