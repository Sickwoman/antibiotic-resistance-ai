"""Version 2.0, steps 2-3: the preparation code scores nothing, and only the restricted schema reader opens AMR.csv.

Also the archive-side pieces of scripts/v20_prepare_marisma.py, on synthetic archives: the reader that records every
attempt with its batch, selection and features end to end, and the mean-spectrum alignment check.
"""

from __future__ import annotations

import ast
import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT / "scripts/v20_prepare_marisma.py"
REFERENCE_CHECK = ROOT / "scripts/v20_reader_reference_check.py"
SCORING_NAMES = {"predict", "predict_proba", "decision_function", "predict_features", "predict_spectrum_file",
                 "fit", "fit_transform", "partial_fit"}


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", [PREPARATION, REFERENCE_CHECK])
def test_the_step3_scripts_never_score_or_fit(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    used |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not used & SCORING_NAMES


def test_only_the_restricted_reader_opens_amr_csv():
    # Amendment A2's reconciled rule: before scoring, the only code that opens AMR.csv is the restricted schema
    # reader. The download script names the file to fetch and checksum it; the preparation script passes its path to
    # the reader and checks its size; nothing else in src/ or scripts/ may name it.
    allowed = {"src/marisma_schema.py", "scripts/download_marisma.py", "scripts/v20_prepare_marisma.py"}
    naming = {p.relative_to(ROOT).as_posix() for folder in ("src", "scripts") for p in (ROOT / folder).rglob("*.py")
              if "AMR.csv" in p.read_text(encoding="utf-8") or "_sealed" in p.read_text(encoding="utf-8")}
    assert naming <= allowed
    tree = ast.parse(PREPARATION.read_text(encoding="utf-8"))
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        if any(isinstance(a, ast.Attribute) and a.attr == "amr" for a in call.args):
            callee = call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id
            assert callee in {"read_columns", "schema_check", "sample_categories", "source_classes"}, callee


def write_spectrum_files(ml1=3.19e6, td=28750, intensities=None, extra=()):
    lines = ["##TITLE= synthetic", f"##$TD= {td}", "##$DELAY= 25000", "##$DW= 2.0", f"##$ML1= {ml1}",
             "##$ML2= 1000", "##$ML3= 0.0025", "##$BYTORDA= 0", "##$AQ_DATE= <2021-03-04T10:00:00.000+01:00>",
             "##$INSTRUM= <synthetic>", *extra]
    if intensities is None:
        mz_peaks = (2500, 4365, 6255, 9000, 15000)
        tof = 25000 + np.arange(td) * 2.0
        b = np.sqrt(1e12 / ml1)
        mz = ((-b + np.sqrt(b * b - 4 * 0.0025 * (1000 - tof))) / (2 * 0.0025)) ** 2
        intensities = np.round(200 + sum(5000 * np.exp(-0.5 * ((mz - c) / 4.0) ** 2) for c in mz_peaks))
    return ("\n".join(lines) + "\n").encode("latin-1"), np.asarray(intensities, dtype="<i4").tobytes()


def test_selection_and_features_on_a_synthetic_archive(tmp_path):
    prep = load(PREPARATION)
    good, bad = write_spectrum_files(), write_spectrum_files(td=2000)          # bad: does not reach 20,000 Da
    layout = {("iso1", "b1", "1"): good, ("iso2", "b1", "1"): bad, ("iso2", "b2", "1"): good,
              ("iso3", "b1", "1"): bad}
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for (iso, b, t), (acqu, fid) in layout.items():
            z.writestr(f"MARISMa/2021/Escherichia/Coli/{iso}/{b}/{t}/1SLin/acqu", acqu)
            z.writestr(f"MARISMa/2021/Escherichia/Coli/{iso}/{b}/{t}/1SLin/fid", fid)
        z.writestr("MARISMa/2021/Escherichia/Coli/iso4/.DS_Store", b"")          # no replicate folder
        z.writestr("MARISMa/2021/Klebsiella/Pneumoniae/iso5/b1/1/1SLin/fid", fid)  # another species
        z.writestr("MARISMa/2021/Escherichia/Coli/iso5/b1/1/1SLin/fid", fid)       # ... and E. coli: inconsistent
    from src.preprocessing import PreprocessingConfig
    from src.zip_index import iter_members
    members = dict(iter_members(archive))
    cohort, counts, folders = prep.build_cohort(list(members))
    assert sorted(cohort) == ["iso1", "iso2", "iso3", "iso4"]
    assert counts["species_consistency_rule"]["excluded_identifiers"] == 1
    assert counts["replicate_folders"]["isolates_without_any"] == 1
    with open(archive, "rb") as fh:
        selection, outcomes, attempts, X = prep.select_and_featurise(cohort, fh, members, PreprocessingConfig(),
                                                                      tmp_path)
    assert selection["status"].tolist() == ["selected", "selected", "no_passing_replicate", "no_replicate_folder"]
    assert selection["biological"].tolist() == ["b1", "b2", "", ""]
    assert selection["failures"].tolist() == ["", "range_not_covered", "range_not_covered", ""]
    assert [o.status for o in outcomes] == selection["status"].tolist()
    assert X.shape == (4, 6000) and np.isfinite(X).all()
    assert np.array_equal(X[0], X[1]) and not X[2].any()                     # same spectrum, same features
    from src.marisma_cohort import pause_check
    assert (pause_check(outcomes)["numerator"], pause_check(outcomes)["denominator"]) == (1, 3)
    assert [a["reason"] for a in attempts] == ["passed", "range_not_covered", "passed", "range_not_covered"]
    assert {a["acquisition_month"] for a in attempts} == {"2021-03"}
    review = prep.batch_review(attempts)
    assert review["by_year"] == {"2021": {"passed": 2, "range_not_covered": 2}}


def test_the_alignment_check_finds_a_known_shift():
    prep = load(PREPARATION)
    from src.preprocessing import PreprocessingConfig
    x = np.zeros(6000)
    for c in (100, 800, 1500, 2200, 3100, 4000, 5000):
        x += np.exp(-0.5 * ((np.arange(6000) - c) / 2.0) ** 2)
    assert prep.peak_offsets(x, x, PreprocessingConfig(), top=5)["best_lag_bins"] == 0
    shifted = prep.peak_offsets(x, np.roll(x, 3), PreprocessingConfig(), top=5)
    assert shifted["best_lag_bins"] == 3
    assert np.allclose(shifted["offset_to_nearest_peak_da"], 9.0)            # 3 bins of 3 Da


def test_the_schema_check_is_refused_while_the_pause_holds(tmp_path):
    import argparse
    prep = load(PREPARATION)
    args = argparse.Namespace(work=tmp_path, amr=tmp_path / "AMR.csv", columns={"ciprofloxacin": "x"})
    with pytest.raises(prep.StepError):
        prep.run_schema(args, {"pause_check": {"pause": True}})
