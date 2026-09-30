"""Version 1.1: a second antibiotic (docs/v1.1_ceftriaxone_plan.md, protocol amendment 8).

Two kinds of test. The first reads the real config.yaml and checks it says what the plan fixed: arm T is the
Version 0.4 LightGBM search unchanged, arm F is the Version 0.4 ciprofloxacin setting exactly as its model card
recorded it, and the generalisation section refits T's setting on the planned splits only.

The second runs scripts/tune_models.py and scripts/measure_generalisation.py end to end, through their main()
functions, on a small synthetic DRIAMS folder with two antibiotics, with every path in pytest's temporary
folder (the helpers are the Version 0.3-0.7 end-to-end tests'). It checks what is new in Version 1.1: a
section with no earlier model to compare with, a pre-registered primary family, a family with its own seeds,
and log lookups that must now tell two antibiotics apart.
"""

from __future__ import annotations

import copy
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from src.dataset import CohortSpec, build_dataset, load_dataset
from src.splits import build_splits
from src.utils import load_config, project_path
from tests.test_model_scripts_e2e import (
    EXTERNAL_SITE,
    EXTERNAL_SITE_2,
    EXTERNAL_SIZE,
    EXTERNAL_SIZE_2,
    SITE,
    SMALL_LIGHTGBM,
    YEAR_SIZES,
    Run,
    identifier,
    make_config,
    run_script,
    write_raw,
)

PRIMARY = "e2e_ciprofloxacin"        # ends in its antibiotic, so the second dataset's name can be derived
DATASET = "e2e_ceftriaxone"
SEEDS = [42, 43]
T, F = "tuned_lightgbm", "tuned_lightgbm_cipro_setting"


# ------------------------------------------------------------------------------------------------ the real config

@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    return load_config()


def test_arm_t_is_the_version_04_search_unchanged(config):
    t, v04 = config["second_antibiotic"], config["tuning"]
    assert t["families"]["lightgbm"] == v04["families"]["lightgbm"]
    for key in ("split", "cv_folds", "cv_seed", "search_metric", "calibration", "seeds"):
        assert t[key] == v04[key], key
    assert t["primary_family"] == "lightgbm" and t["retune_experiments"] == []


def test_arm_f_is_the_version_04_ciprofloxacin_setting_exactly(config):
    f = config["second_antibiotic"]["families"]["lightgbm_cipro_setting"]
    card = json.loads(project_path("results/metrics/v0.4/ecoli_ciprofloxacin/best_model_card.json")
                      .read_text(encoding="utf-8"))
    assert len(f["grid"]) == 1 and all(len(values) == 1 for values in f["grid"][0].values())
    assert {name: values[0] for name, values in f["grid"][0].items()} == card["params"]
    assert f["fixed"] == config["tuning"]["families"]["lightgbm"]["fixed"]
    assert (f["kind"], card["model_kind"]) == ("lightgbm", "lightgbm")
    assert f["seeds"] == [42] and f["stochastic"] is True


def test_the_version_11_generalisation_section_matches_the_plan(config):
    g, t = config["second_antibiotic_generalisation"], config["second_antibiotic"]
    assert g["source_model"] == t["model_dir"] and g["source_split"] == t["split"]
    assert g["experiments"] == ["temporal", "external"]
    assert g["reuse_logged"] == ["random"] and g["score_saved_model_on"] == []
    assert "zone_transfer" not in g and "shift_diagnostic" not in g
    assert (g["stage"], g["reference_stage"]) == ("v1.1-generalisation", "v1.1-reference")
    assert g["seeds"] == config["generalisation"]["seeds"]
    for section in (t, g):                           # new outputs only under the v1.1 folders
        assert section["model_dir"].startswith("models/v1.1")
        assert section["report_dir"].startswith("results/metrics/v1.1")
        assert section["plot_dir"] == "results/plots/v1.1"
    assert g["report_dir"] != t["report_dir"]        # both write test_metrics.csv


# ------------------------------------------------------------------------------------------------ end to end

def write_two_antibiotics(root: Path) -> None:
    """The Version 0.3-0.7 synthetic DRIAMS, with a ceftriaxone result next to the ciprofloxacin one.

    The spectra carry the ceftriaxone signal. Some isolates are resistant to one antibiotic only, and a few
    have only a ceftriaxone result, like the 93 real isolates Version 1.1 leaves out of its splits.
    """
    for year, n in YEAR_SIZES.items():
        rows = []
        for k in range(n):
            patient = k // 2
            cef = "R" if patient % 10 in (1, 4, 7) else "S"
            cip = "-" if k % 20 == 13 else "R" if patient % 10 in (1, 4, 8) else "S"
            ids = {col: identifier(f"{year}|{col}", patient) for col in ("patient_no", "case_no", "order_no")}
            date = pd.Timestamp(f"{year}-01-10") + pd.Timedelta(days=k * 340 // n)
            code = f"y{year}_{k:03d}"
            rows.append({"code": code, "species": "Escherichia coli", "laboratory_species": "Escherichia coli",
                         "Ciprofloxacin": cip, "Ceftriaxone": cef, "acquisition_date": date.strftime("%Y-%m-%d"),
                         "acquisition_time": "10:00:00", "workstation": "Blood" if k % 3 == 0 else "Urine", **ids})
            write_raw(root / SITE / "raw" / year / f"{code}.txt", int(year) * 1000 + k, cef == "R")
        id_dir = root / SITE / "id" / year
        id_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(id_dir / f"{year}_strat.csv", index=False)
    for site, size, letter, offset in ((EXTERNAL_SITE, EXTERNAL_SIZE, "z", 900_000),
                                       (EXTERNAL_SITE_2, EXTERNAL_SIZE_2, "w", 950_000)):
        rows = []
        for k in range(size):
            cef = "R" if k % 4 == 0 else "S"
            rows.append({"species": "Escherichia coli", "code": f"{letter}{k:03d}", "combined_code": None,
                         "Ciprofloxacin": "-" if k % 10 == 5 else cef, "Ceftriaxone": cef})
            write_raw(root / site / "raw" / "2018" / f"{letter}{k:03d}.txt", offset + k, cef == "R")
        id_dir = root / site / "id" / "2018"
        id_dir.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(id_dir / "2018_clean.csv", index=False)


def v11_config(tmp: Path, root: Path) -> dict[str, Any]:
    config = make_config(tmp, root)
    config["dataset"]["name"] = PRIMARY
    small = {"kind": "lightgbm", "n_jobs": 1, "stochastic": True,
             "fixed": {"subsample_freq": 1, "n_jobs": 1, **SMALL_LIGHTGBM}}
    t = config["second_antibiotic"]
    t.update(cv_folds=3, seeds=SEEDS, model_dir=str(tmp / "models" / "v1.1"),
             report_dir=str(tmp / "results" / "metrics" / "v1.1"), plot_dir=str(tmp / "results" / "plots" / "v1.1"))
    t["families"] = {
        "lightgbm": {**copy.deepcopy(small), "random_draws": 2,
                     "space": {"n_estimators": [5, 10], "learning_rate": [0.1], "num_leaves": [4],
                               "colsample_bytree": [0.3, 1.0], "subsample": [0.7, 1.0]}},
        "lightgbm_cipro_setting": {**copy.deepcopy(small), "seeds": [42],
                                   "grid": [{"n_estimators": [8], "learning_rate": [0.1], "num_leaves": [4],
                                             "colsample_bytree": [0.3], "subsample": [1.0]}]},
    }
    g = config["second_antibiotic_generalisation"]
    g.update(source_model=t["model_dir"], seeds=SEEDS, model_dir=str(tmp / "models" / "v1.1" / "generalisation"),
             report_dir=str(tmp / "results" / "metrics" / "v1.1" / "generalisation"), plot_dir=t["plot_dir"])
    for section in (t, g):
        assert all(Path(section[k]).resolve().is_relative_to(tmp.resolve()) for k in ("model_dir", "report_dir"))
    return config


@dataclass
class Result:
    tmp: Path
    config: dict[str, Any]
    config_path: Path
    runs: dict[str, Run]
    logs: dict[str, pd.DataFrame]

    def report(self, section: str) -> Path:
        return Path(self.config[section]["report_dir"]) / DATASET

    def read_json(self, section: str, name: str) -> dict[str, Any]:
        return json.loads((self.report(section) / name).read_text(encoding="utf-8"))

    def new_rows(self, before: str, after: str) -> pd.DataFrame:
        return self.logs[after].iloc[len(self.logs[before]):]


@pytest.fixture(scope="module")
def v11(tmp_path_factory) -> Result:
    tmp = tmp_path_factory.mktemp("second_antibiotic")
    root = tmp / "DRIAMS"
    write_two_antibiotics(root)
    config = v11_config(tmp, root)
    out = Path(config["dataset"]["output_dir"])
    for antibiotic in (None, "Ceftriaxone"):         # the primary first: the second reuses its splits
        spec = CohortSpec.from_config(config, antibiotic=antibiotic)
        build_dataset(config, spec)
        X, meta, _ = load_dataset(out / spec.name)
        del X
        for name, split in build_splits(meta, config, spec.name, out).items():
            split.save(out / spec.name / "splits" / f"{name}.json", meta)
    config_path = tmp / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    ws, log_path = SimpleNamespace(root=tmp), Path(config["evaluation"]["test_log"])
    tune, generalise = importlib.import_module("tune_models"), importlib.import_module("measure_generalisation")
    tune_args = ("--config", config_path, "--section", "second_antibiotic", "--dataset", DATASET)
    result = Result(tmp, config, config_path, {}, {})

    def read_log() -> pd.DataFrame:
        return pd.read_csv(log_path) if log_path.is_file() and log_path.stat().st_size else pd.DataFrame()

    result.logs["start"] = read_log()
    result.runs["development"] = run_script(tune, ws, *tune_args)
    result.logs["development"] = read_log()
    result.runs["test"] = run_script(tune, ws, *tune_args, "--evaluate-test")
    result.logs["test"] = read_log()
    # A row for the *other* antibiotic with the same split, model name and seed, appended after Version 1.1's.
    # Without a dataset filter, the generalisation run would check its stored probabilities against this row.
    decoy = result.logs["test"].iloc[[0]].copy()
    for column, value in (("dataset", PRIMARY), ("stage", "v0.4-tuned"), ("roc_auc", 0.123456), ("brier", 0.456789)):
        decoy[column] = value
    decoy.to_csv(log_path, mode="a", header=False, index=False, lineterminator="\n")
    result.logs["decoy"] = read_log()
    result.runs["generalisation"] = run_script(
        generalise, ws, "--config", config_path, "--section", "second_antibiotic_generalisation",
        "--dataset", DATASET)
    result.logs["generalisation"] = read_log()
    return result


def test_the_development_run_fits_both_arms_and_scores_nothing(v11):
    run = v11.runs["development"]
    assert run.code == 0, run.log.messages
    assert len(v11.logs["development"]) == len(v11.logs["start"]) == 0
    validation = pd.read_csv(v11.report("second_antibiotic") / "validation_metrics.csv")
    calibrated = validation[validation["calibrated"]]
    assert sorted(calibrated.loc[calibrated["model"] == T, "seed"]) == SEEDS          # T: every seed
    assert sorted(calibrated.loc[calibrated["model"] == F, "seed"]) == [42]           # F: its own seeds only
    assert "v0.3_prevalence" not in set(validation["model"])                          # no earlier run to carry


def test_the_pre_registered_family_is_saved_not_the_best_on_validation(v11):
    card = v11.read_json("second_antibiotic", "best_model_card.json")
    assert card["model"] == T and "pre-registered primary family" in card["selection"]["rule"]
    # Name the other arm as primary (own folders, so nothing above is touched): the saved model follows the
    # configuration, so whichever arm validates better, one of these two runs saved the one that did not.
    config = copy.deepcopy(v11.config)
    other = v11.tmp / "primary_is_f"
    config["second_antibiotic"].update(primary_family="lightgbm_cipro_setting", model_dir=str(other / "models"),
                                       report_dir=str(other / "reports"), plot_dir=str(other / "plots"))
    path = v11.tmp / "config_primary_is_f.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run = run_script(importlib.import_module("tune_models"), SimpleNamespace(root=v11.tmp), "--config", path,
                     "--section", "second_antibiotic", "--dataset", DATASET)
    assert run.code == 0, run.log.messages
    card = json.loads((other / "reports" / DATASET / "best_model_card.json").read_text(encoding="utf-8"))
    assert card["model"] == F


def test_the_test_run_scores_both_arms_once_with_no_earlier_model(v11):
    run = v11.runs["test"]
    assert run.code == 0, run.log.messages
    rows = v11.new_rows("development", "test")
    assert set(rows["stage"]) == {"v1.1-tuned"} and set(rows["dataset"]) == {DATASET}
    assert set(rows["split"]) == {"random"}
    assert sorted(zip(rows["model"], rows["seed"], strict=True)) == [(T, 42), (T, 43), (F, 42)]
    intervals = v11.read_json("second_antibiotic", "test_intervals.json")["random"]
    assert F in intervals["differences_to_reference"]                               # T minus F, paired
    assert "differences_to_reference_model" not in intervals
    stored = np.load(Path(v11.config["second_antibiotic"]["model_dir"]) / DATASET / "test_probabilities.npz")
    assert {"random__rows", f"random__{T}__seed42", f"random__{F}__seed42"} <= set(stored.files)


def test_generalisation_runs_its_own_section_and_nothing_the_plan_did_not_name(v11):
    run = v11.runs["generalisation"]
    assert run.code == 0, run.log.messages
    assert run.log.has("refit proved faithful")
    rows = v11.new_rows("decoy", "generalisation")
    refits = rows[rows["stage"] == "v1.1-generalisation"]
    assert set(refits["dataset"]) == {DATASET} and set(refits["model"]) == {T}
    assert sorted(set(refits["experiment"])) == [f"external__{EXTERNAL_SITE_2}", f"external__{EXTERNAL_SITE}",
                                                  f"temporal__{SITE}"]
    assert sorted(set(refits["seed"])) == SEEDS and len(refits) == 3 * len(SEEDS)
    assert set(rows["stage"]) == {"v1.1-generalisation", "v1.1-reference"}           # no Version 0.7 stage
    assert not any("saved_model" in e for e in rows["experiment"])                   # the saved model is not scored
    report = v11.report("second_antibiotic_generalisation")
    assert not (report / "zone_transfer.csv").exists() and not (report / "shift_diagnostic.csv").exists()
    gaps = pd.read_csv(report / "generalisation_gaps.csv")
    assert len(gaps) == 3 and gaps["low"].notna().all()


def test_log_lookups_tell_the_two_antibiotics_apart(v11):
    # The decoy (other dataset, same split, model and seed, wrong AUROC and Brier) is the last matching row in
    # the log; the run passing at all shows the stored probabilities were checked against Version 1.1's own row.
    assert v11.runs["generalisation"].code == 0
    reused = pd.read_csv(v11.report("second_antibiotic_generalisation") / "reused_from_log.csv")
    assert sorted(reused["seed"]) == SEEDS                          # T's two seeds, and nothing else
    assert 0.123456 not in set(reused["roc_auc"])


def test_every_earlier_log_row_is_unchanged(v11):
    before, after = v11.logs["decoy"], v11.logs["generalisation"]
    pd.testing.assert_frame_equal(after.iloc[:len(before)].reset_index(drop=True), before.reset_index(drop=True))
