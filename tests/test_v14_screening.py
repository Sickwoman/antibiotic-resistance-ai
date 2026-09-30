"""Version 1.4: excluded screening isolates as training-only data (docs/v1.4_screening_plan.md, amendment 11).

Unit tests of src/screening.py and the dataset option that builds the screening isolates, then
scripts/v14_screening.py end to end on a synthetic DRIAMS folder with synthetic HospitalHygiene rows. The real
Version 1.2 study runs first, so the Version 1.4 baseline must reproduce its arm C exactly, as on the real data.
What they protect: a screening spectrum is never evaluated and never trains while its patient is held out; a
patient with a clinical row outside the pool contributes nothing; nothing dated 2018 or later is used; the baseline
is the recorded one; a failed check stops the run and is logged; the production log is never touched.
"""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.dataset import CohortSpec, DatasetError, build_dataset, load_dataset
from src.development import development_pool
from src.screening import (
    JOINS,
    NEW,
    ScreeningError,
    corrected_cv_interval,
    group_bootstrap,
    link_screening,
    training_screening,
    verdict,
)
from src.splits import build_splits, load_splits
from tests.test_model_scripts_e2e import (
    SITE,
    SMALL_LIGHTGBM,
    YEAR_SIZES,
    identifier,
    make_config,
    run_script,
    write_driams,
    write_raw,
)

SMALL = {"kind": "lightgbm", "n_jobs": 1, "stochastic": True,
         "fixed": {"subsample_freq": 1, "n_jobs": 1, **SMALL_LIGHTGBM}}
SCREENING = "HospitalHygiene"


# ------------------------------------------------------------------------------------------------ the dataset option

def test_an_excluded_workstation_is_kept_only_under_another_name(tmp_path):
    config = make_config(tmp_path, tmp_path / "DRIAMS")
    plain = CohortSpec.from_config(config)
    kept = CohortSpec.from_config(config, keep_workstations=[SCREENING])
    assert SCREENING in plain.exclude_workstations and SCREENING not in kept.exclude_workstations
    assert kept.name == plain.name + "__with-hospitalhygiene"
    assert CohortSpec.from_config(config, name="x_with_screening", keep_workstations=[SCREENING]).name == \
        "x_with_screening"
    with pytest.raises(DatasetError, match="would be overwritten"):
        CohortSpec.from_config(config, name=plain.name, keep_workstations=[SCREENING])
    with pytest.raises(DatasetError, match="nothing to keep"):
        CohortSpec.from_config(config, keep_workstations=["Urine"])


# ------------------------------------------------------------------------------------------------ linking, by hand

def _hand_made():
    """A base dataset of 8 clinical rows and a full one with them (other order, other group numbers) + screening."""
    base = pd.DataFrame({
        "site": "DRIAMS-A", "year_folder": "2016", "code": [f"c{i}" for i in range(8)],
        "label": [1, 0, 0, 1, 0, 0, 1, 0], "group_id": [0, 0, 1, 2, 3, 4, 5, 6],
        "workstation": "Urine", "acquisition_date": pd.date_range("2016-01-01", periods=8, freq="10D")})
    rng = np.random.default_rng(0)
    X = rng.random((8, 5)).astype(np.float32)
    pool = np.array([0, 1, 2, 3, 4, 5])                                  # rows 6, 7 (groups 5, 6) outside the pool
    screen = pd.DataFrame({
        "site": "DRIAMS-A", "year_folder": "2016", "code": [f"s{i}" for i in range(5)],
        "label": [1, 1, 1, 1, 0],
        "group_id": [100, 101, 101, 105, 102],       # full numbering: 100 = patient of group 0; 105 = of group 5
        "workstation": SCREENING,
        "acquisition_date": pd.to_datetime(["2016-03-01", "2016-04-01", "2016-05-01", "2016-06-01", "2018-02-01"])})
    full_numbering = {0: 100, 1: 103, 2: 104, 3: 106, 4: 107, 5: 105, 6: 108}
    clinical = base.assign(group_id=base["group_id"].map(full_numbering))
    order = [3, 7, 0, 5, 1, 6, 2, 4]
    full = pd.concat([clinical.iloc[order], screen], ignore_index=True)
    X_full = np.concatenate([X[order], rng.random((5, 5)).astype(np.float32)])
    return base, X, pool, full, X_full


def test_screening_rows_join_their_pool_patient_or_become_new_patients_and_nothing_else():
    base, X, pool, full, X_full = _hand_made()
    s = link_screening(base, X, pool, full, X_full, SCREENING, "2018-01-01")
    # s0 joins pool patient 0; s1, s2 are one new patient; s3's patient (group 5) is outside the pool; s4 is 2018
    assert full.loc[s.rows, "code"].tolist() == ["s0", "s1", "s2"]
    assert s.kind.tolist() == [JOINS, NEW, NEW]
    assert s.groups[0] == 0 and s.groups[1] == s.groups[2] > base["group_id"].max()
    assert s.counts == {"usable": 3, "usable_resistant": 3, "patient_groups": 2, "joining": 1, "joining_groups": 1,
                        "new": 2, "new_resistant": 2, "new_groups": 1, "new_resistant_groups": 1,
                        "excluded_outside_pool": 1}
    assert (s.dates < np.datetime64("2018-01-01")).all()


@pytest.mark.parametrize("tamper, message", [
    ("features", "different features"), ("label", "different label"), ("grouping", "group the shared rows"),
    ("missing", "missing from"), ("other workstation", "not HospitalHygiene")])
def test_the_link_refuses_any_difference_but_the_screening_rows(tamper, message):
    base, X, pool, full, X_full = _hand_made()
    if tamper == "features":
        X_full[2, 0] += 1e-3
    elif tamper == "label":
        full.loc[2, "label"] = 1 - full.loc[2, "label"]
    elif tamper == "grouping":
        full.loc[full["code"] == "c2", "group_id"] = 100          # c2 would join c0's patient
    elif tamper == "missing":
        full, X_full = full.drop(index=0).reset_index(drop=True), X_full[1:]
    else:
        full.loc[full["code"] == "s1", "workstation"] = "Urine"
    with pytest.raises(ScreeningError, match=message):
        link_screening(base, X, pool, full, X_full, SCREENING, "2018-01-01")


def test_a_fold_never_trains_on_a_held_out_patients_screening_rows():
    base, X, pool, full, X_full = _hand_made()
    s = link_screening(base, X, pool, full, X_full, SCREENING, "2018-01-01")
    assert training_screening(s, np.array([0])).tolist() == [1, 2]                # patient 0 held out: s0 dropped
    assert training_screening(s, np.array([3])).tolist() == [0, 1, 2]
    assert training_screening(s, np.array([3]), before="2016-04-15").tolist() == [0, 1]


# ------------------------------------------------------------------------------------------------ uncertainty

def test_the_corrected_interval_is_the_nadeau_bengio_formula():
    d = np.array([0.02, -0.01, 0.03, 0.00, 0.01, 0.04, -0.02, 0.02, 0.01, 0.00, 0.03, 0.01, -0.01, 0.02, 0.00])
    out = corrected_cv_interval(d, folds=5, repeats=3)
    se = np.sqrt((1 / 15 + 1 / 4) * d.var(ddof=1))
    assert out["estimate"] == pytest.approx(d.mean()) and out["se"] == pytest.approx(se) and out["df"] == 14
    assert out["high"] - out["estimate"] == pytest.approx(2.1447866879 * se, rel=1e-6)      # t(0.975, 14)
    naive = 2.1447866879 * d.std(ddof=1) / np.sqrt(15)
    assert out["high"] - out["estimate"] > 2 * naive                              # the correction widens it
    with pytest.raises(ScreeningError, match="Expected 15"):
        corrected_cv_interval(d[:14], folds=5, repeats=3)


def test_the_bootstrap_resamples_whole_patient_groups_and_counts_skipped_replicates():
    groups = np.array([7, 7, 7, 3, 3, 9, 1, 1, 1, 1])
    sizes = {g: int((groups == g).sum()) for g in np.unique(groups)}

    def whole(rows):
        drawn = pd.Series(groups[rows]).value_counts()
        assert all(n % sizes[g] == 0 for g, n in drawn.items())                  # every group comes whole
        return None if 9 not in drawn.index else {"n": float(len(rows))}

    values, skipped = group_bootstrap(groups, whole, resamples=300, seed=1)
    assert skipped > 0 and len(values["n"]) + skipped == 300


def test_the_verdict_follows_the_plan():
    assert verdict(0.001, 0.05) == "improved ranking"
    assert verdict(-0.05, -0.001) == "worse ranking"
    assert verdict(-0.01, 0.02) == verdict(0.0, 0.02) == "not demonstrated"


# ------------------------------------------------------------------------------------------------ end to end

def add_screening_rows(root: Path) -> None:
    """Synthetic HospitalHygiene rows: one for every second existing patient (so some belong to patients whose
    clinical spectra are in a spent test part) and 4 new patients, in every year folder (2018's are dated 2018, so
    the study must drop them). Mostly resistant, as the real screening isolates are."""
    for year, n in YEAR_SIZES.items():
        path = root / SITE / "id" / year / f"{year}_strat.csv"
        table = pd.read_csv(path, dtype=str)
        rows = []
        for j, patient in enumerate([*range(0, n // 2, 2), *range(1000, 1004)]):
            ids = {col: identifier(f"{year}|{col}", patient) for col in ("patient_no", "case_no", "order_no")}
            value = "S" if j % 6 == 0 else "R"
            code = f"h{year}_{j:03d}"
            date = pd.Timestamp(f"{year}-02-01") + pd.Timedelta(days=j * 10)
            rows.append({"code": code, "species": "Escherichia coli", "laboratory_species": "Escherichia coli",
                         "Ciprofloxacin": value, "acquisition_date": date.strftime("%Y-%m-%d"),
                         "acquisition_time": "10:00:00", "workstation": SCREENING, **ids})
            write_raw(root / SITE / "raw" / year / f"{code}.txt", 500_000 + int(year) * 100 + j, value != "S")
        pd.concat([table, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v14")
    root = tmp / "DRIAMS"
    write_driams(root)
    add_screening_rows(root)
    config = make_config(tmp, root)
    build_dataset(config, CohortSpec.from_config(config))
    out, name = Path(config["dataset"]["output_dir"]), config["dataset"]["name"]
    X, meta, _ = load_dataset(out / name)
    del X
    for split_name, split in build_splits(meta, config, name, out).items():
        split.save(out / name / "splits" / f"{split_name}.json", meta)
    build_dataset(config, CohortSpec.from_config(config, name=f"{name}_with_screening", keep_workstations=[SCREENING]))

    card = tmp / "card_F.json"
    card.write_text(json.dumps({"model_kind": "lightgbm", "params": {"n_estimators": 5, "num_leaves": 4}}),
                    encoding="utf-8")
    card_t = tmp / "card_T.json"
    card_t.write_text(json.dumps({"model_kind": "lightgbm", "params": {"n_estimators": 8, "num_leaves": 4}}),
                      encoding="utf-8")
    config["second_antibiotic"]["families"] = {"lightgbm": {**SMALL, "grid": [{"n_estimators": [8]}]},
                                               "lightgbm_cipro_setting": {**SMALL, "grid": [{"n_estimators": [5]}]}}
    vc = config["v12_development"]                             # the real Version 1.2 study, run first
    vc.update(dataset=name, expected_pool=None, require_clean_tree=False, outer_folds=3, partition_seeds=[42, 43],
              inner_folds=3, validation_slice_denominator=4, min_resistant_per_heldout_fold=2,
              development_log=str(tmp / "dev_v12.csv"), model_dir=str(tmp / "models" / "v1.2"),
              report_dir=str(tmp / "results" / "metrics" / "v1.2"))
    vc["threshold"]["min_resistant_for_support"] = 12
    vc["forward"]["min_resistant"] = 2
    vc["settings"]["T"]["card"], vc["settings"]["F"]["card"] = str(card_t), str(card)
    tc = config["v14_screening"]
    tc.update(dataset=name, screening_dataset=f"{name}_with_screening", expected_pool=None, require_clean_tree=False,
              outer_folds=3, partition_seeds=[42, 43], inner_folds=3, min_resistant_per_heldout_fold=2,
              development_log=str(tmp / "dev_v14.csv"), model_dir=str(tmp / "models" / "v1.4"),
              report_dir=str(tmp / "results" / "metrics" / "v1.4"))
    tc["screening"]["expected"] = None
    tc["threshold"]["min_resistant_for_support"] = 12
    tc["setting"]["card"] = str(card)
    tc["reproduce"]["predictions"] = str(tmp / "models" / "v1.2" / name / "heldout_predictions.npz")
    tc["bootstrap"]["resamples"] = 60
    production = Path(config["evaluation"]["test_log"])
    production.parent.mkdir(parents=True, exist_ok=True)
    production.write_text("the production log\n", encoding="utf-8")

    path = tmp / "config_v12.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    v12 = run_script(importlib.import_module("v12_development"), SimpleNamespace(root=tmp), "--config", path)
    assert v12.code == 0, v12.stdout
    splits = load_splits(out / name / "splits", meta)
    spent = np.unique(np.concatenate([splits[s].test for s in tc["spent_splits"]]))
    pool = development_pool(meta, splits, "random", list(tc["spent_splits"]))
    return SimpleNamespace(tmp=tmp, config=config, meta=meta, spent=spent, pool=pool, production=production,
                           module=importlib.import_module("v14_screening"), name=name)


def _run(world, config, tag):
    """One run with its own development log and a clean-tree commit id, as in CI (see the Version 1.3 tests)."""
    config = copy.deepcopy(config)
    config["v14_screening"]["development_log"] = str(world.tmp / f"dev_v14_{tag}.csv")
    path = world.tmp / f"config_v14_{tag}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    seen = []
    original = world.module.Ledger.use

    def spy(self, what, clinical, screening, held_out, where, before):
        seen.append({"what": what, "clinical": np.asarray(clinical).copy(),
                     "screening": self.run.screen.rows[np.asarray(screening, dtype=np.int64)].copy(),
                     "screening_groups": self.run.screen.groups[np.asarray(screening, dtype=np.int64)].copy(),
                     "held_out": np.asarray(held_out).copy(), "where": where, "before": before})
        return original(self, what, clinical, screening, held_out, where, before)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(world.module.Ledger, "use", spy)
        mp.setattr(world.module, "git_commit", lambda: "abc1234")
        run = run_script(world.module, SimpleNamespace(root=world.tmp), "--config", path)
    return run, seen, Path(config["v14_screening"]["development_log"])


@pytest.fixture(scope="module")
def feasible(world):
    return _run(world, world.config, "feasible")


def test_the_study_runs_reproduces_version_1_2_and_reports_the_primary(world, feasible):
    run, _, _ = feasible
    assert run.code == 0, run.stdout
    report = Path(world.config["v14_screening"]["report_dir"]) / world.name
    primary = json.loads((report / "primary_result.json").read_text(encoding="utf-8"))
    assert primary["exploratory"] is True and primary["verdict"] in ("improved ranking", "worse ranking",
                                                                     "not demonstrated")
    assert primary["folds"] == 6 and len(primary["per_fold"]) == 6                # 3 folds x 2 partitions
    assert all(v["largest_probability_difference"] <= 1e-12 for v in primary["reproduction"].values())
    assert primary["reproduction"]["42"]["decisions_identical"] is True
    folds = pd.read_csv(report / "fold_results.csv")
    assert (folds.loc[folds["arm"] == "A0", "screening_rows"] == 0).all()
    assert (folds.loc[folds["arm"] == "A1", "screening_rows"] > 0).all()
    tables = (report / "tables.md").read_text(encoding="utf-8")
    assert "exploratory" in tables and "no usefulness requirement exists" in tables


def test_no_screening_row_is_evaluated_or_trains_while_its_patient_is_held_out(world, feasible):
    _, seen, _ = feasible
    groups = world.meta["group_id"].to_numpy()
    dates = pd.to_datetime(world.meta["acquisition_date"]).to_numpy()
    kinds = {s["what"] for s in seen}
    assert {"A0 fit and calibration", "A1 fit and calibration", "source diagnostic"} <= kinds
    assert any(s["screening"].size for s in seen if s["what"] == "A1 fit and calibration")
    for s in seen:
        held_groups = np.unique(groups[s["held_out"]])
        assert np.isin(s["held_out"], world.pool).all()                          # only clinical pool rows are held out
        assert not np.intersect1d(s["clinical"], s["held_out"]).size, s["where"]
        assert not np.isin(s["screening_groups"], held_groups).any(), s["where"]
        assert np.isin(s["clinical"], world.pool).all() and not np.intersect1d(s["clinical"], world.spent).size
        assert (dates[s["clinical"]] < np.datetime64(pd.Timestamp(s["before"]))).all()
        if s["what"].startswith("A0"):
            assert s["screening"].size == 0


def test_screening_rows_of_patients_outside_the_pool_and_of_2018_are_never_used(world, feasible):
    _, seen, _ = feasible
    full = load_dataset(Path(world.config["dataset"]["output_dir"]) / f"{world.name}_with_screening")[1]
    used = np.unique(np.concatenate([s["screening"] for s in seen]))
    assert used.size and (full.loc[used, "workstation"] == SCREENING).all()
    assert (pd.to_datetime(full.loc[used, "acquisition_date"]) < pd.Timestamp("2018-01-01")).all()

    def key(m: pd.DataFrame) -> pd.Series:
        return m["site"] + "|" + m["year_folder"].astype(str) + "|" + m["code"]

    clinical = full[full["workstation"] != SCREENING]
    outside = clinical[~key(clinical).isin(set(key(world.meta.iloc[world.pool])))]
    screening = full[(full["workstation"] == SCREENING)
                     & (pd.to_datetime(full["acquisition_date"]) < pd.Timestamp("2018-01-01"))]
    assert screening["group_id"].isin(outside["group_id"]).any()             # the rule is exercised here
    assert not full.loc[used, "group_id"].isin(outside["group_id"]).any()


def test_the_development_log_gains_the_planned_rows_and_production_is_untouched(world, feasible):
    _, _, dev_log = feasible
    log = pd.read_csv(dev_log)
    ok = log[log["status"] == "ok"]
    assert len(ok) == 2 * 2 + 2 + 1          # two arms x two partitions, two forward rows, one source diagnostic
    keys = list(zip(ok["dataset"], ok["experiment"], ok["model"], ok["seed"], strict=True))
    assert len(set(keys)) == len(keys)
    assert world.production.read_text(encoding="utf-8") == "the production log\n"


def test_a_baseline_that_does_not_reproduce_version_1_2_stops_the_run(world, feasible):
    config = copy.deepcopy(world.config)
    tc = config["v14_screening"]
    saved = dict(np.load(tc["reproduce"]["predictions"]))
    saved["p42__C"] = saved["p42__C"] + 1e-6
    tampered = world.tmp / "tampered.npz"
    np.savez(tampered, **saved)
    tc.update(report_dir=str(world.tmp / "not_reproduced"))
    tc["reproduce"]["predictions"] = str(tampered)
    run, _, dev_log = _run(world, config, "not_reproduced")
    assert run.code == 1
    status = pd.read_csv(dev_log)["status"].iloc[-1]
    assert status.startswith("failed:") and "does not reproduce" in status
    assert not (world.tmp / "not_reproduced" / world.name / "primary_result.json").exists()


def test_screening_counts_that_differ_from_the_audit_stop_the_run(world, feasible):
    config = copy.deepcopy(world.config)
    config["v14_screening"]["screening"]["expected"] = {"usable": 10 ** 6}
    config["v14_screening"]["report_dir"] = str(world.tmp / "mismatch")
    run, _, dev_log = _run(world, config, "mismatch")
    assert run.code == 1
    status = pd.read_csv(dev_log)["status"].iloc[-1]
    assert status.startswith("failed:") and "differ from the plan's audit" in status
