"""Version 1.2: the development-only study (docs/v1.2_calibration_plan.md, protocol amendment 9).

Unit tests of src/development.py on small synthetic inputs, then scripts/v12_development.py end to end on a
synthetic DRIAMS folder (the Version 0.3-0.7 end-to-end helpers), with every path in pytest's temporary folder.
What they protect: the pool never touches a spent row or patient; folds keep patients together; nothing a fit,
a calibrator or a cut-off sees is held out; the cut-off is chosen on the predictions the calibrator was fitted on
and is flagged when its target is unsupported; the development log is append-only, dataset-keyed, and can never
be the production log; every output carries its provenance.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.dataset import CohortSpec, build_dataset, load_dataset
from src.development import (
    DEV_LOG_COLUMNS,
    DevelopmentError,
    DevelopmentLog,
    assert_same_scale,
    cross_fitted_probabilities,
    development_pool,
    outer_folds,
    supported_cutoff,
    validation_slice,
    wilson,
)
from src.evaluate import EvaluationError, choose_threshold
from src.splits import Split, build_splits, load_splits
from src.tuning import FamilySpec, fit_calibrated, grouped_folds
from tests.test_model_scripts_e2e import SMALL_LIGHTGBM, make_config, run_script, write_driams

RULE = {"rule": "min_sensitivity", "min_sensitivity": 0.90}
SMALL = {"kind": "lightgbm", "n_jobs": 1, "stochastic": True,
         "fixed": {"subsample_freq": 1, "n_jobs": 1, **SMALL_LIGHTGBM}}


def small_spec(name: str = "lightgbm") -> FamilySpec:
    return FamilySpec.from_config(name, {**SMALL, "grid": [{"n_estimators": [10], "num_leaves": [4]}]})


# ------------------------------------------------------------------------------------------------ units

def test_the_pool_never_touches_a_spent_row_or_a_spent_patient():
    groups = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    meta = pd.DataFrame({"group_id": groups})
    random = Split("random", np.array([0, 2, 4, 5, 6, 7]), np.array([8, 9]), np.array([1, 10, 11]), "")
    temporal = Split("temporal", np.array([4]), np.array([5]), np.array([7]), "")      # row 7 spent here
    pool = development_pool(meta, {"random": random, "temporal": temporal}, "random", ["random", "temporal"])
    # row 0 shares patient 0 with spent row 1; row 7 is spent by `temporal`
    assert pool.tolist() == [2, 4, 5, 6, 8, 9]
    with pytest.raises(DevelopmentError, match="no \\['external'\\] split"):
        development_pool(meta, {"random": random}, "random", ["random", "external"])


def test_outer_folds_keep_patients_together_and_refuse_too_few_resistant():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 3)
    y = np.repeat((rng.random(60) < 0.3).astype(int), 3)
    for tr, te in outer_folds(y, groups, 5, 42, min_resistant=1):
        assert not set(groups[tr]) & set(groups[te])
    with pytest.raises(DevelopmentError, match="fewer than"):
        outer_folds(y, groups, 5, 42, min_resistant=10_000)


def test_the_validation_slice_is_patient_grouped_and_about_one_eighth():
    groups = np.repeat(np.arange(80), 2)
    y = np.repeat(np.arange(80) % 4 == 0, 2).astype(int)
    fit, val = validation_slice(y, groups, 8, 42)
    assert not set(groups[fit]) & set(groups[val]) and len(fit) + len(val) == len(y)
    assert 0.08 <= len(val) / len(y) <= 0.17 and y[val].sum() > 0


def test_a_cutoff_supports_its_target_only_with_fifty_resistant():
    rng = np.random.default_rng(1)
    for positives, supported in ((49, False), (50, True)):
        y = np.r_[np.ones(positives, int), np.zeros(200, int)]
        p = np.r_[rng.uniform(0.3, 1, positives), rng.uniform(0, 0.7, 200)]
        cut = supported_cutoff(y, p, RULE, min_resistant=50)
        assert cut["supported"] is supported and cut["selection_resistant"] == positives
        assert cut["threshold"] == choose_threshold(y, p, RULE) and cut["selection_sensitivity"] >= 0.90
        assert cut["selection_sensitivity_low"] < cut["selection_sensitivity"] < cut["selection_sensitivity_high"]
    low, high = wilson(45, 50)                              # the plan's section 7 figures
    assert (round(low, 2), round(high, 2)) == (0.79, 0.96)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    X = rng.random((240, 30)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.random(240) > 0.9).astype(int)
    groups = np.repeat(np.arange(120), 2)
    spec, setting = small_spec(), {"n_estimators": 10, "num_leaves": 4}
    folds = grouped_folds(y, groups, 5, 42)
    return SimpleNamespace(X=X, y=y, spec=spec, setting=setting, folds=folds,
                           model=fit_calibrated(spec, setting, X, y, folds, 42))


def test_the_cutoff_is_chosen_on_the_predictions_the_calibrator_was_fitted_on(fitted):
    f = fitted
    p = cross_fitted_probabilities(f.model, f.spec, f.setting, f.X, f.y, f.folds, 42)
    assert p.shape == f.y.shape and np.isfinite(p).all() and 0 <= p.min() and p.max() <= 1
    with pytest.raises(DevelopmentError, match="do not reproduce the model's calibrator"):
        cross_fitted_probabilities(f.model, f.spec, f.setting, f.X, 1 - f.y, f.folds, 42)


def test_the_cutoff_and_the_model_share_one_probability_scale(fitted):
    assert_same_scale(fitted.model, fitted.X[:50])
    tampered = copy.deepcopy(fitted.model)
    tampered.calibrated_classifiers_[0].calibrators[0].a_ *= 1.5
    object.__setattr__(tampered, "predict_proba", fitted.model.predict_proba)   # output unchanged, calibrator not
    with pytest.raises(DevelopmentError, match="another scale"):
        assert_same_scale(tampered, fitted.X[:50])


def _dev_row(**over):
    return dict.fromkeys(DEV_LOG_COLUMNS, "") | {"dataset": "d", "experiment": "v1.2/x/cv", "model": "B",
                                                 "seed": 42, "status": "ok"} | over


def test_the_development_log_is_append_only_dataset_keyed_and_never_the_production_log(tmp_path):
    production = tmp_path / "test_evaluations.csv"
    production.write_text("x\n", encoding="utf-8")
    with pytest.raises(DevelopmentError, match="production test log"):
        DevelopmentLog(production, production)
    dev = DevelopmentLog(tmp_path / "dev.csv", production)
    state = dev.append([_dev_row(), _dev_row(model="C")])
    assert state["pre_write_rows"] == 0 and state["expected_rows_after"] == 2
    first = dev.path.read_bytes()
    with pytest.raises(EvaluationError, match="already exist"):                     # same dataset, same key
        dev.append([_dev_row()])
    dev.append([_dev_row(dataset="other")])                                         # another dataset: new key
    assert dev.path.read_bytes().startswith(first) and len(pd.read_csv(dev.path)) == 3
    assert production.read_text(encoding="utf-8") == "x\n"


# ------------------------------------------------------------------------------------------------ end to end

@pytest.fixture(scope="module")
def study(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v12")
    root = tmp / "DRIAMS"
    write_driams(root)
    config = make_config(tmp, root)
    build_dataset(config, CohortSpec.from_config(config))
    out, name = Path(config["dataset"]["output_dir"]), config["dataset"]["name"]
    X, meta, _ = load_dataset(out / name)
    del X
    for split_name, split in build_splits(meta, config, name, out).items():
        split.save(out / name / "splits" / f"{split_name}.json", meta)

    cards = {}
    for key, params in (("T", {"n_estimators": 8, "num_leaves": 4}), ("F", {"n_estimators": 5, "num_leaves": 4})):
        cards[key] = tmp / f"card_{key}.json"
        cards[key].write_text(json.dumps({"model_kind": "lightgbm", "params": params}), encoding="utf-8")
    config["second_antibiotic"]["families"] = {"lightgbm": {**SMALL, "grid": [{"n_estimators": [8]}]},
                                               "lightgbm_cipro_setting": {**SMALL, "grid": [{"n_estimators": [5]}]}}
    vc = config["v12_development"]
    vc.update(dataset=name, expected_pool=None, require_clean_tree=False, outer_folds=3, partition_seeds=[42, 43],
              inner_folds=3, validation_slice_denominator=4, min_resistant_per_heldout_fold=2,
              development_log=str(tmp / "results" / "experiments" / "development_runs.csv"),
              model_dir=str(tmp / "models" / "v1.2"), report_dir=str(tmp / "results" / "metrics" / "v1.2"))
    vc["threshold"]["min_resistant_for_support"] = 12
    vc["forward"]["min_resistant"] = 2
    vc["settings"]["T"]["card"], vc["settings"]["F"]["card"] = str(cards["T"]), str(cards["F"])
    production = Path(config["evaluation"]["test_log"])
    production.parent.mkdir(parents=True, exist_ok=True)
    production.write_text("the production log\n", encoding="utf-8")
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    module = importlib.import_module("v12_development")
    seen: list[tuple[str, np.ndarray, np.ndarray]] = []
    original = module.Ledger.use

    def spy(self, what, rows, held_out, where):
        seen.append((what, np.asarray(rows).copy(), np.asarray(held_out).copy()))
        return original(self, what, rows, held_out, where)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(module.Ledger, "use", spy)
        run = run_script(module, SimpleNamespace(root=tmp), "--config", path)
    splits = load_splits(out / name / "splits", meta)
    spent = np.unique(np.concatenate([splits[s].test for s in vc["spent_splits"]]))
    return SimpleNamespace(tmp=tmp, config=config, path=path, run=run, seen=seen, meta=meta, splits=splits,
                           spent=spent, production=production, module=module,
                           report=Path(vc["report_dir"]) / name, dev_log=Path(vc["development_log"]))


def test_the_study_runs_and_writes_every_report(study):
    assert study.run.code == 0, study.run.stdout
    for name in ("fold_results.csv", "stability.csv", "forward_check.csv", "pooled_results.json",
                 "primary_result.json", "tables.md"):
        assert (study.report / name).is_file(), name
    primary = json.loads((study.report / "primary_result.json").read_text(encoding="utf-8"))
    assert primary["verdict"] in ("C better", "C worse", "not demonstrated") and primary["exploratory"] is True
    assert primary["low"] <= primary["estimate"] <= primary["high"]
    assert set(primary["repeats"]) == {"42", "43"}


def test_nothing_a_fit_calibrator_or_cutoff_sees_is_held_out_or_spent(study):
    assert study.seen, "the ledger recorded nothing"
    kinds = {what for what, _, _ in study.seen}
    assert {"B fit and calibration", "B cut-off", "C fit and calibration", "C cut-off", "D1 cut-off"} <= kinds
    for what, rows, held_out in study.seen:
        assert not np.intersect1d(rows, held_out).size, what
        assert not np.intersect1d(rows, study.spent).size, what


def test_every_pool_spectrum_is_held_out_once_per_partition_and_patients_stay_together(study):
    folds = pd.read_csv(study.report / "fold_results.csv")
    stored = np.load(Path(study.config["v12_development"]["model_dir"]) / study.config["dataset"]["name"] /
                     "heldout_predictions.npz")
    pool, groups = stored["pool_rows"], study.meta["group_id"].to_numpy()
    assert not np.intersect1d(pool, study.spent).size
    assert not np.isin(groups[pool], groups[study.spent]).any()
    for seed in (42, 43):
        fold = stored[f"p{seed}__fold"]
        assert (fold >= 0).all() and set(fold) == {0, 1, 2}
        for g in np.unique(groups[pool]):
            assert np.unique(fold[groups[pool] == g]).size == 1               # a patient is in one fold
        assert len(folds[(folds["partition"] == seed) & (folds["arm"] == "B")]) == 3


def test_unsupported_cutoffs_are_flagged_not_hidden(study):
    folds = pd.read_csv(study.report / "fold_results.csv")
    b, c = folds[folds["arm"] == "B"], folds[folds["arm"] == "C"]
    assert (b["supported"] == (b["selection_resistant"] >= 12)).all()
    assert (c["supported"] == (c["selection_resistant"] >= 12)).all()
    assert (c["selection_resistant"].to_numpy() > b["selection_resistant"].to_numpy()).all()   # C: whole training part


def test_the_development_log_holds_each_arm_once_and_the_production_log_is_untouched(study):
    log = pd.read_csv(study.dev_log)
    assert log.columns.tolist() == DEV_LOG_COLUMNS
    assert len(log) == 2 * 4 + 4 and set(log["status"]) == {"ok"}
    keys = list(zip(log["dataset"], log["experiment"], log["model"], log["seed"], strict=True))
    assert len(set(keys)) == len(keys)
    assert study.production.read_text(encoding="utf-8") == "the production log\n"


def test_every_output_carries_its_provenance(study):
    primary = json.loads((study.report / "primary_result.json").read_text(encoding="utf-8"))["provenance"]
    plan = Path(study.config["v12_development"]["plan"])
    plan = plan if plan.is_absolute() else Path(__file__).resolve().parents[1] / plan
    assert primary["plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()
    for key in ("git_commit", "config_section_sha256", "dataset_row_fingerprint", "pool_fingerprint",
                "production_log_sha256_at_start", "development_log_pre_write"):
        assert primary[key], key
    assert {"card_sha256", "params"} <= set(primary["settings"]["F"])
    log = pd.read_csv(study.dev_log)
    assert set(log["plan_sha256"]) == {primary["plan_sha256"]}
    assert set(log["pool_fingerprint"]) == {primary["pool_fingerprint"]}
    tables = (study.report / "tables.md").read_text(encoding="utf-8")
    assert "exploratory" in tables and "never \"equivalent\"" in tables


def test_a_run_over_budget_is_recorded_as_failed_and_reports_nothing_new(study):
    config = copy.deepcopy(study.config)
    config["v12_development"].update(budget_seconds=0, report_dir=str(study.tmp / "over_budget"))
    path = study.tmp / "config_budget.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    before = len(pd.read_csv(study.dev_log))
    run = run_script(study.module, SimpleNamespace(root=study.tmp), "--config", path)
    assert run.code == 1
    log = pd.read_csv(study.dev_log)
    assert len(log) == before + 1 and log["status"].iloc[-1].startswith("failed: The compute budget")
    assert not (study.tmp / "over_budget" / config["dataset"]["name"] / "primary_result.json").exists()


def test_a_dirty_tree_is_refused_when_a_clean_one_is_required(study, monkeypatch):
    config = copy.deepcopy(study.config)
    config["v12_development"]["require_clean_tree"] = True
    path = study.tmp / "config_clean.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(study.module, "git_commit", lambda: "abc1234-dirty")
    run = run_script(study.module, SimpleNamespace(root=study.tmp), "--config", path)
    assert run.code == 1
