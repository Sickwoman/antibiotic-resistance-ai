"""Version 1.3: cut-off rules for a sensitivity target, over time (docs/v1.3_threshold_plan.md, amendment 10).

Unit tests of src/threshold_rules.py - the two rules, their stated properties checked by simulation, time
ordering, label availability, patients seen earlier, one selection spectrum per patient - then
scripts/v13_threshold.py end to end on a synthetic DRIAMS folder (the Version 0.3-0.7 helpers), including an
infeasible rule U and a ledger that sees every row used.
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
from scipy.stats import beta

from src.dataset import CohortSpec, build_dataset, load_dataset
from src.development import DevelopmentError, development_pool
from src.evaluate import choose_threshold
from src.splits import build_splits, load_splits
from src.threshold_rules import (
    ThresholdRuleError,
    apply_shift,
    assert_available,
    delivered,
    drop_seen_patients,
    empirical_cutoff,
    intercept_shift,
    minimum_feasible_n,
    one_per_group,
    order_statistic_k,
    origin_windows,
    tolerance_cutoff,
)
from tests.test_model_scripts_e2e import SMALL_LIGHTGBM, make_config, run_script, write_driams

# ------------------------------------------------------------------------------------------------ the rules

def test_rule_u_needs_29_resistant_at_the_plans_target_and_confidence():
    assert minimum_feasible_n(0.90, 0.95) == 29
    assert order_statistic_k(28, 0.90, 0.95) == 0 and order_statistic_k(29, 0.90, 0.95) == 1
    assert (order_statistic_k(39, 0.90, 0.95), order_statistic_k(63, 0.90, 0.95)) == (1, 3)   # the plan's origins
    ks = [order_statistic_k(n, 0.90, 0.95) for n in range(1, 300)]
    assert ks == sorted(ks)                                                    # never falls as n grows


def test_an_infeasible_rule_u_gives_no_cutoff_rather_than_a_weaker_one():
    cut = tolerance_cutoff(np.linspace(0.1, 0.9, 28), 0.90, 0.95)
    assert cut["feasible"] is False and cut["threshold"] is None and cut["k"] == 0
    assert delivered(np.array([1, 0]), np.array([0.9, 0.1]), None)["sensitivity"] != 0.5   # NaN, not a number


def test_rule_u_holds_its_stated_level_by_simulation_and_rule_e_does_not():
    rng = np.random.default_rng(0)
    for n in (29, 39, 63):
        cut = tolerance_cutoff(rng.random(n), 0.90, 0.95)
        stated = cut["stated_probability"]
        u = np.mean([1 - tolerance_cutoff(rng.random(n), 0.90, 0.95)["threshold"] >= 0.90 for _ in range(4000)])
        e = np.mean([1 - empirical_cutoff(rng.random(n), 0.90)["threshold"] >= 0.90 for _ in range(4000)])
        assert stated >= 0.95 and u >= 0.95 - 0.01, (n, u)    # uniform scores: true sensitivity at t is 1 - t
        assert e < 0.80, (n, e)                               # E targets 0.90 exactly, so it falls short often


def test_rule_e_is_the_protocols_rule_and_never_below_rule_u():
    rng = np.random.default_rng(1)
    for n in (29, 40, 63, 100):
        r = rng.random(n)
        e, u = empirical_cutoff(r, 0.90), tolerance_cutoff(r, 0.90, 0.95)
        rule = {"rule": "min_sensitivity", "min_sensitivity": 0.9}
        assert e["threshold"] == choose_threshold(np.ones(n, int), r, rule)
        assert e["misses_allowed"] == int(np.floor(0.1 * n + 1e-9))
        assert u["threshold"] <= e["threshold"]                             # U flags a superset of E


def test_rule_u_equals_the_clopper_pearson_description_for_tie_free_scores():
    """'The highest cut-off whose one-sided 95 % Clopper-Pearson lower bound is >= 0.90' picks the same cut-off.
    The plan says so; its validity still comes from the order statistics, not from the interval."""
    rng = np.random.default_rng(2)
    for n in (29, 39, 63, 120):
        r = np.sort(rng.random(n))
        best = None
        for i, t in enumerate(r):                              # candidate cut-offs at each resistant score
            caught = n - i
            lower = beta.ppf(0.05, caught, n - caught + 1) if caught < n else 0.05 ** (1 / n)
            if lower >= 0.90:
                best = t
        assert best == tolerance_cutoff(r, 0.90, 0.95)["threshold"], n


# ------------------------------------------------------------------------------------------------ time and patients

def test_windows_respect_the_origin_and_the_label_availability_gap():
    dates = pd.to_datetime(["2016-12-20", "2016-12-24", "2016-12-25", "2016-12-31", "2017-01-01", "2017-06-30",
                            "2017-07-01"]).to_numpy()
    rows = np.arange(7)
    w = origin_windows(dates, rows, "2017-01-01", "2017-07-01", 7)
    assert w["available"].tolist() == [0, 1]                  # before 2016-12-25: labels available at the origin
    assert w["gap"].tolist() == [2, 3]                        # labelled too late to use, and not evaluated either
    assert w["later"].tolist() == [4, 5]                      # the later period; 2017-07-01 belongs to the next
    assert_available(w["available"], dates, "2017-01-01", 7, "fit")
    with pytest.raises(ThresholdRuleError, match="not yet available"):
        assert_available(np.array([0, 2]), dates, "2017-01-01", 7, "fit")


def test_patients_seen_before_an_origin_are_removed_from_its_later_period():
    groups = np.array([10, 11, 12, 11, 13, 10])
    later, removed = drop_seen_patients(np.array([3, 4, 5]), np.array([0, 1, 2]), groups)
    assert later.tolist() == [4] and removed == 2


def test_one_selection_spectrum_per_patient_group_the_earliest():
    dates = pd.to_datetime(["2016-03-01", "2016-01-01", "2016-01-01", "2016-02-01", "2016-05-01"]).to_numpy()
    groups = np.array([1, 1, 2, 2, 3])
    assert one_per_group(np.arange(5), dates, groups).tolist() == [1, 2, 4]   # ties on date: the lower row


def test_the_intercept_shift_recovers_a_known_offset_and_keeps_the_order():
    rng = np.random.default_rng(3)
    logit = rng.normal(-2, 1, 20000)
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(20000) < 1 / (1 + np.exp(-(logit + 0.7)))).astype(int)
    d = intercept_shift(p, y)
    assert abs(d - 0.7) < 0.05
    q = apply_shift(p, d)
    assert (np.argsort(q, kind="stable") == np.argsort(p, kind="stable")).all()   # no cut-off decision can change


# ------------------------------------------------------------------------------------------------ end to end

@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v13")
    root = tmp / "DRIAMS"
    write_driams(root)
    config = make_config(tmp, root)
    build_dataset(config, CohortSpec.from_config(config))
    out, name = Path(config["dataset"]["output_dir"]), config["dataset"]["name"]
    X, meta, _ = load_dataset(out / name)
    del X
    for split_name, split in build_splits(meta, config, name, out).items():
        split.save(out / name / "splits" / f"{split_name}.json", meta)
    card = tmp / "card.json"
    card.write_text(json.dumps({"model_kind": "lightgbm", "params": {"n_estimators": 8, "num_leaves": 4}}),
                    encoding="utf-8")
    config["second_antibiotic"]["families"]["lightgbm_cipro_setting"] = {
        "kind": "lightgbm", "n_jobs": 1, "stochastic": True,
        "fixed": {"subsample_freq": 1, "n_jobs": 1, **SMALL_LIGHTGBM}, "grid": [{"n_estimators": [8]}]}
    tc = config["v13_threshold"]
    tc.update(dataset=name, expected_pool=None, require_clean_tree=False, inner_folds=3, confidence=0.3,
              development_log=str(tmp / "results" / "experiments" / "development_runs.csv"),
              model_dir=str(tmp / "models" / "v1.3"), report_dir=str(tmp / "results" / "metrics" / "v1.3"))
    tc["setting"]["card"] = str(card)
    tc["recalibration"] = {"min_resistant": 2, "min_resistant_groups": 1}
    for o in tc["origins"]:
        o.pop("expected")
    production = Path(config["evaluation"]["test_log"])
    production.parent.mkdir(parents=True, exist_ok=True)
    production.write_text("the production log\n", encoding="utf-8")
    splits = load_splits(out / name / "splits", meta)
    spent = np.unique(np.concatenate([splits[s].test for s in tc["spent_splits"]]))
    pool = development_pool(meta, splits, "random", list(tc["spent_splits"]))
    return SimpleNamespace(tmp=tmp, config=config, meta=meta, spent=spent, pool=pool, production=production,
                           module=importlib.import_module("v13_threshold"), name=name)


def _run(world, config, tag):
    path = world.tmp / f"config_{tag}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    seen = []
    original = world.module.Ledger.use

    def spy(self, what, rows, later, origin):
        seen.append((what, np.asarray(rows).copy(), np.asarray(later).copy(), origin))
        return original(self, what, rows, later, origin)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(world.module.Ledger, "use", spy)
        run = run_script(world.module, SimpleNamespace(root=world.tmp), "--config", path)
    return run, seen


@pytest.fixture(scope="module")
def feasible(world):
    return _run(world, world.config, "feasible")


def test_the_study_runs_and_reports_the_primary_with_its_cost(world, feasible):
    run, _ = feasible
    assert run.code == 0, run.stdout
    report = Path(world.config["v13_threshold"]["report_dir"]) / world.name
    primary = json.loads((report / "primary_result.json").read_text(encoding="utf-8"))
    assert primary["exploratory"] is True and primary["verdict"]
    assert primary["U"]["sensitivity"] >= primary["E"]["sensitivity"]           # U's cut-off is never above E's
    assert "E_minus_U_specificity" in primary and "attainment_probability" in primary
    tables = (report / "tables.md").read_text(encoding="utf-8")
    assert "exploratory" in tables and "heuristic" in tables and "not operational success" not in tables
    assert "no verdict here is operational success" in tables.lower()


def test_nothing_used_comes_from_the_later_period_the_gap_or_a_spent_part(world, feasible):
    _, seen = feasible
    dates = pd.to_datetime(world.meta["acquisition_date"]).to_numpy()
    kinds = {what for what, _, _, _ in seen}
    assert {"fit and calibration", "cut-off selection", "recalibration intercept"} <= kinds
    for what, rows, later, origin in seen:
        assert not np.intersect1d(rows, later).size, what
        assert not np.intersect1d(rows, world.spent).size, what
        assert (dates[rows] < np.datetime64(pd.Timestamp(origin)) - np.timedelta64(7, "D")).all(), what
        assert np.isin(rows, world.pool).all(), what                            # only the development pool


def test_selection_uses_one_spectrum_per_patient_and_later_periods_drop_seen_patients(world, feasible):
    _, seen = feasible
    groups = world.meta["group_id"].to_numpy()
    for what, rows, later, _ in seen:
        if what == "cut-off selection":
            assert np.unique(groups[rows]).size == rows.size
        assert not np.isin(groups[later], groups[rows]).any() or what != "fit and calibration"


def test_the_development_log_gains_only_the_planned_rows_and_production_is_untouched(world, feasible):
    log = pd.read_csv(world.config["v13_threshold"]["development_log"])
    ok = log[log["status"] == "ok"]
    assert len(ok) == 2 * 2 + 2 + 2          # rules per origin, pooled rules, one recalibration row per origin
    keys = list(zip(ok["dataset"], ok["experiment"], ok["model"], ok["seed"], strict=True))
    assert len(set(keys)) == len(keys)
    assert world.production.read_text(encoding="utf-8") == "the production log\n"


def test_an_infeasible_rule_u_is_reported_as_infeasible_not_weakened(world, feasible):
    config = copy.deepcopy(world.config)
    config["v13_threshold"].update(confidence=0.95, report_dir=str(world.tmp / "infeasible"))
    run, _ = _run(world, config, "infeasible")
    assert run.code == 0, run.stdout
    report = Path(config["v13_threshold"]["report_dir"]) / world.name
    primary = json.loads((report / "primary_result.json").read_text(encoding="utf-8"))
    assert primary["verdict"] == "infeasible"
    periods = pd.read_csv(report / "periods.csv")
    assert not periods[periods["rule"] == "U"]["feasible"].any()
    assert "infeasible (no cut-off)" in (report / "tables.md").read_text(encoding="utf-8")


def test_counts_that_differ_from_the_audit_stop_the_run(world, feasible):
    config = copy.deepcopy(world.config)
    config["v13_threshold"]["origins"][0]["expected"] = {"available": 10 ** 6}
    config["v13_threshold"]["report_dir"] = str(world.tmp / "mismatch")
    run, _ = _run(world, config, "mismatch")
    assert run.code == 1
    log = pd.read_csv(world.config["v13_threshold"]["development_log"])
    assert log["status"].iloc[-1].startswith("failed:") and "audit" in log["status"].iloc[-1]


def test_the_ledger_refuses_a_later_row_a_spent_row_and_an_unavailable_label(world):
    dates = pd.to_datetime(world.meta["acquisition_date"]).to_numpy()
    ledger = world.module.Ledger(world.spent, dates, 7)
    early = world.pool[dates[world.pool] < np.datetime64("2016-12-01")][:3]
    with pytest.raises(DevelopmentError, match="later-period"):
        ledger.use("fit", early, early[:1], "2017-01-01")
    with pytest.raises(DevelopmentError, match="spent test part"):
        ledger.use("fit", np.r_[early, world.spent[:1]], np.array([], dtype=np.int64), "2017-01-01")
    late = world.pool[dates[world.pool] >= np.datetime64("2017-03-01")][:1]
    with pytest.raises(ThresholdRuleError, match="not yet available"):
        ledger.use("fit", late, np.array([], dtype=np.int64), "2017-01-01")
