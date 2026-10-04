"""Version 2.0's scoring code (amendment F5), on synthetic numbers only: no MARISMa label, no model, no real log.

The production log is never touched: every log here is a temporary copy. The evaluation module's rules, the run's
stages, its failure handling and its log gate are each exercised; so is the guard that stops production prediction
and label access without the owner's recorded authorisation.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.evaluate import TEST_LOG_COLUMNS, EvaluationError
from src.uncertainty import ZoneRule, Zones
from src.v20_evaluation import (
    NOT_ESTIMABLE,
    brunner_munzel_parts,
    evaluate_population,
    family,
    gap,
    labels,
    linkage_robustness,
    primary,
    zone_reading,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v20_score_marisma.py"
THRESHOLD, LOWER = 0.14261540693905073, 0.10255963637363619
ZONES = Zones(THRESHOLD, LOWER, None, ZoneRule())


@pytest.fixture(scope="module")
def score():
    spec = importlib.util.spec_from_file_location("v20_score", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic(n_r=120, n_s=480, shift=1.0, seed=1):
    rng = np.random.default_rng(seed)
    p = 1 / (1 + np.exp(-np.r_[rng.normal(-1.0 + shift, 1, n_r), rng.normal(-2.3, 1, n_s)]))
    return np.r_[np.ones(n_r, int), np.zeros(n_s, int)], p


# --- labels ---------------------------------------------------------------------------------------------------------

def test_the_label_rule_maps_r_and_i_to_one_and_counts_the_rest():
    tokens = {"a": "R", "b": " i ", "c": "S", "d": "s", "e": "SDD", "f": "NR"}
    y, counts = labels(tokens)
    assert y == {"a": 1, "b": 1, "c": 0, "d": 0}
    assert counts["by_value"] == {"I": 1, "R": 1, "S": 2, "other (excluded)": 2}
    assert counts["excluded_other_values"] == {"NR": 1, "SDD": 1}
    y_i, counts_i = labels(tokens, exclude_intermediate=True)
    assert y_i == {"a": 1, "c": 0, "d": 0} and counts_i["by_value"]["I excluded"] == 1


# --- the primary test, the family and the wording -------------------------------------------------------------------

def test_brunner_munzel_parts_reproduce_scipy():
    y, p = synthetic()
    r, s = p[y == 1], p[y == 0]
    statistic, df = brunner_munzel_parts(r, s)
    expected = stats.brunnermunzel(r, s, alternative="greater", distribution="t")
    assert statistic == pytest.approx(expected.statistic, rel=1e-12)
    assert stats.t.sf(-statistic, df) == pytest.approx(expected.pvalue, rel=1e-9)


def test_linkage_robustness_doubles_the_variance():
    y, p = synthetic(shift=0.6)
    r, s = p[y == 1], p[y == 0]
    statistic, df = brunner_munzel_parts(r, s)
    out = linkage_robustness(r, s)
    assert out["statistic"] == pytest.approx(statistic / math.sqrt(2))
    assert out["p_value"] == pytest.approx(stats.t.sf(-statistic / math.sqrt(2), df))
    assert out["p_value"] > stats.t.sf(-statistic, df)                 # weaker evidence, never stronger


def test_the_dropped_antibiotic_enters_holm_with_p_one():
    y, p = synthetic()
    cip = primary(y, p)
    fam = family(cip, None)
    assert fam["holm_inputs"] == {"ciprofloxacin": cip["holm_input"], "ceftriaxone": 1.0}
    assert fam["adjusted"]["ciprofloxacin"] == pytest.approx(min(1.0, 2 * cip["holm_input"]))
    assert fam["conclusions"]["ceftriaxone"] == "unavailable"
    assert fam["error_control"].startswith("Brunner–Munzel inference is approximate")


def test_holm_rejection_and_wording():
    y, p = synthetic(shift=1.5)
    fam = family(primary(y, p), None)
    assert fam["rejected"]["ciprofloxacin"]
    assert fam["conclusions"]["ciprofloxacin"] == (
        "Evidence of above-chance ranking on MARISMa under the isolate-independence assumption.")
    y0, p0 = synthetic(shift=-1.3, seed=3)                                # no signal at all
    assert family(primary(y0, p0), None)["conclusions"]["ciprofloxacin"] == (
        "Above-chance ranking on MARISMa not demonstrated.")


def test_degenerate_scores_are_not_estimable():
    y = np.r_[np.ones(60, int), np.zeros(60, int)]
    p = np.r_[np.full(60, 0.9), np.full(60, 0.1)]                         # complete separation
    res = primary(y, p)
    assert res["status"] == NOT_ESTIMABLE and res["holm_input"] == 1.0 and res["auroc"] == 1.0
    assert family(res, None)["conclusions"]["ciprofloxacin"] == (
        "Primary test not estimable; above-chance ranking on MARISMa not demonstrated.")


def test_below_the_minimum_class_count_the_reading_is_descriptive():
    y, p = synthetic(n_r=49, n_s=400, shift=2.0)
    res = primary(y, p)
    assert res["status"] == "tested" and not res["confirmatory"] and res["holm_input"] == 1.0
    assert res["reading"].startswith("descriptive")
    assert not family(res, None)["rejected"]["ciprofloxacin"]           # p = 1: never rejected


# --- intervals, zone, undefined metrics -----------------------------------------------------------------------------

def test_intervals_are_stratified_and_cover_the_estimate():
    y, p = synthetic()
    out = evaluate_population(y, p, THRESHOLD, ZONES, test=True, resamples=300)
    auc = out["metrics"]["roc_auc"]
    assert auc["low"] <= auc["estimate"] <= auc["high"] and auc["undefined_resamples"] == 0
    assert out["_auroc_draws"].size == 300
    assert out["pr_auc_no_skill"] == pytest.approx(y.mean())
    assert out["metrics"]["brier_base_rate"]["estimate"] == pytest.approx(y.mean() * (1 - y.mean()))


def test_an_empty_zone_has_coverage_zero_and_npv_not_estimable():
    y, p = synthetic()
    p = np.clip(p, LOWER + 1e-6, 1)                                      # nothing below the zone's edge
    out = evaluate_population(y, p, THRESHOLD, ZONES, test=False, resamples=100)
    assert out["metrics"]["zone_coverage"]["estimate"] == 0.0
    assert out["metrics"]["zone_npv"]["estimate"] == NOT_ESTIMABLE
    assert out["zone"]["point"] == NOT_ESTIMABLE and out["zone"]["n_in_zone"] == 0


def test_more_than_20_undefined_resamples_make_an_interval_not_estimable():
    y = np.r_[np.ones(60, int), np.zeros(300, int)]
    p = np.r_[np.full(59, 0.05), [0.5], np.full(300, 0.05)]              # one isolate flagged at the cut-off
    out = evaluate_population(y, p, THRESHOLD, ZONES, test=False, resamples=300)
    precision = out["metrics"]["precision"]                              # undefined when nothing is flagged
    assert precision["status"] == NOT_ESTIMABLE and precision["undefined_resamples"] > 20


def test_zone_readings_are_literal():
    assert zone_reading(0.97, {"low": 0.955, "high": 0.99})["interval"] == "lies above 0.95"
    reading = zone_reading(0.94, {"low": 0.90, "high": 0.97})
    assert reading["point"] == "misses 0.95" and reading["interval"] == "contains 0.95"
    assert reading["version_0_7_reading"].startswith("the point estimate meets 0.95, or")
    assert zone_reading(0.90, {"low": 0.85, "high": 0.93})["version_0_7_reading"] == "neither"


def test_a_population_lacking_a_class_is_not_estimable_without_a_p_value():
    y, p = np.zeros(40, int), np.linspace(0.01, 0.6, 40)
    out = evaluate_population(y, p, THRESHOLD, ZONES, test=False, resamples=50)
    assert out["metrics"]["roc_auc"]["status"] == NOT_ESTIMABLE and "primary" not in out


def test_the_gap_is_demonstrated_only_if_its_interval_excludes_zero():
    yi, pi = synthetic(n_r=150, n_s=600, shift=1.6, seed=5)
    groups = np.arange(yi.size) // 2                                     # pairs of isolates share a patient
    ye, pe = synthetic(shift=0.1, seed=6)
    ext = evaluate_population(ye, pe, THRESHOLD, ZONES, test=False, resamples=300)
    g = gap(yi, pi, groups, THRESHOLD, ext["metrics"]["roc_auc"]["estimate"], ext["_auroc_draws"], resamples=300)
    assert g["demonstrated"] and g["low"] > 0
    same = gap(ye, pe, np.arange(ye.size), THRESHOLD, ext["metrics"]["roc_auc"]["estimate"], ext["_auroc_draws"],
               resamples=300)
    assert not same["demonstrated"] and "never means performance was maintained" in same["reading"]


# --- the run: stages, failures, the log gate ------------------------------------------------------------------------

def make_log(path: Path) -> tuple[str, int]:
    rows = [{c: (0 if c not in ("logged_at", "git_commit", "stage", "dataset", "dataset_fingerprint", "x_sha256",
                                "experiment", "split", "model") else f"x{i}") for c in TEST_LOG_COLUMNS}
            for i in range(3)]
    pd.DataFrame(rows, columns=TEST_LOG_COLUMNS).to_csv(path, index=False, lineterminator="\n")
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest(), 3


def make_run(score, tmp_path, *, predict=None, read_labels=None, log_path=None):
    rng = np.random.default_rng(9)
    ids = [f"c{i:04d}" for i in range(400)]
    truth = {i: ("R" if k < 90 else "I" if k < 110 else "S") for k, i in enumerate(ids)}
    X = rng.normal(size=(400, 6000)).astype(np.float32)
    sources = {i: ("ambiguous" if k % 7 == 0 else "clinical") for k, i in enumerate(ids)}
    base = np.array([0.35 if truth[i] in ("R", "I") else 0.12 for i in ids])

    def fake_predict(X_):
        assert X_.shape == (400, 6000)
        return np.clip(base + rng.normal(0, 0.08, size=400), 0.001, 0.999)

    def fake_labels(chosen):
        tokens = {i: truth[i] for i in chosen[:380]}                     # 20 isolates have no record
        tokens[chosen[0]] = "conflicting"
        return tokens, {"cohort": len(chosen), "matched": 380, "unmatched": 20, "missing": 0, "conflicting": 1}

    yi, pi = synthetic(n_r=100, n_s=400, seed=11)
    internal = {"y": yi, "p": pi, "groups": np.arange(yi.size), "logged_auroc": 0.75, "reproduced_auroc": 0.75,
                "n": int(yi.size), "n_resistant": int(yi.sum())}
    log_path = log_path or tmp_path / "log.csv"
    expected = make_log(log_path) if not log_path.exists() else None
    return score.Run(tmp_path / "work", log_path, tmp_path / "report.json", ids=ids, X=X, sources=sources,
                     zones=ZONES, internal=internal, predict=predict or fake_predict,
                     read_labels=read_labels or fake_labels, log_expected=expected, commit="test", resamples=200)


def test_a_complete_run_appends_two_rows_and_writes_an_aggregate_report(score, tmp_path):
    run = make_run(score, tmp_path)
    before = (tmp_path / "log.csv").read_bytes()
    result = run.execute()
    after = pd.read_csv(tmp_path / "log.csv")
    assert (tmp_path / "log.csv").read_bytes().startswith(before) and len(after) == 5
    assert set(after.tail(2)["dataset"]) == {"marisma_ecoli_ciprofloxacin",
                                             "marisma_ecoli_ciprofloxacin__intermediate-exclude"}
    assert json.loads(run.state_path.read_text())["stage"] == "complete"
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["holm_family"]["holm_inputs"]["ceftriaxone"] == 1.0
    assert report["populations"]["primary_analysis"]["by_value"] == {"I": 20, "R": 89, "S": 270}
    assert report["populations"]["conflicting_interpretations"] == 1
    assert report["sensitivity_analyses"]["all_sources"].startswith("identical to the primary cohort")
    assert "c0001" not in (tmp_path / "report.json").read_text()       # no identifier in the report
    assert result["sensitivity_analyses"]["by_period"].startswith("unavailable")


def test_a_second_run_refuses_and_nothing_is_rescored(score, tmp_path):
    run = make_run(score, tmp_path)
    run.execute()
    calls = []
    again = make_run(score, tmp_path, predict=lambda X: calls.append(1), log_path=tmp_path / "log.csv")
    with pytest.raises(score.RunError):
        again.execute()
    assert calls == []


def test_a_failure_records_its_stage_keeps_the_predictions_and_stops(score, tmp_path):
    def broken_labels(_ids):
        raise OSError("disk unavailable")
    run = make_run(score, tmp_path, read_labels=broken_labels)
    before = (tmp_path / "log.csv").read_bytes()
    with pytest.raises(OSError):
        run.execute()
    state = json.loads(run.state_path.read_text())
    assert state["stage"] == "failed" and state["failed_stage"] == "labels"
    assert (run.work / "predictions.npz").is_file() and state["predictions_sha256"]
    assert (tmp_path / "log.csv").read_bytes() == before                 # nothing appended
    with pytest.raises(score.RunError):
        score.rebuild_report(run)                                       # not a report failure: no rebuild


def test_a_report_failure_is_rebuilt_without_predicting_or_appending(score, tmp_path, monkeypatch):
    run = make_run(score, tmp_path)
    original = score.Run._report
    monkeypatch.setattr(score.Run, "_report", lambda self, result: (_ for _ in ()).throw(OSError("full disk")))
    with pytest.raises(OSError):
        run.execute()
    assert json.loads(run.state_path.read_text())["failed_stage"] == "report"
    appended = (tmp_path / "log.csv").read_bytes()
    monkeypatch.setattr(score.Run, "_report", original)
    run.predict = score._never                                           # rebuilding must never predict
    run.read_labels = score._never_labels
    score.rebuild_report(run)
    assert (tmp_path / "log.csv").read_bytes() == appended               # and never appends again
    assert json.loads(run.state_path.read_text())["stage"] == "complete"
    assert (tmp_path / "report.json").is_file()


def test_the_strict_gate_refuses_a_duplicate_key(score, tmp_path):
    log_path = tmp_path / "log.csv"
    make_log(log_path)
    frame = pd.read_csv(log_path)
    frame.loc[0, ["dataset", "experiment", "model", "seed"]] = list(score.KEYS[0])
    frame.to_csv(log_path, index=False, lineterminator="\n")
    import hashlib
    run = make_run(score, tmp_path, log_path=log_path)
    run.log_expected = (hashlib.sha256(log_path.read_bytes()).hexdigest(), 3)
    before = log_path.read_bytes()
    with pytest.raises(EvaluationError):
        run.execute()
    assert log_path.read_bytes() == before
    assert json.loads(run.state_path.read_text())["failed_stage"] == "append"


# --- guards: no production prediction or label access without the owner's record ----------------------------------

def test_production_refuses_without_a_recorded_authorisation(score, monkeypatch):
    monkeypatch.setattr(score, "production_predict", lambda X: pytest.fail("predicted without authorisation"))
    assert score.main(["--production"]) == 2


def test_the_label_reader_refuses_the_sealed_file_without_authorisation(tmp_path):
    from src.marisma_labels import parse_interpretations, read_interpretations
    from src.v20_scoring_guard import GuardError
    sealed = tmp_path / "X_sealed" / "AMR.csv"
    sealed.parent.mkdir()
    sealed.write_text("Identifier,MIC_Ciprofloxacin,Ciprofloxacin\na,0.5,S\n", encoding="utf-8")
    with pytest.raises(GuardError):
        parse_interpretations(sealed, ["a"], "Ciprofloxacin")
    with pytest.raises(GuardError):
        read_interpretations(sealed, ["a"], "Ciprofloxacin")
    plain = tmp_path / "synthetic.csv"
    plain.write_text("Identifier,MIC_Ciprofloxacin,Ciprofloxacin\na,0.5,S\na,0.5,S\nb,1,R\nb,2,S\nc,,\n",
                     encoding="utf-8")
    tokens, counts = parse_interpretations(plain, ["a", "b", "c", "d"], "Ciprofloxacin")
    assert tokens == {"a": "S", "b": "conflicting", "c": "missing"}
    assert counts == {"cohort": 4, "matched": 3, "unmatched": 1, "missing": 1, "conflicting": 1}


def test_prediction_and_label_access_live_only_in_the_production_path():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    owners = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for inner in ast.walk(node):
                name = inner.attr if isinstance(inner, ast.Attribute) else inner.id if isinstance(inner, ast.Name) \
                    else None
                if name in ("predict_features", "load_bundle", "read_interpretations"):
                    owners.setdefault(name, set()).add(node.name)
    assert owners == {"predict_features": {"production_predict"}, "load_bundle": {"production_predict"},
                      "read_interpretations": {"production_run", "read_labels"}}
    others = [p for folder in ("src", "scripts") for p in (ROOT / folder).rglob("*.py")
              if "read_interpretations" in p.read_text(encoding="utf-8")]
    assert {p.relative_to(ROOT).as_posix() for p in others} == {"src/marisma_labels.py",
                                                                 "scripts/v20_score_marisma.py"}


def test_the_guard_refuses_without_an_authorisation_block():
    from src.v20_scoring_guard import GuardError, recorded_authorisation, verify
    assert recorded_authorisation("no block here") is None
    with pytest.raises(GuardError):
        verify(ROOT)
