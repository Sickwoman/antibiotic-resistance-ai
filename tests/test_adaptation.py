"""Safety and correctness tests for the Version 0.8 adaptation machinery (synthetic data only).

These are the checks that must hold before DRIAMS-C is opened. Several are adversarial: they build an
invalid configuration or a leaking partition on purpose and assert that the code refuses it, because a
guard that has never been seen to fire is not evidence of anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.adaptation import (
    HARM,
    MIXED,
    NULL,
    SUCCESS,
    AdaptationError,
    PlattRecalibration,
    ZoneCriterion,
    assert_disjoint,
    bootstrap_p_value,
    brier,
    classify,
    eligibility,
    holm,
    partition_fingerprint,
    split_adaptation_and_holdout,
)


def make_cohort(n: int = 300, per_group: int = 2, seed: int = 0) -> pd.DataFrame:
    """A synthetic single-site cohort with patient groups, shaped like a built DRIAMS cohort."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n // per_group), per_group)[:n]
    return pd.DataFrame({
        "site": "DRIAMS-X",
        "year_folder": "2018",
        "code": [f"s{i:05d}" for i in range(n)],
        "label": rng.integers(0, 2, n),
        "group_id": groups,
    })


# --- the primary endpoint -------------------------------------------------------------------------------------

def test_brier_matches_its_mathematical_definition():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.6, 0.3])
    assert brier(y, p) == pytest.approx(float(np.mean((p - y) ** 2)))
    assert brier(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0       # a perfect forecast scores 0
    assert brier(np.array([1, 0]), np.array([0.0, 1.0])) == 1.0       # the worst possible scores 1


@pytest.mark.parametrize("prob", [np.array([0.5, 1.5]), np.array([-0.1, 0.5]), np.array([np.nan, 0.5])])
def test_brier_refuses_values_that_are_not_probabilities(prob):
    with pytest.raises(AdaptationError):
        brier(np.array([0, 1]), prob)


def test_brier_refuses_a_length_mismatch():
    with pytest.raises(AdaptationError):
        brier(np.array([0, 1, 1]), np.array([0.5, 0.5]))


# --- the co-primary zone rule ---------------------------------------------------------------------------------

def test_a_safe_zone_that_covers_almost_nobody_is_not_an_improvement():
    """The gaming case the plan exists to block: NPV bought by shrinking coverage."""
    rule = ZoneCriterion(target_npv=0.95, coverage_slack=0.05)
    # baseline covered 25 %; the "adapted" zone is safer but covers 6 %
    assert not rule.passes(npv=0.99, coverage=0.06, baseline_coverage=0.25)
    # the same NPV while holding coverage is an improvement
    assert rule.passes(npv=0.99, coverage=0.24, baseline_coverage=0.25)
    # exactly on the floor passes; a hair below does not
    assert rule.passes(npv=0.95, coverage=0.20, baseline_coverage=0.25)
    assert not rule.passes(npv=0.95, coverage=0.1999, baseline_coverage=0.25)
    # missing the NPV target fails however wide the zone is
    assert not rule.passes(npv=0.9499, coverage=0.90, baseline_coverage=0.25)


def test_the_zone_rule_explains_which_half_failed():
    rule = ZoneCriterion()
    out = rule.explain(npv=0.99, coverage=0.06, baseline_coverage=0.25)
    assert out["npv_ok"] and not out["coverage_ok"] and not out["passes"]
    assert out["coverage_floor"] == pytest.approx(0.20)


def test_a_non_finite_zone_number_never_passes():
    rule = ZoneCriterion()
    assert not rule.passes(np.nan, 0.5, 0.25)          # no covered rows -> NPV undefined
    assert not rule.passes(0.99, np.nan, 0.25)


# --- the pre-registered verdict -------------------------------------------------------------------------------

def test_the_verdict_is_arithmetic_and_has_exactly_four_outcomes():
    # interval excludes 0 favourably and the zone holds -> success
    assert classify(0.002, 0.010, zone_passes=True, baseline_zone_passes=True, npv_adapted=0.96)[0] == SUCCESS
    # interval includes 0 -> not demonstrated, whatever the zone did
    assert classify(-0.002, 0.010, zone_passes=True, baseline_zone_passes=True, npv_adapted=0.96)[0] == NULL
    assert classify(-0.002, 0.010, zone_passes=False, baseline_zone_passes=True, npv_adapted=0.96)[0] == NULL
    # interval excludes 0 unfavourably -> harm
    assert classify(-0.010, -0.002, zone_passes=True, baseline_zone_passes=True, npv_adapted=0.96)[0] == HARM
    # better Brier bought by a zone that no longer holds -> mixed
    assert classify(0.002, 0.010, zone_passes=False, baseline_zone_passes=False, npv_adapted=0.93)[0] == MIXED


def test_losing_a_zone_the_baseline_held_is_harm_even_with_a_better_brier():
    """Safety dominates: if the baseline could be trusted at 0.95 and the adapted model cannot, that is harm."""
    verdict, reasons = classify(0.002, 0.010, zone_passes=False, baseline_zone_passes=True, npv_adapted=0.90)
    assert verdict == HARM
    assert any("falls below" in r for r in reasons)


def test_the_verdict_refuses_an_undefined_case_instead_of_inventing_one():
    with pytest.raises(AdaptationError):
        classify(np.nan, 0.01, zone_passes=True, baseline_zone_passes=True, npv_adapted=0.96)
    with pytest.raises(AdaptationError):
        classify(0.01, -0.01, zone_passes=True, baseline_zone_passes=True, npv_adapted=0.96)


def test_no_subjective_verdict_exists():
    """There is no 'promising' outcome: the four constants are the whole vocabulary."""
    outcomes = set()
    for low, high in ((0.002, 0.01), (-0.002, 0.01), (-0.01, -0.002)):
        for zp in (True, False):
            for bzp in (True, False):
                outcomes.add(classify(low, high, zp, bzp, 0.96 if zp else 0.90)[0])
    assert outcomes <= {SUCCESS, NULL, HARM, MIXED}


# --- statistics -----------------------------------------------------------------------------------------------

def test_the_bootstrap_p_value_tracks_how_much_of_the_distribution_crosses_zero():
    rng = np.random.default_rng(0)
    assert bootstrap_p_value(rng.normal(0.0, 1.0, 4000)) > 0.5          # centred on 0
    assert bootstrap_p_value(rng.normal(5.0, 1.0, 4000)) < 0.01         # far from 0
    assert bootstrap_p_value(np.array([])) != bootstrap_p_value(np.array([]))   # nan for no draws
    # it can never claim more resolution than the number of draws allows
    assert bootstrap_p_value(np.full(100, 5.0)) >= 1 / 101


def test_holm_is_a_step_down_procedure_over_the_two_confirmatory_arms():
    out = holm({"recalibrated": 0.004, "refit_a_plus_c": 0.03}, alpha=0.05)
    assert out["recalibrated"]["threshold"] == pytest.approx(0.025)     # alpha / 2 for the smallest
    assert out["recalibrated"]["rejected"]
    assert out["refit_a_plus_c"]["threshold"] == pytest.approx(0.05)
    assert out["refit_a_plus_c"]["rejected"]


def test_holm_stops_at_the_first_failure():
    """A step-down procedure must not reject a larger p-value after a smaller one failed."""
    out = holm({"recalibrated": 0.04, "refit_a_plus_c": 0.045}, alpha=0.05)
    assert not out["recalibrated"]["rejected"]      # 0.04 > 0.025
    assert not out["refit_a_plus_c"]["rejected"]    # so this one cannot be rejected either
    assert out["refit_a_plus_c"]["p_value"] == 0.045


# --- the partition --------------------------------------------------------------------------------------------

def test_the_partition_is_deterministic_and_reproducible():
    meta = make_cohort()
    rows = np.arange(len(meta))
    a1, h1 = split_adaptation_and_holdout(meta, rows, 0.70, seed=42)
    a2, h2 = split_adaptation_and_holdout(meta, rows, 0.70, seed=42)
    assert np.array_equal(a1, a2) and np.array_equal(h1, h2)
    assert partition_fingerprint(meta, a1) == partition_fingerprint(meta, a2)


def test_a_different_seed_gives_a_different_partition():
    meta = make_cohort()
    rows = np.arange(len(meta))
    a42, _ = split_adaptation_and_holdout(meta, rows, 0.70, seed=42)
    a43, _ = split_adaptation_and_holdout(meta, rows, 0.70, seed=43)
    assert not np.array_equal(a42, a43), "the seed must actually drive the partition"


def test_the_partition_covers_every_row_exactly_once_and_respects_the_fraction():
    meta = make_cohort(n=300)
    rows = np.arange(len(meta))
    adapt, hold = split_adaptation_and_holdout(meta, rows, 0.70, seed=42)
    assert np.array_equal(np.union1d(adapt, hold), rows)        # nothing lost
    assert adapt.size + hold.size == rows.size                  # nothing duplicated
    assert 0.69 <= adapt.size / rows.size <= 0.71               # whole groups, so approximately 70 %


def test_no_patient_group_spans_the_adaptation_and_held_out_parts():
    meta = make_cohort(n=300, per_group=3)
    adapt, hold = split_adaptation_and_holdout(meta, np.arange(len(meta)), 0.70, seed=42)
    g = meta["group_id"].to_numpy()
    assert np.intersect1d(np.unique(g[adapt]), np.unique(g[hold])).size == 0


def test_a_leaking_partition_is_refused_loudly():
    """Adversarial: hand the guard a partition that shares a group and check that it raises."""
    meta = make_cohort(n=100, per_group=2)
    g = meta["group_id"].to_numpy()
    first_group = np.where(g == g[0])[0]
    adapt, hold = first_group[:1], first_group[1:]              # same patient, both sides
    with pytest.raises(AdaptationError, match="patient group"):
        assert_disjoint(meta, adapt, hold)
    with pytest.raises(AdaptationError, match="row"):
        assert_disjoint(meta, np.array([0, 1]), np.array([1, 2]))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.5])
def test_an_invalid_partition_fraction_fails_loudly(fraction):
    meta = make_cohort(n=50)
    with pytest.raises(AdaptationError):
        split_adaptation_and_holdout(meta, np.arange(len(meta)), fraction, seed=42)


def test_the_partition_fingerprint_identifies_the_samples_not_their_order():
    meta = make_cohort(n=60)
    rows = np.arange(30)
    assert partition_fingerprint(meta, rows) == partition_fingerprint(meta, rows[::-1])
    assert partition_fingerprint(meta, rows) != partition_fingerprint(meta, np.arange(30, 60))


# --- the eligibility gate -------------------------------------------------------------------------------------

def test_both_eligibility_gates_must_pass():
    ok = eligibility({"resistant": 150, "susceptible": 400}, 30, 30,
                     {"resistant": 45, "susceptible": 120})
    assert ok["eligible"] and ok["stop_reason"] is None

    small_site = eligibility({"resistant": 12, "susceptible": 400}, 30, 30,
                             {"resistant": 4, "susceptible": 120})
    assert not small_site["eligible"] and small_site["stop_reason"].startswith("E1")

    weak_holdout = eligibility({"resistant": 60, "susceptible": 400}, 30, 30,
                               {"resistant": 18, "susceptible": 120})
    assert not weak_holdout["eligible"] and weak_holdout["stop_reason"].startswith("E2")


def test_the_eligibility_gate_reports_the_counts_it_judged_on():
    out = eligibility({"resistant": 10, "susceptible": 10}, 30, 30, {"resistant": 3, "susceptible": 3})
    assert out["cohort_counts"] == {"resistant": 10, "susceptible": 10}
    assert out["holdout_counts"] == {"resistant": 3, "susceptible": 3}
    assert out["e1_min_per_class"] == 30 and out["e2_min_per_class"] == 30


# --- recalibration --------------------------------------------------------------------------------------------

def test_the_recalibration_reproduces_scikit_learns_own_sigmoid():
    from sklearn.calibration import _SigmoidCalibration

    rng = np.random.default_rng(42)
    raw = rng.uniform(0.01, 0.99, 400)
    y = (rng.uniform(size=400) < raw).astype(int)
    fitted = _SigmoidCalibration().fit(raw, y)
    ours = PlattRecalibration(float(fitted.a_), float(fitted.b_), y.size, int(y.sum())).apply(raw)
    assert np.allclose(ours, fitted.predict(raw), atol=1e-12)


def test_recalibration_cannot_change_the_ranking_and_therefore_cannot_change_auroc():
    """The fact that decided the primary endpoint: AUROC is unreachable by this intervention."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(7)
    raw = rng.uniform(0.01, 0.99, 500)
    y = (rng.uniform(size=500) < raw).astype(int)
    before = roc_auc_score(y, raw)
    for a, b in ((-1.0, 0.5), (-2.0, -0.3), (-0.5, 1.2)):
        after = roc_auc_score(y, PlattRecalibration(a, b, 500, int(y.sum())).apply(raw))
        assert after == pytest.approx(before, abs=1e-12)


def test_recalibration_records_what_moved_and_what_stayed_frozen():
    d = PlattRecalibration(-1.2, 0.3, 200, 60).to_dict()
    assert d["changed"] == ["calibrator_a", "calibrator_b"]
    assert set(d["frozen"]) == {"model", "preprocessing", "features", "threshold_rule"}
    assert d["method"] == "platt_sigmoid" and d["fitted_on"] == "adaptation"
