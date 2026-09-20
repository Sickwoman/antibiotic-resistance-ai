"""Tests for the Version 0.6 confidence zones (synthetic probabilities, no data files needed).

The rule is pre-registered in docs/v0.6_explainability_plan.md, so these tests fix its behaviour: the
edges are the extreme cuts that still meet their target at the required coverage, a side that cannot meet
its target produces no zone (and reports what it does reach), and the model's own cut-off never moves.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.uncertainty import (
    RESISTANT,
    SUSCEPTIBLE,
    UNCERTAIN,
    UncertaintyError,
    ZoneRule,
    Zones,
    bootstrap_zone_metrics,
    fit_side,
    fit_zones,
    side_curve,
    zone_metrics,
)


def separable(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities that are almost right: the extremes are pure, the middle is mixed."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.int64)
    y[: n // 4] = 1
    prob = np.where(y == 1, rng.uniform(0.55, 0.99, n), rng.uniform(0.01, 0.45, n))
    muddle = rng.choice(n, size=n // 10, replace=False)      # a tenth of the rows land in the middle
    prob[muddle] = rng.uniform(0.45, 0.55, muddle.size)
    return y, prob


# --- the rule ---------------------------------------------------------------------------------------------------

def test_rule_rejects_impossible_targets():
    for bad in ({"target_npv": 0.4}, {"target_precision": 1.5}, {"min_coverage": 0.0}, {"min_coverage": 2}):
        with pytest.raises(UncertaintyError):
            ZoneRule.from_config(bad)


def test_rule_defaults_are_the_preregistered_ones():
    rule = ZoneRule.from_config({})
    assert (rule.target_npv, rule.target_precision, rule.min_coverage) == (0.95, 0.95, 0.05)


def test_candidate_cuts_stay_on_their_own_side_of_the_threshold():
    y, prob = separable()
    low = side_curve(y, prob, 0.5, "susceptible")
    high = side_curve(y, prob, 0.5, "resistant")
    assert all(row["cut"] <= 0.5 for row in low)
    assert all(row["cut"] >= 0.5 for row in high)
    assert side_curve(y, prob, 0.5, "susceptible")[0]["side"] == "susceptible"
    with pytest.raises(UncertaintyError):
        side_curve(y, prob, 0.5, "sideways")


def test_susceptible_edge_is_the_largest_cut_that_keeps_its_promise():
    # 10 rows: the four lowest probabilities are susceptible, then one resistant at 0.30.
    prob = np.array([0.01, 0.02, 0.03, 0.04, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    side = fit_side(y, prob, 0.5, "susceptible", target=1.0, min_coverage=0.1)
    assert side.exists and side.edge == pytest.approx(0.30)   # 0.30 covers exactly the four susceptible rows
    assert side.n_covered == 4 and side.coverage == pytest.approx(0.4)
    assert side.achieved == pytest.approx(1.0)


def test_resistant_edge_is_the_smallest_cut_that_keeps_its_promise():
    prob = np.array([0.10, 0.11, 0.12, 0.13, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    side = fit_side(y, prob, 0.5, "resistant", target=1.0, min_coverage=0.1)
    assert side.exists and side.edge == pytest.approx(0.60)   # everything above 0.60 is resistant
    assert side.n_covered == 5 and side.achieved == pytest.approx(1.0)


def test_a_side_that_cannot_reach_its_target_reports_no_zone_and_says_what_it_reaches():
    # Every high probability is a coin flip, so no cut can be 95 % precise.
    prob = np.linspace(0.5, 0.99, 20)
    y = np.tile([0, 1], 10)
    side = fit_side(y, prob, 0.5, "resistant", target=0.95, min_coverage=0.1)
    assert not side.exists and side.edge is None
    assert side.n_covered == 0 and side.coverage == 0.0
    assert np.isnan(side.achieved)
    assert 0 < side.best_achieved < 0.95              # the honest fallback: what it does reach
    assert side.best_edge is not None and side.best_coverage >= 0.1


def test_coverage_floor_rejects_a_tiny_but_pure_zone():
    prob = np.concatenate([[0.99], np.full(99, 0.30)])
    y = np.concatenate([[1], np.zeros(99, dtype=np.int64)])
    pure = fit_side(y, prob, 0.5, "resistant", target=0.95, min_coverage=0.005)
    assert pure.exists and pure.n_covered == 1        # one perfect row is enough when 0.5 % is allowed
    floored = fit_side(y, prob, 0.5, "resistant", target=0.95, min_coverage=0.05)
    assert not floored.exists                        # the same row covers 1 %, below the 5 % floor


# --- fitting both sides -----------------------------------------------------------------------------------------

def test_fit_zones_brackets_the_threshold_and_records_what_it_was_fitted_on():
    y, prob = separable()
    zones, low, high, curve = fit_zones(y, prob, 0.5, ZoneRule())
    assert zones.lower is None or zones.lower <= 0.5
    assert zones.upper is None or zones.upper >= 0.5
    assert zones.fitted_on == "validation" and zones.n_fitted == y.size
    assert {row["side"] for row in curve} == {"susceptible", "resistant"}
    assert low.side == "susceptible" and high.side == "resistant"


def test_fit_zones_refuses_input_it_cannot_describe():
    y, prob = separable()
    with pytest.raises(UncertaintyError):
        fit_zones(y[:10], prob, 0.5, ZoneRule())
    with pytest.raises(UncertaintyError):
        fit_zones(np.zeros(20, dtype=np.int64), np.full(20, 0.3), 0.5, ZoneRule())
    with pytest.raises(UncertaintyError):
        fit_zones(y, prob, 1.5, ZoneRule())


# --- labelling --------------------------------------------------------------------------------------------------

def test_labels_follow_the_edges_and_are_strict_inequalities():
    zones = Zones(0.5, 0.2, 0.8, ZoneRule())
    assert list(zones.label([0.19, 0.2, 0.5, 0.8, 0.81])) == [SUSCEPTIBLE, UNCERTAIN, UNCERTAIN, UNCERTAIN,
                                                              RESISTANT]
    assert zones.one(0.05) == SUSCEPTIBLE


def test_a_missing_edge_means_that_zone_does_not_exist():
    assert list(Zones(0.5, 0.2, None, ZoneRule()).label([0.05, 0.99])) == [SUSCEPTIBLE, UNCERTAIN]
    assert list(Zones(0.5, None, 0.8, ZoneRule()).label([0.05, 0.99])) == [UNCERTAIN, RESISTANT]
    assert list(Zones(0.5, None, None, ZoneRule()).label([0.05, 0.5, 0.99])) == [UNCERTAIN] * 3


def test_labels_refuse_values_that_are_not_probabilities():
    zones = Zones(0.5, 0.2, 0.8, ZoneRule())
    for bad in ([1.2], [-0.1], [np.nan]):
        with pytest.raises(UncertaintyError):
            zones.label(bad)


# --- saving and reading back ------------------------------------------------------------------------------------

def test_zones_round_trip_through_json(tmp_path):
    y, prob = separable()
    zones, _, _, _ = fit_zones(y, prob, 0.5, ZoneRule())
    path = tmp_path / "uncertainty.json"
    path.write_text(json.dumps(zones.to_dict()), encoding="utf-8")
    again = Zones.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert (again.lower, again.upper, again.threshold) == (zones.lower, zones.upper, zones.threshold)
    assert again.rule == zones.rule and again.n_fitted == zones.n_fitted


def test_from_dict_refuses_a_record_that_is_not_zones_or_is_inside_out():
    with pytest.raises(UncertaintyError):
        Zones.from_dict({"threshold": 0.5, "lower": 0.2})
    with pytest.raises(UncertaintyError):
        Zones.from_dict({"threshold": 0.5, "lower": 0.9, "upper": 0.1})


# --- what the zones do on a set of rows -------------------------------------------------------------------------

def test_zone_metrics_counts_add_up_and_keep_the_isolate_count_separate():
    y = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.01, 0.02, 0.30, 0.45, 0.90, 0.95])
    out = zone_metrics(y, prob, Zones(0.4, 0.2, 0.8, ZoneRule()))
    assert out["n"] == 6 and out["n_resistant"] == 3            # isolates, not a zone count
    assert out["n_zone_susceptible"] + out["n_zone_uncertain"] + out["n_zone_resistant"] == 6
    assert out["n_zone_susceptible"] == 2 and out["n_zone_resistant"] == 2
    assert out["npv_susceptible"] == pytest.approx(1.0)
    assert out["precision_resistant"] == pytest.approx(1.0)
    assert out["confident_share"] == pytest.approx(4 / 6)
    assert out["accuracy_confident"] == pytest.approx(1.0)


def test_zone_metrics_reports_not_a_number_for_an_empty_zone():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.30, 0.35, 0.40, 0.45])
    out = zone_metrics(y, prob, Zones(0.4, None, None, ZoneRule()))
    assert out["n_zone_uncertain"] == 4 and out["confident_share"] == 0.0
    assert np.isnan(out["npv_susceptible"]) and np.isnan(out["accuracy_confident"])


def test_a_wrong_confident_call_lowers_the_confident_accuracy():
    y = np.array([1, 0, 0, 1])                                   # the lowest probability is resistant: wrong
    prob = np.array([0.01, 0.02, 0.03, 0.95])
    out = zone_metrics(y, prob, Zones(0.5, 0.1, 0.8, ZoneRule()))
    assert out["n_zone_susceptible"] == 3 and out["n_zone_resistant"] == 1
    assert out["accuracy_confident"] == pytest.approx(0.75)      # three of four confident calls are right
    assert out["npv_susceptible"] == pytest.approx(2 / 3)
    assert out["precision_resistant"] == pytest.approx(1.0)


# --- intervals --------------------------------------------------------------------------------------------------

def test_bootstrap_intervals_bracket_the_point_estimate_and_respect_groups():
    y, prob = separable(n=160, seed=3)
    groups = np.repeat(np.arange(80), 2)                         # two rows per patient
    out = bootstrap_zone_metrics(y, groups, prob, Zones(0.5, 0.2, 0.8, ZoneRule()), resamples=200, seed=7)
    for key in ("share_susceptible", "confident_share"):
        assert out[key]["low"] <= out[key]["value"] <= out[key]["high"]
        assert out[key]["draws"] > 0
    assert out["confident_share"]["value"] == pytest.approx(
        zone_metrics(y, prob, Zones(0.5, 0.2, 0.8, ZoneRule()))["confident_share"])


def test_bootstrap_refuses_a_group_per_row_mismatch():
    y, prob = separable(n=40)
    with pytest.raises(UncertaintyError):
        bootstrap_zone_metrics(y, np.arange(10), prob, Zones(0.5, 0.2, 0.8, ZoneRule()), resamples=5)
