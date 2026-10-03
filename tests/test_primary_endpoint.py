"""Version 2.0's primary-test rule (docs/v2.0_marisma_plan.md, amendment B), on synthetic scores only.

These tests fix the behaviour before any data exist:
- the direction of the one-sided test;
- that a non-finite Brunner-Munzel statistic or p-value is "not estimable" with exactly p = 1, with no fallback;
- that descriptive metrics are kept;
- that the Holm family is fixed to the two antibiotics;
- the exact conclusion sentences.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.primary_endpoint import (
    CONCLUSION_NOT_ESTIMABLE,
    CONCLUSION_NOT_REJECTED,
    CONCLUSION_REJECTED,
    FAMILY,
    NOT_ESTIMABLE,
    TESTED,
    PrimaryTest,
    conclusion,
    holm,
    primary_test,
)


def test_resistant_scores_larger_give_a_small_p():
    rng = np.random.default_rng(42)
    t = primary_test(rng.normal(0.8, 1.0, 60), rng.normal(0.0, 1.0, 60))
    assert t.status == TESTED
    assert t.p_value < 0.01
    assert t.auroc > 0.5


def test_resistant_scores_smaller_give_a_large_p():
    rng = np.random.default_rng(43)
    t = primary_test(rng.normal(-0.8, 1.0, 60), rng.normal(0.0, 1.0, 60))
    assert t.status == TESTED
    assert t.p_value > 0.99
    assert t.auroc < 0.5


@pytest.mark.parametrize(("resistant", "susceptible", "auroc"), [
    (np.linspace(0.6, 0.9, 50), np.linspace(0.1, 0.4, 50), 1.0),   # complete separation, resistant above
    (np.linspace(0.1, 0.4, 50), np.linspace(0.6, 0.9, 50), 0.0),   # complete separation, resistant below
    (np.full(50, 0.3), np.full(50, 0.3), 0.5),                     # every score tied
    (np.full(50, 0.7), np.full(50, 0.2), 1.0),                     # each class one constant score
])
def test_a_non_finite_result_is_not_estimable_with_p_one(resistant, susceptible, auroc):
    t = primary_test(resistant, susceptible)
    assert t.status == NOT_ESTIMABLE
    assert t.p_value == 1.0                   # exactly 1: no permutation or other fallback p-value
    assert t.auroc == pytest.approx(auroc)    # the descriptive metric is kept where it is defined


def test_a_single_resistant_isolate_is_not_estimable():
    t = primary_test([0.5], np.random.default_rng(44).normal(0.0, 1.0, 50))
    assert t.status == NOT_ESTIMABLE
    assert t.p_value == 1.0
    assert math.isfinite(t.auroc)


def test_an_empty_class_is_not_estimable_and_has_no_auroc():
    t = primary_test([], [0.1, 0.2])
    assert t.status == NOT_ESTIMABLE
    assert t.p_value == 1.0
    assert math.isnan(t.auroc)


def test_non_finite_scores_are_refused():
    with pytest.raises(ValueError):
        primary_test([0.1, float("nan")], [0.2, 0.3])


def test_a_not_estimable_antibiotic_enters_holm_with_p_one():
    result = holm({"ciprofloxacin": 0.01, "ceftriaxone": 1.0})
    assert result.adjusted == pytest.approx({"ciprofloxacin": 0.02, "ceftriaxone": 1.0})
    assert result.rejected == {"ciprofloxacin": True, "ceftriaxone": False}


def test_the_first_step_stays_at_half_alpha_when_the_other_antibiotic_is_dropped():
    result = holm({"ciprofloxacin": 0.03, "ceftriaxone": 1.0})
    assert result.rejected == {"ciprofloxacin": False, "ceftriaxone": False}   # 0.03 > 0.025


def test_holm_steps_down():
    result = holm({"ciprofloxacin": 0.02, "ceftriaxone": 0.04})
    assert result.rejected == {"ciprofloxacin": True, "ceftriaxone": True}
    assert result.adjusted == pytest.approx({"ciprofloxacin": 0.04, "ceftriaxone": 0.04})


def test_both_dropped_reject_nothing():
    assert not any(holm(dict.fromkeys(FAMILY, 1.0)).rejected.values())


@pytest.mark.parametrize("p_values", [
    {"ciprofloxacin": 0.01},                                             # an antibiotic missing from the family
    {"ciprofloxacin": 0.01, "ceftriaxone": 0.02, "cefotaxime": 0.03},    # a substitute added to it
])
def test_the_holm_family_is_fixed(p_values):
    with pytest.raises(ValueError):
        holm(p_values)


def test_a_nan_p_value_cannot_enter_holm():
    with pytest.raises(ValueError):
        holm({"ciprofloxacin": float("nan"), "ceftriaxone": 0.5})


def test_the_conclusion_sentences_are_exactly_those_of_amendment_b4():
    assert CONCLUSION_REJECTED == (
        "Evidence of above-chance ranking on MARISMa under the isolate-independence assumption.")
    assert CONCLUSION_NOT_REJECTED == "Above-chance ranking on MARISMa not demonstrated."
    assert CONCLUSION_NOT_ESTIMABLE == "Primary test not estimable; above-chance ranking on MARISMa not demonstrated."
    tested = PrimaryTest(TESTED, -3.0, 0.001, 0.7)
    assert conclusion(tested, rejected=True) == CONCLUSION_REJECTED
    assert conclusion(tested, rejected=False) == CONCLUSION_NOT_REJECTED
    assert conclusion(PrimaryTest(NOT_ESTIMABLE, math.nan, 1.0, 1.0), rejected=False) == CONCLUSION_NOT_ESTIMABLE
