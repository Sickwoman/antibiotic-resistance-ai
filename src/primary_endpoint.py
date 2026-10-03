"""Version 2.0's primary-test rule, as proposed in docs/v2.0_marisma_plan.md (amendments A3 and B).

Pure functions on score arrays, with no input or output: nothing here reads data.

For each antibiotic the primary test is a one-sided Brunner-Munzel test (Brunner & Munzel 2000):
H0: theta <= 0.5 against H1: theta > 0.5. Here theta = P(S_R > S_S) + P(S_R = S_S) / 2, the AUROC with ties counted
one half, and the resistant isolates' scores are the first sample.

If the statistic or the p-value is not finite, the test is "not estimable", and the antibiotic enters the fixed
two-hypothesis Holm family (Holm 1979) with p = 1. No fallback test is run (amendment B2). Descriptive metrics are kept
wherever they are defined.

Brunner-Munzel inference is approximate and assumes independent isolates. Holm adjustment does not repair invalid
component p-values. Missing patient linkage leaves actual error control uncertain (amendment B3).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

FAMILY = ("ciprofloxacin", "ceftriaxone")   # fixed: a dropped antibiotic stays in it with p = 1
NOMINAL_ALPHA = 0.05                        # the nominal family-wise design level, not a guarantee (amendment B3)

TESTED = "tested"
NOT_ESTIMABLE = "not estimable"

CONCLUSION_REJECTED = "Evidence of above-chance ranking on MARISMa under the isolate-independence assumption."
CONCLUSION_NOT_REJECTED = "Above-chance ranking on MARISMa not demonstrated."
CONCLUSION_NOT_ESTIMABLE = "Primary test not estimable; above-chance ranking on MARISMa not demonstrated."


@dataclass(frozen=True)
class PrimaryTest:
    status: str          # TESTED or NOT_ESTIMABLE
    statistic: float     # scipy's statistic as returned: negative when resistant scores tend to be larger
    p_value: float       # what enters Holm: the one-sided p if TESTED, exactly 1.0 if NOT_ESTIMABLE
    auroc: float         # descriptive, ties counted one half; nan only if a class is empty


@dataclass(frozen=True)
class HolmResult:
    adjusted: dict[str, float]
    rejected: dict[str, bool]


def descriptive_auroc(resistant: Sequence[float], susceptible: Sequence[float]) -> float:
    """AUROC with ties counted one half; defined whenever both classes are present."""
    r, s = _scores(resistant), _scores(susceptible)
    if len(r) == 0 or len(s) == 0:
        return math.nan
    return float(roc_auc_score(np.r_[np.ones(len(r)), np.zeros(len(s))], np.r_[r, s]))


def primary_test(resistant: Sequence[float], susceptible: Sequence[float]) -> PrimaryTest:
    """The one-sided Brunner-Munzel test of AUROC > 0.5, or "not estimable" with p = 1 if it is not finite."""
    r, s = _scores(resistant), _scores(susceptible)
    auroc = descriptive_auroc(r, s)
    if len(r) == 0 or len(s) == 0:
        return PrimaryTest(NOT_ESTIMABLE, math.nan, 1.0, auroc)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)   # degenerate inputs warn; the status below records them
        result = stats.brunnermunzel(r, s, alternative="greater", distribution="t")
    statistic, p = float(result.statistic), float(result.pvalue)
    if not (math.isfinite(statistic) and math.isfinite(p)):
        return PrimaryTest(NOT_ESTIMABLE, statistic, 1.0, auroc)
    return PrimaryTest(TESTED, statistic, p, auroc)


def holm(p_values: Mapping[str, float], alpha: float = NOMINAL_ALPHA) -> HolmResult:
    """Holm's step-down procedure over the fixed family; adjusted p-values and rejections at the nominal level."""
    if set(p_values) != set(FAMILY):
        raise ValueError(f"The Holm family is fixed: exactly {FAMILY}. A dropped or not-estimable antibiotic "
                         f"enters with p = 1, and no other antibiotic may join it.")
    ps = {k: float(p_values[k]) for k in FAMILY}
    outside = [k for k, v in ps.items() if not 0.0 <= v <= 1.0]   # also catches nan
    if outside:
        raise ValueError(f"p-values must lie in [0, 1], got {outside}: a non-finite test enters as 1.0, never as nan.")
    order = sorted(FAMILY, key=lambda k: (ps[k], FAMILY.index(k)))
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * ps[k]))
        adjusted[k] = running
    return HolmResult(adjusted={k: adjusted[k] for k in FAMILY},
                      rejected={k: adjusted[k] <= alpha for k in FAMILY})


def conclusion(test: PrimaryTest, rejected: bool) -> str:
    """The one sentence an antibiotic's primary result may be reported with (amendment B4)."""
    if test.status == NOT_ESTIMABLE:
        return CONCLUSION_NOT_ESTIMABLE
    return CONCLUSION_REJECTED if rejected else CONCLUSION_NOT_REJECTED


def _scores(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float).ravel()
    if not np.isfinite(a).all():
        raise ValueError("Scores must be finite: a frozen model's probabilities are never missing or infinite.")
    return a
