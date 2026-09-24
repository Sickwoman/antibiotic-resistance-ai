"""Version 0.8: adapting the saved model to a new site, and judging the result by pre-registered rules.

Every rule here is fixed in docs/v0.8_adaptive_plan.md and protocol amendment 4, approved before DRIAMS-C
was opened. Nothing in this module may be changed after an evaluation result has been seen.

Two adaptation arms, both leaving the hyperparameters alone:

* `recalibrate_only` -- the frozen model keeps every tree; only the two Platt parameters and the zone edge
  are refitted on the new site's adaptation part. This is the primary arm.
* the A + C refit, which reuses `src.tuning.fit_calibrated` with the Version 0.4 winning setting and is
  therefore not implemented again here.

The decision logic is deliberately arithmetic. There is no "promising" or "trending" verdict: a result is
success, null, harm, or mixed, and `classify` returns exactly one of them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SUCCESS = "success"
NULL = "not demonstrated"
HARM = "harm"
MIXED = "mixed"

# The primary arm is the recalibration; the confirmatory secondary arm is the A + C refit. Holm is applied
# across exactly these two, and across nothing else (the plan, section 8).
CONFIRMATORY_ARMS = ("recalibrated", "refit_a_plus_c")


class AdaptationError(ValueError):
    pass


# --- the primary endpoint -------------------------------------------------------------------------------------

def brier(y: np.ndarray, prob: np.ndarray) -> float:
    """Brier score, BS = (1/n) * sum_i (p_i - y_i)^2, lower is better.

    A strictly proper scoring rule: it cannot be improved by reporting anything other than the believed
    probability, which is why the plan makes it the primary endpoint. AUROC could not be: recalibration is
    monotone and therefore leaves AUROC exactly unchanged.
    """
    y = np.asarray(y).astype(np.float64)
    prob = np.asarray(prob, dtype=np.float64)
    if y.shape != prob.shape or y.size == 0:
        raise AdaptationError(f"Need one probability per label, got {prob.shape} for {y.shape}.")
    if not np.all(np.isfinite(prob)) or prob.min() < 0 or prob.max() > 1:
        raise AdaptationError("Probabilities must be finite and within [0, 1].")
    return float(np.mean((prob - y) ** 2))


# --- the primary adaptation arm -------------------------------------------------------------------------------

@dataclass(frozen=True)
class PlattRecalibration:
    """The two parameters of a refitted sigmoid, and the data they were fitted on.

    `a` and `b` map the frozen model's uncalibrated probability to a new probability. Nothing else moves:
    the trees, the preprocessing, the feature space and the threshold rule are all untouched.
    """

    a: float
    b: float
    n_fitted: int
    n_resistant: int
    fitted_on: str = "adaptation"

    def apply(self, uncalibrated: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(uncalibrated, dtype=np.float64), 1e-12, 1 - 1e-12)
        return 1.0 / (1.0 + np.exp(self.a * p + self.b))

    def to_dict(self) -> dict[str, Any]:
        return {"method": "platt_sigmoid", "a": self.a, "b": self.b, "n_fitted": self.n_fitted,
                "n_resistant": self.n_resistant, "fitted_on": self.fitted_on,
                "changed": ["calibrator_a", "calibrator_b"], "frozen": ["model", "preprocessing",
                                                                        "features", "threshold_rule"]}


def uncalibrated_probabilities(bundle_pipeline: Any, X: np.ndarray) -> np.ndarray:
    """The frozen model's own probability, before any calibration step.

    `CalibratedClassifierCV(method="sigmoid")` fits its sigmoid on exactly this quantity when the estimator
    has no `decision_function`, which is the case for the project's LightGBM pipeline. Recalibration
    therefore has to start from the same place, or it would not be the same procedure.
    """
    from src.tuning import uncalibrated

    inner = uncalibrated(bundle_pipeline)
    if not hasattr(inner, "predict_proba"):
        raise AdaptationError("The frozen model has no predict_proba, so it cannot be recalibrated.")
    return np.asarray(inner.predict_proba(X)[:, 1], dtype=np.float64)


def recalibrate_only(bundle_pipeline: Any, X_adapt: np.ndarray, y_adapt: np.ndarray) -> PlattRecalibration:
    """Refit the sigmoid on the new site's adaptation part. The model itself is never touched.

    Uses scikit-learn's own `_SigmoidCalibration`, the same primitive `CalibratedClassifierCV` uses, so the
    recalibrated model differs from the saved one in exactly two numbers.
    """
    from sklearn.calibration import _SigmoidCalibration

    y = np.asarray(y_adapt).astype(np.int64)
    if y.size == 0:
        raise AdaptationError("The adaptation part is empty.")
    if np.unique(y).size < 2:
        raise AdaptationError("Recalibration needs both classes in the adaptation part.")
    raw = uncalibrated_probabilities(bundle_pipeline, X_adapt)
    if raw.size != y.size:
        raise AdaptationError(f"Got {raw.size} probabilities for {y.size} adaptation labels.")
    fitted = _SigmoidCalibration().fit(raw.reshape(-1, 1).ravel(), y)
    return PlattRecalibration(float(fitted.a_), float(fitted.b_), int(y.size), int((y == 1).sum()))


# --- the co-primary endpoint ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class ZoneCriterion:
    """The pre-registered confidence-zone rule: a safe zone that covers almost nobody is not an improvement.

    `baseline_coverage` is the coverage of the *carried-over* zone, on the same evaluation rows, by the
    baseline model named in the plan. It is passed in rather than inferred so that which baseline it came
    from is explicit in the record.
    """

    target_npv: float = 0.95
    coverage_slack: float = 0.05

    def passes(self, npv: float, coverage: float, baseline_coverage: float) -> bool:
        if not np.isfinite(npv) or not np.isfinite(coverage) or not np.isfinite(baseline_coverage):
            return False
        return bool(npv >= self.target_npv and coverage >= baseline_coverage - self.coverage_slack)

    def explain(self, npv: float, coverage: float, baseline_coverage: float) -> dict[str, Any]:
        floor = baseline_coverage - self.coverage_slack
        return {"npv": npv, "npv_target": self.target_npv, "npv_ok": bool(np.isfinite(npv)
                                                                          and npv >= self.target_npv),
                "coverage": coverage, "baseline_coverage": baseline_coverage, "coverage_floor": floor,
                "coverage_ok": bool(np.isfinite(coverage) and coverage >= floor),
                "passes": self.passes(npv, coverage, baseline_coverage)}


# --- statistics -----------------------------------------------------------------------------------------------

def bootstrap_p_value(draws: np.ndarray) -> float:
    """Two-sided achieved significance level of a bootstrap difference against 0.

    The project reports intervals rather than p-values, but Holm needs an ordering, so this is derived from
    the same draws the interval comes from: twice the smaller tail, floored at 1/(draws+1) because a
    bootstrap cannot resolve a p-value below its own resolution.
    """
    d = np.asarray(draws, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    tail = min(float(np.mean(d <= 0)), float(np.mean(d >= 0)))
    return float(min(1.0, max(2.0 * tail, 1.0 / (d.size + 1))))


def holm(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Holm-Bonferroni across the confirmatory arms, in the plan's fixed order.

    Holm is a step-down procedure: the smallest p-value is tested at alpha/m, the next at alpha/(m-1), and
    testing stops at the first failure, so every later arm is not rejected regardless of its own p-value.
    """
    usable = {k: v for k, v in p_values.items() if np.isfinite(v)}
    m = len(usable)
    out: dict[str, dict[str, Any]] = {k: {"p_value": v, "threshold": float("nan"), "rejected": False,
                                          "adjusted_level": float("nan")}
                                      for k, v in p_values.items()}
    still_rejecting = True
    for rank, (name, p) in enumerate(sorted(usable.items(), key=lambda kv: kv[1])):
        threshold = alpha / (m - rank)
        rejected = bool(still_rejecting and p <= threshold)
        if not rejected:
            still_rejecting = False
        out[name] = {"p_value": float(p), "threshold": float(threshold), "rejected": rejected,
                     "adjusted_level": float(1.0 - threshold), "rank": rank + 1, "n_compared": m}
    return out


# --- the pre-registered verdict -------------------------------------------------------------------------------

@dataclass
class ArmResult:
    """One adaptation arm, judged against the pre-registered rules and nothing else."""

    arm: str
    brier_baseline: float
    brier_adapted: float
    delta: float                     # baseline - adapted; positive favours adaptation
    low: float
    high: float
    zone: dict[str, Any] = field(default_factory=dict)
    baseline_zone_ok: bool = False
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)


def classify(delta_low: float, delta_high: float, zone_passes: bool, baseline_zone_passes: bool,
             npv_adapted: float, target_npv: float = 0.95) -> tuple[str, list[str]]:
    """The pre-registered arithmetic verdict. Exactly one of success / null / harm / mixed.

    Success needs *both* the Brier interval to exclude 0 favourably and the zone pair to hold. Harm is
    either an unfavourable Brier interval excluding 0, or an adapted zone that falls below the target where
    the baseline's held it. A favourable Brier interval with a failing zone is `mixed`, which the plan calls
    a real and interpretable finding -- better average probabilities bought by refusing to answer.
    """
    if not np.isfinite(delta_low) or not np.isfinite(delta_high):
        raise AdaptationError("The Brier interval is not finite; the protocol has no rule for that case.")
    if delta_low > delta_high:
        raise AdaptationError(f"Interval bounds are reversed ({delta_low} > {delta_high}).")

    better = delta_low > 0                       # interval wholly above 0: adaptation improves Brier
    worse = delta_high < 0                       # interval wholly below 0: adaptation worsens Brier
    zone_harm = bool(baseline_zone_passes and np.isfinite(npv_adapted) and npv_adapted < target_npv)

    reasons = []
    if better:
        reasons.append(f"Brier interval [{delta_low:.4f}, {delta_high:.4f}] excludes 0 in favour of adaptation")
    elif worse:
        reasons.append(f"Brier interval [{delta_low:.4f}, {delta_high:.4f}] excludes 0 against adaptation")
    else:
        reasons.append(f"Brier interval [{delta_low:.4f}, {delta_high:.4f}] includes 0")
    reasons.append(f"zone pair {'holds' if zone_passes else 'does not hold'}")
    if zone_harm:
        reasons.append(f"the adapted zone falls below the {target_npv:.2f} target where the baseline's held")

    if worse or zone_harm:
        return HARM, reasons
    if better and zone_passes:
        return SUCCESS, reasons
    if better and not zone_passes:
        return MIXED, reasons
    return NULL, reasons


# --- the deterministic partition ------------------------------------------------------------------------------

def partition_fingerprint(meta: pd.DataFrame, rows: np.ndarray) -> str:
    """Short hash of *which samples* are in a part, independent of row order and of the row numbering.

    Keyed on (site, year_folder, code) rather than on positions, so it still identifies the same part if the
    cohort is rebuilt. Recording it is what makes the pre-registered partition checkable after the fact.
    """
    from src.dataset import sample_keys

    keys = sample_keys(meta)[np.asarray(rows, dtype=np.int64)]
    payload = "\n".join(sorted(str(k) for k in keys))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def split_adaptation_and_holdout(meta: pd.DataFrame, rows: np.ndarray, adaptation_fraction: float,
                                 seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The pre-registered 70 / 30 partition, by whole patient groups, seed 42.

    Reuses `grouped_subsample`, which adds whole groups in a seeded permutation and refuses a result below
    99 % of the requested size -- so a partition that could not be drawn as specified fails loudly instead
    of quietly becoming a different partition.
    """
    from src.train import grouped_subsample

    rows = np.asarray(rows, dtype=np.int64)
    if not 0 < adaptation_fraction < 1:
        raise AdaptationError(f"adaptation_fraction must be strictly between 0 and 1, got {adaptation_fraction}.")
    if rows.size < 2:
        raise AdaptationError("Too few rows to partition.")
    wanted = int(round(adaptation_fraction * rows.size))
    adaptation = grouped_subsample(meta, rows, wanted, seed)
    holdout = np.setdiff1d(rows, adaptation, assume_unique=False)
    assert_disjoint(meta, adaptation, holdout)
    return np.sort(adaptation), np.sort(holdout)


def assert_disjoint(meta: pd.DataFrame, adaptation: np.ndarray, holdout: np.ndarray) -> None:
    """No row and no patient group may be in both parts. Raises rather than warning."""
    shared_rows = np.intersect1d(adaptation, holdout)
    if shared_rows.size:
        raise AdaptationError(f"{shared_rows.size} row(s) are in both the adaptation and the held-out part.")
    groups = meta["group_id"].to_numpy()
    shared = np.intersect1d(np.unique(groups[adaptation]), np.unique(groups[holdout]))
    if shared.size:
        raise AdaptationError(f"{shared.size} patient group(s) span the adaptation and held-out parts, so a "
                              "patient would be both adapted on and evaluated on.")


def eligibility(counts: dict[str, int], min_per_class_site: int, min_per_class_holdout: int,
                holdout_counts: dict[str, int]) -> dict[str, Any]:
    """The two pre-registered gates, evaluated on class counts only -- never on any model output.

    E1 is the existing site rule (`pair_selection.min_per_class_external`). E2 is the Version 0.8 power
    precondition on the part that carries the result. Both are fixed before the cohort is seen, and a
    failure stops the experiment instead of relaxing the rule or substituting another site.
    """
    e1 = bool(counts.get("resistant", 0) >= min_per_class_site
              and counts.get("susceptible", 0) >= min_per_class_site)
    e2 = bool(holdout_counts.get("resistant", 0) >= min_per_class_holdout
              and holdout_counts.get("susceptible", 0) >= min_per_class_holdout)
    return {"e1_site_eligible": e1, "e1_min_per_class": min_per_class_site, "cohort_counts": dict(counts),
            "e2_holdout_powered": e2, "e2_min_per_class": min_per_class_holdout,
            "holdout_counts": dict(holdout_counts), "eligible": bool(e1 and e2),
            "stop_reason": None if (e1 and e2) else
            ("E1: the cohort does not meet the site eligibility rule" if not e1
             else "E2: the held-out part would carry too few of a class to evaluate")}
