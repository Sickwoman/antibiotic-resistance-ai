"""Version 2.0's registered evaluation: every available endpoint, as amended (docs/v2.0_marisma_plan.md: endpoints
1-8; amendments A3-A6, B2-B4, D6 and F5).

Pure functions: labels and predictions in, results out. Nothing here reads data, runs a model or writes a file, so
every rule can be tested on synthetic numbers. The scoring script feeds it the one set of saved predictions.

- **Primary (A3, B2, B4).** AUROC with ties counted one half, and a one-sided Brunner-Munzel test of AUROC > 0.5
  (`src/primary_endpoint.py`). A non-finite test is "not estimable" (Holm input p = 1). Fewer than 50 resistant or 50
  susceptible isolates make the reading descriptive (Holm input p = 1; endpoint 7, A4). Holm runs over the fixed pair
  {ciprofloxacin, ceftriaxone}; a dropped antibiotic enters with p = 1 (A6.1).
- **Intervals (A3.5, A3.6).** 2,000 stratified bootstrap resamples (seed 42), resistant and susceptible isolates
  resampled separately. A metric undefined in a resample is left out and counted; more than 20 undefined resamples
  (over 1 %) make its interval "not estimable"; a metric undefined on the full population is "not estimable". The
  intervals are descriptive.
- **Linkage robustness (A3.7).** The Brunner-Munzel statistic with its variance doubled, and its one-sided p.
- **The zone (A5).** Coverage, and NPV with its interval, read literally; an empty zone's NPV is "not estimable".
- **The gap (endpoint 2, A3.5).** Internal AUROC (patient-group bootstrap, as Version 0.7) minus external AUROC
  (stratified bootstrap), with the unpaired second-level bootstrap interval; "demonstrated" only if it excludes 0.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import bootstrap as group_bootstrap
from src.evaluate import calibration_slope_intercept, interval, unpaired_difference
from src.primary_endpoint import (
    CONCLUSION_NOT_ESTIMABLE,
    CONCLUSION_NOT_REJECTED,
    CONCLUSION_REJECTED,
    FAMILY,
    NOMINAL_ALPHA,
    TESTED,
    holm,
    primary_test,
)
from src.uncertainty import Zones, zone_metrics

RESAMPLES, SEED, LEVEL = 2000, 42, 0.95
UNDEFINED_LIMIT = 20                     # more than 1 % of 2,000 resamples undefined: "not estimable" (A3.6)
MIN_PER_CLASS = 50                       # endpoint 7 and A4: a confirmatory reading needs 50 of each class
SENSITIVITY_TARGET = 0.90                # a research target, not a clinical standard (endpoint 3)
NPV_TARGET = 0.95                        # a research criterion (A5)
DESIGN_EFFECT = 2.0                      # A3.7: an assumed design effect, not an estimate
NOT_ESTIMABLE = "not estimable"
UNAVAILABLE = "unavailable"
ERROR_CONTROL = ("Brunner–Munzel inference is approximate and assumes independent isolates. Holm adjustment does not "
                 "repair invalid component p-values. Missing patient linkage leaves actual error control uncertain.")
POSITIVE, NEGATIVE, INTERMEDIATE = ("R", "I"), ("S",), "I"
METRICS = ("roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "brier", "brier_base_rate",
           "calibration_intercept", "calibration_slope", "zone_coverage", "zone_npv")


# --- labels ---------------------------------------------------------------------------------------------------------

def labels(tokens: Mapping[str, str], *, exclude_intermediate: bool = False) -> tuple[dict[str, int], dict[str, Any]]:
    """The registered label rule (F5.1): stripped and upper-cased, R or I -> 1, S -> 0, anything else excluded and
    counted. With `exclude_intermediate`, I is excluded too (the I-excluded analysis)."""
    out: dict[str, int] = {}
    counts: Counter[str] = Counter()
    others: Counter[str] = Counter()
    for isolate, token in tokens.items():
        value = str(token).strip().upper()
        if value == INTERMEDIATE and exclude_intermediate:
            counts["I excluded"] += 1
        elif value in POSITIVE:
            out[isolate] = 1
            counts[value] += 1
        elif value in NEGATIVE:
            out[isolate] = 0
            counts[value] += 1
        else:
            counts["other (excluded)"] += 1
            others[value] += 1
    return out, {"by_value": dict(sorted(counts.items())), "excluded_other_values": dict(sorted(others.items())),
                 "labelled": len(out), "resistant": sum(out.values()), "susceptible": len(out) - sum(out.values())}


# --- point metrics and stratified intervals -------------------------------------------------------------------------

def point_metrics(y: np.ndarray, p: np.ndarray, threshold: float, zones: Zones) -> dict[str, float]:
    """Every metric of one population; NaN where it is undefined."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    pred = p >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum())
    rate = n_pos / y.size if y.size else math.nan
    slope, intercept = calibration_slope_intercept(y, p) if n_pos and n_neg else (math.nan, math.nan)
    zone = zone_metrics(y, p, zones)
    return {
        "roc_auc": float(roc_auc_score(y, p)) if n_pos and n_neg else math.nan,
        "pr_auc": float(average_precision_score(y, p)) if n_pos else math.nan,
        "sensitivity": tp / n_pos if n_pos else math.nan,
        "specificity": tn / n_neg if n_neg else math.nan,
        "precision": tp / (tp + fp) if tp + fp else math.nan,
        "brier": float(np.mean((p - y) ** 2)),
        "brier_base_rate": float(np.mean((rate - y) ** 2)),        # a constant prediction at the population's rate
        "calibration_intercept": intercept, "calibration_slope": slope,
        "zone_coverage": float(zone["share_susceptible"]),
        "zone_npv": float(zone["npv_susceptible"]),
    }


def stratified_intervals(y: np.ndarray, p: np.ndarray, threshold: float, zones: Zones, *,
                         resamples: int = RESAMPLES, seed: int = SEED) -> tuple[dict[str, Any], np.ndarray]:
    """Percentile intervals over stratified resamples (A3.5, A3.6), and the AUROC draws (the gap's external side)."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if pos.size == 0 or neg.size == 0:
        return {m: {"status": NOT_ESTIMABLE, "reason": "a class is empty"} for m in METRICS}, np.array([])
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in METRICS}
    for _ in range(resamples):
        idx = np.r_[rng.choice(pos, pos.size, replace=True), rng.choice(neg, neg.size, replace=True)]
        values = point_metrics(y[idx], p[idx], threshold, zones)
        for m in METRICS:
            draws[m].append(values[m])
    out: dict[str, Any] = {}
    for m in METRICS:
        v = np.asarray(draws[m], dtype=np.float64)
        undefined = int((~np.isfinite(v)).sum())
        if undefined > UNDEFINED_LIMIT:
            out[m] = {"status": NOT_ESTIMABLE, "undefined_resamples": undefined}
        else:
            low, high = interval(v[np.isfinite(v)], LEVEL)
            out[m] = {"low": low, "high": high, "undefined_resamples": undefined}
    return out, np.asarray(draws["roc_auc"], dtype=np.float64)


# --- tests ----------------------------------------------------------------------------------------------------------

def brunner_munzel_parts(resistant: Sequence[float], susceptible: Sequence[float]) -> tuple[float, float]:
    """scipy 1.18.1's Brunner-Munzel statistic and Satterthwaite degrees of freedom, resistant scores first."""
    x, y = np.asarray(resistant, dtype=np.float64), np.asarray(susceptible, dtype=np.float64)
    nx, ny = x.size, y.size
    rankc = rankdata(np.r_[x, y])
    rcx, rcy = rankc[:nx], rankc[nx:]
    rx, ry = rankdata(x), rankdata(y)
    with np.errstate(all="ignore"):
        sx = float(np.sum((rcx - rx - rcx.mean() + rx.mean()) ** 2) / (nx - 1))
        sy = float(np.sum((rcy - ry - rcy.mean() + ry.mean()) ** 2) / (ny - 1))
        statistic = nx * ny * (rcy.mean() - rcx.mean()) / ((nx + ny) * math.sqrt(nx * sx + ny * sy)) \
            if nx * sx + ny * sy > 0 else math.nan
        denominator = (nx * sx) ** 2 / (nx - 1) + (ny * sy) ** 2 / (ny - 1)
        df = (nx * sx + ny * sy) ** 2 / denominator if denominator > 0 else math.nan
    return float(statistic), float(df)


def linkage_robustness(resistant: Sequence[float], susceptible: Sequence[float]) -> dict[str, Any]:
    """A3.7: the statistic with its variance doubled (an assumed design effect of 2), and its one-sided p.
    Descriptive: it changes no decision."""
    statistic, df = brunner_munzel_parts(resistant, susceptible)
    if not (math.isfinite(statistic) and math.isfinite(df)):
        return {"status": NOT_ESTIMABLE, "design_effect": DESIGN_EFFECT}
    doubled = statistic / math.sqrt(DESIGN_EFFECT)
    p = float(stats.t.sf(-doubled, df))
    if not math.isfinite(p):
        return {"status": NOT_ESTIMABLE, "design_effect": DESIGN_EFFECT}
    return {"status": TESTED, "design_effect": DESIGN_EFFECT, "statistic": doubled, "df": df, "p_value": p}


def primary(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """Endpoint 1 for one antibiotic: the test, its confirmatory status, and the p that enters Holm."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    resistant, susceptible = p[y == 1], p[y == 0]
    test = primary_test(resistant, susceptible)
    confirmatory = resistant.size >= MIN_PER_CLASS and susceptible.size >= MIN_PER_CLASS
    holm_input = test.p_value if (test.status == TESTED and confirmatory) else 1.0
    return {"status": test.status, "statistic": test.statistic, "p_value": test.p_value, "auroc": test.auroc,
            "n_resistant": int(resistant.size), "n_susceptible": int(susceptible.size),
            "confirmatory": confirmatory,
            "reading": "confirmatory" if confirmatory else
            f"descriptive: fewer than {MIN_PER_CLASS} resistant or susceptible isolates",
            "holm_input": holm_input, "linkage_robustness": linkage_robustness(resistant, susceptible)}


def family(ciprofloxacin: dict[str, Any] | None, ceftriaxone: dict[str, Any] | None = None) -> dict[str, Any]:
    """Holm over the fixed pair; an unavailable antibiotic enters with p = 1 (A6.1). B4's sentence for each."""
    inputs = {"ciprofloxacin": 1.0 if ciprofloxacin is None else ciprofloxacin["holm_input"],
              "ceftriaxone": 1.0 if ceftriaxone is None else ceftriaxone["holm_input"]}
    result = holm(inputs)
    out: dict[str, Any] = {"family": list(FAMILY), "nominal_alpha": NOMINAL_ALPHA, "holm_inputs": inputs,
                           "adjusted": result.adjusted, "rejected": result.rejected, "error_control": ERROR_CONTROL,
                           "conclusions": {}}
    for name, res in (("ciprofloxacin", ciprofloxacin), ("ceftriaxone", ceftriaxone)):
        if res is None:
            out["conclusions"][name] = UNAVAILABLE
        elif res["status"] != TESTED:
            out["conclusions"][name] = CONCLUSION_NOT_ESTIMABLE
        else:
            out["conclusions"][name] = CONCLUSION_REJECTED if result.rejected[name] else CONCLUSION_NOT_REJECTED
    return out


# --- readings -------------------------------------------------------------------------------------------------------

def zone_reading(point: float, bounds: dict[str, Any]) -> dict[str, Any]:
    """A5: the point estimate and the interval relative to 0.95, literally; both historical readings."""
    if not math.isfinite(point):
        return {"point": NOT_ESTIMABLE, "interval": NOT_ESTIMABLE, "version_0_8_reading": NOT_ESTIMABLE,
                "version_0_7_reading": NOT_ESTIMABLE}
    meets = point >= NPV_TARGET
    if bounds.get("status") == NOT_ESTIMABLE:
        where = NOT_ESTIMABLE
    elif bounds["high"] < NPV_TARGET:
        where = "lies below 0.95"
    elif bounds["low"] > NPV_TARGET:
        where = "lies above 0.95"
    else:
        where = "contains 0.95"
    return {"point": "meets 0.95" if meets else "misses 0.95", "interval": where,
            "version_0_8_reading": "the point estimate meets 0.95" if meets else "the point estimate misses 0.95",
            "version_0_7_reading": ("the point estimate meets 0.95, or the interval contains 0.95"
                                    if meets or where == "contains 0.95" else "neither"),
            "note": "0.95 is a research criterion; no reading here establishes clinical reliability"}


def sensitivity_reading(point: float, bounds: dict[str, Any]) -> str:
    if not math.isfinite(point):
        return NOT_ESTIMABLE
    if bounds.get("status") == NOT_ESTIMABLE:
        return f"point {'meets' if point >= SENSITIVITY_TARGET else 'misses'} 0.90; interval not estimable"
    return (f"point {'meets' if point >= SENSITIVITY_TARGET else 'misses'} 0.90; interval "
            + ("lies below 0.90" if bounds["high"] < SENSITIVITY_TARGET else
               "lies above 0.90" if bounds["low"] > SENSITIVITY_TARGET else "contains 0.90"))


# --- populations ----------------------------------------------------------------------------------------------------

def evaluate_population(y: np.ndarray, p: np.ndarray, threshold: float, zones: Zones, *, test: bool,
                        resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, Any]:
    """Every registered metric of one population, with intervals; the primary test only if `test`."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    points = point_metrics(y, p, threshold, zones) if y.size else {m: math.nan for m in METRICS}
    bounds, auroc_draws = stratified_intervals(y, p, threshold, zones, resamples=resamples, seed=seed) \
        if y.size else ({m: {"status": NOT_ESTIMABLE} for m in METRICS}, np.array([]))
    metrics = {}
    for m in METRICS:
        estimate = points[m]
        metrics[m] = {"estimate": estimate if math.isfinite(estimate) else NOT_ESTIMABLE, **bounds[m]}
        if not math.isfinite(estimate):
            metrics[m]["status"] = NOT_ESTIMABLE
    out: dict[str, Any] = {
        "n": int(y.size), "n_resistant": int(y.sum()), "n_susceptible": int((y == 0).sum()),
        "prevalence": float(y.mean()) if y.size else NOT_ESTIMABLE,
        "threshold": threshold, "zone_lower_edge": zones.lower, "metrics": metrics,
        "pr_auc_no_skill": float(y.mean()) if y.size else NOT_ESTIMABLE,
        "sensitivity_against_target": sensitivity_reading(points["sensitivity"], bounds["sensitivity"]),
        "zone": {"n_in_zone": int((p < zones.lower).sum()) if zones.lower is not None else 0,
                 **zone_reading(points["zone_npv"], bounds["zone_npv"])},
        "_auroc_draws": auroc_draws,
    }
    if test:
        out["primary"] = primary(y, p)
    return out


def gap(internal_y: np.ndarray, internal_p: np.ndarray, internal_groups: np.ndarray, internal_threshold: float,
        external_auroc: float, external_draws: np.ndarray, *, resamples: int = RESAMPLES,
        seed: int = SEED) -> dict[str, Any]:
    """Endpoint 2: internal AUROC minus MARISMa AUROC. The internal side resamples whole patient groups (Version 0.7);
    the external side is the stratified draws; the interval is the unpaired second-level bootstrap."""
    internal_point = float(roc_auc_score(internal_y, internal_p))
    samples, skipped = group_bootstrap(internal_y, internal_groups, {"m": internal_p}, {"m": internal_threshold},
                                       resamples=resamples, seed=seed, metrics=("roc_auc",))
    internal_draws = samples["m"]["roc_auc"]
    if not math.isfinite(external_auroc) or external_draws.size == 0:
        return {"status": NOT_ESTIMABLE, "internal_auroc": internal_point}
    low, high = unpaired_difference(internal_draws, external_draws, LEVEL, seed=seed)
    demonstrated = bool(low > 0 or high < 0)
    return {"internal_auroc": internal_point, "internal_interval": interval(internal_draws, LEVEL),
            "internal_single_class_resamples_skipped": int(skipped),
            "external_auroc": external_auroc, "gap": internal_point - external_auroc, "low": low, "high": high,
            "demonstrated": demonstrated,
            "reading": "gap demonstrated (the interval excludes 0)" if demonstrated
            else "gap not demonstrated (the interval includes 0); this never means performance was maintained"}


def strip_private(result: Any) -> Any:
    """Drop the internal draws before a result is written out."""
    if isinstance(result, dict):
        return {k: strip_private(v) for k, v in result.items() if not k.startswith("_")}
    if isinstance(result, list):
        return [strip_private(v) for v in result]
    return result
