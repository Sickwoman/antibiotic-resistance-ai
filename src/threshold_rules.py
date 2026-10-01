"""Version 1.3: cut-off rules for a sensitivity target, and the time-ordered windows they are chosen in.

Fixed by docs/v1.3_threshold_plan.md (protocol amendment 10), recorded before this file existed.

Two rules are compared, both functions of the resistant scores of a selection sample only:

- **E**, the protocol's empirical rule: the highest cut-off whose selection sensitivity is at least the target
  (`src.evaluate.choose_threshold`) - the ceil(target * n)-th highest resistant score.
- **U**, the order-statistic tolerance rule: the k*-th lowest resistant score, with
  k* = max{k : P(Binomial(n, 1 - target) >= k) >= confidence}. For n independent resistant scores drawn from the
  distribution future resistant scores come from, by a scoring function that did not see them, F(r(k)) is
  Beta(k, n - k + 1), so P(true sensitivity >= target) = P(Binomial(n, 1 - target) >= k) >= confidence. This is
  the nonparametric tolerance-limit argument of the Neyman-Pearson umbrella algorithm (Tong, Feng & Li, Sci Adv
  2018; 4: eaao1659). It accounts for the cut-off being chosen from the data; a confidence bound computed after a
  free search over cut-offs would not. When no k qualifies (n too small), U is **infeasible** and gives no cut-off.

Everything else here enforces time and patients: windows that end a label-availability gap before an origin,
one selection spectrum per patient group, and no later-period spectrum from a patient already seen that year.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.evaluate import choose_threshold


class ThresholdRuleError(RuntimeError):
    """A Version 1.3 design, time-ordering or support check failed."""


# --- the rules ---------------------------------------------------------------------------------------------------

def order_statistic_k(n: int, target: float, confidence: float) -> int:
    """k* for U; 0 when no cut-off can carry the stated confidence (U is infeasible)."""
    if not (0 < target < 1 and 0 < confidence < 1):
        raise ThresholdRuleError("target and confidence must lie strictly between 0 and 1.")
    best = 0
    for k in range(1, int(n) + 1):
        if binom.sf(k - 1, n, 1 - target) >= confidence:     # P(Binomial(n, 1 - target) >= k)
            best = k
        else:
            break                                            # the tail only shrinks as k grows
    return best


def minimum_feasible_n(target: float, confidence: float) -> int:
    """The smallest number of resistant cases for which U gives any cut-off (29 at 0.90 / 0.95)."""
    n = 1
    while order_statistic_k(n, target, confidence) == 0:
        n += 1
    return n


def tolerance_cutoff(resistant_scores: np.ndarray, target: float, confidence: float) -> dict[str, Any]:
    """Rule U on one selection sample's resistant scores."""
    r = np.sort(np.asarray(resistant_scores, dtype=np.float64))
    k = order_statistic_k(r.size, target, confidence)
    if k == 0:
        return {"rule": "U", "feasible": False, "threshold": None, "n": int(r.size), "k": 0,
                "misses_allowed": None, "stated_probability": None}
    return {"rule": "U", "feasible": True, "threshold": float(r[k - 1]), "n": int(r.size), "k": k,
            "misses_allowed": k - 1, "stated_probability": float(binom.sf(k - 1, r.size, 1 - target))}


def empirical_cutoff(resistant_scores: np.ndarray, target: float) -> dict[str, Any]:
    """Rule E, through the protocol's own implementation, on the same selection sample's resistant scores."""
    r = np.asarray(resistant_scores, dtype=np.float64)
    if r.size == 0:
        raise ThresholdRuleError("No resistant case in the selection sample: rule E cannot be applied.")
    threshold = choose_threshold(np.ones(r.size, dtype=np.int64), r, {"rule": "min_sensitivity",
                                                                         "min_sensitivity": target})
    flagged = int((r >= threshold).sum())
    return {"rule": "E", "feasible": True, "threshold": float(threshold), "n": int(r.size),
            "k": int(r.size - flagged + 1), "misses_allowed": int(r.size - flagged), "stated_probability": None}


# --- time and patients ---------------------------------------------------------------------------------------------

def origin_windows(dates: np.ndarray, rows: np.ndarray, origin: str, end: str, gap_days: int) -> dict[str, np.ndarray]:
    """Rows (a subset of `rows`) usable before an origin, in its label-availability gap, and in its later period.

    A label counts as available `gap_days` after its spectrum's acquisition date, so only rows dated before
    origin - gap may be used for fitting, calibration, a cut-off or a recalibration.
    """
    rows = np.asarray(rows, dtype=np.int64)
    d = dates[rows]
    start, stop = np.datetime64(pd.Timestamp(origin)), np.datetime64(pd.Timestamp(end))
    available = start - np.timedelta64(int(gap_days), "D")
    if np.isnat(d).any():
        raise ThresholdRuleError("A pool spectrum has no acquisition date; the time order cannot be established.")
    return {"available": rows[d < available], "gap": rows[(d >= available) & (d < start)],
            "later": rows[(d >= start) & (d < stop)], "available_before": np.array([available])}


def drop_seen_patients(later: np.ndarray, seen: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, int]:
    """Later-period rows of patient groups not seen before the origin (groups are within-year identifiers)."""
    keep = ~np.isin(groups[later], np.unique(groups[seen]))
    return later[keep], int((~keep).sum())


def one_per_group(rows: np.ndarray, dates: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """One row per patient group: the earliest-dated, then the lowest row number."""
    rows = np.asarray(rows, dtype=np.int64)
    order = np.lexsort((rows, dates[rows]))                    # by date, then row
    ordered = rows[order]
    _, first = np.unique(groups[ordered], return_index=True)
    return np.sort(ordered[first])


def assert_available(rows: np.ndarray, dates: np.ndarray, origin: str, gap_days: int, what: str) -> None:
    """Every row used for `what` was labelled at least `gap_days` before the origin."""
    limit = np.datetime64(pd.Timestamp(origin)) - np.timedelta64(int(gap_days), "D")
    late = int((dates[np.asarray(rows, dtype=np.int64)] >= limit).sum())
    if late:
        raise ThresholdRuleError(f"{what} would use {late} row(s) whose label is not yet available at {origin}.")


# --- what a cut-off delivers ----------------------------------------------------------------------------------------

def delivered(y: np.ndarray, p: np.ndarray, threshold: float | None) -> dict[str, float]:
    """Sensitivity, specificity, flag rate and precision of 'resistant if p >= threshold'."""
    y = np.asarray(y).astype(np.int64)
    if threshold is None:
        return dict.fromkeys(("sensitivity", "specificity", "flag_rate", "precision"), float("nan"))
    flagged = np.asarray(p) >= threshold
    positives, negatives = int(y.sum()), int((y == 0).sum())
    return {"sensitivity": float((flagged & (y == 1)).sum() / positives) if positives else float("nan"),
            "specificity": float((~flagged & (y == 0)).sum() / negatives) if negatives else float("nan"),
            "flag_rate": float(flagged.mean()),
            "precision": float((flagged & (y == 1)).sum() / flagged.sum()) if flagged.any() else float("nan")}


def intercept_shift(p: np.ndarray, y: np.ndarray, iterations: int = 50) -> float:
    """The single intercept d maximising the likelihood of y under logit(p) + d (Newton's method)."""
    offset = np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1))
    y = np.asarray(y, dtype=np.float64)
    d = 0.0
    for _ in range(iterations):
        q = 1 / (1 + np.exp(-(offset + d)))
        step = (y - q).sum() / max((q * (1 - q)).sum(), 1e-12)
        d += step
        if abs(step) < 1e-12:
            break
    return float(d)


def apply_shift(p: np.ndarray, d: float) -> np.ndarray:
    offset = np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1))
    return 1 / (1 + np.exp(-(offset + d)))
