"""Version 0.6: three answers instead of two, so the model can decline to call a borderline isolate.

A calibrated probability just above the decision cut-off is not evidence; forcing it into "resistant"
hides that. This module fits two edges around the model's cut-off on the **validation** part, by the rule
pre-registered in docs/v0.6_explainability_plan.md:

    p < lower   -> high-confidence susceptible   (at least `target_npv` of those calls are truly S)
    lower <= p <= upper -> uncertain, conventional AST confirmation recommended
    p > upper   -> high-confidence resistant      (at least `target_precision` of those calls are truly R)

The edges are the extreme cuts that still meet their target with at least `min_coverage` of rows, so each
zone is as large as it can be while keeping its promise. A target is never lowered after seeing the answer:
if a side cannot reach it, that zone does not exist and the report says what the side does reach instead
(`SideResult.best_*`). The model's own cut-off is not moved by any of this; only the name of the output
changes.

Nothing here decides treatment. "High-confidence resistant" is a research label on a research prototype.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.evaluate import group_resample_indices

SUSCEPTIBLE = "High-confidence susceptible"
RESISTANT = "High-confidence resistant"
UNCERTAIN = "Uncertain"
ADVICE = "Uncertain - conventional antimicrobial susceptibility testing recommended."


class UncertaintyError(ValueError):
    """The zone rule cannot be applied to the numbers given."""


@dataclass(frozen=True)
class ZoneRule:
    """The pre-registered targets. `min_coverage` stops a zone that only ever covers a handful of rows."""

    target_npv: float = 0.95
    target_precision: float = 0.95
    min_coverage: float = 0.05

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ZoneRule:
        rule = cls(float(cfg.get("target_npv", 0.95)), float(cfg.get("target_precision", 0.95)),
                   float(cfg.get("min_coverage", 0.05)))
        for name, value in (("target_npv", rule.target_npv), ("target_precision", rule.target_precision)):
            if not 0.5 < value <= 1:
                raise UncertaintyError(f"{name} must be above 0.5 and at most 1, got {value}.")
        if not 0 < rule.min_coverage <= 1:
            raise UncertaintyError(f"min_coverage must be in (0, 1], got {rule.min_coverage}.")
        return rule


@dataclass(frozen=True)
class SideResult:
    """One side of the band: the edge that met the target, or why none did."""

    side: str                       # "susceptible" or "resistant"
    target: float
    edge: float | None              # None: no cut meets the target at the required coverage
    n_covered: int
    coverage: float
    achieved: float                 # NPV (susceptible side) or precision (resistant side) at `edge`
    best_achieved: float            # the most that side reaches at >= min_coverage, whatever the target
    best_edge: float | None
    best_coverage: float

    @property
    def exists(self) -> bool:
        return self.edge is not None


def _side_values(y: np.ndarray, prob: np.ndarray, cut: float, side: str) -> tuple[int, float]:
    """(rows covered, share of them that the zone claims) for one candidate cut."""
    covered = prob < cut if side == "susceptible" else prob > cut
    n = int(covered.sum())
    if n == 0:
        return 0, float("nan")
    correct = (y[covered] == 0).mean() if side == "susceptible" else (y[covered] == 1).mean()
    return n, float(correct)


def side_curve(y: np.ndarray, prob: np.ndarray, threshold: float, side: str) -> list[dict[str, Any]]:
    """Every candidate cut for one side, with its coverage and what it achieves.

    Candidates are the observed probabilities on that side of the cut-off plus the cut-off itself, so the
    rule has no free parameter beyond the targets.
    """
    if side not in ("susceptible", "resistant"):
        raise UncertaintyError(f"side must be 'susceptible' or 'resistant', got {side!r}.")
    values = np.unique(prob[prob <= threshold]) if side == "susceptible" else np.unique(prob[prob >= threshold])
    cuts = np.unique(np.append(values, threshold))
    rows = []
    for cut in cuts.tolist():
        n, achieved = _side_values(y, prob, cut, side)
        rows.append({"side": side, "cut": float(cut), "n_covered": n, "coverage": n / len(prob),
                     "achieved": achieved})
    return rows


def fit_side(y: np.ndarray, prob: np.ndarray, threshold: float, side: str, target: float,
             min_coverage: float) -> SideResult:
    """The extreme cut on one side that still meets `target` at `min_coverage` (largest for the
    susceptible side, smallest for the resistant side: both mean as many confident calls as possible)."""
    curve = [row for row in side_curve(y, prob, threshold, side) if row["n_covered"] > 0]
    wide = [row for row in curve if row["coverage"] >= min_coverage]
    passing = [row for row in wide if row["achieved"] >= target]
    pick = max if side == "susceptible" else min
    best = max(wide, key=lambda r: (r["achieved"], r["coverage"]), default=None)
    chosen = pick(passing, key=lambda r: r["cut"], default=None) if passing else None
    return SideResult(side=side, target=target,
                      edge=None if chosen is None else chosen["cut"],
                      n_covered=0 if chosen is None else chosen["n_covered"],
                      coverage=0.0 if chosen is None else chosen["coverage"],
                      achieved=float("nan") if chosen is None else chosen["achieved"],
                      best_achieved=float("nan") if best is None else best["achieved"],
                      best_edge=None if best is None else best["cut"],
                      best_coverage=0.0 if best is None else best["coverage"])


@dataclass(frozen=True)
class Zones:
    """The fitted band. `lower`/`upper` are None where that confident zone does not exist."""

    threshold: float
    lower: float | None
    upper: float | None
    rule: ZoneRule
    fitted_on: str = "validation"
    n_fitted: int = 0

    def label(self, prob: np.ndarray | float) -> np.ndarray:
        """The three-way output for one probability or an array of them."""
        p = np.atleast_1d(np.asarray(prob, dtype=np.float64))
        if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
            raise UncertaintyError("Probabilities must be finite and between 0 and 1.")
        out = np.full(p.shape, UNCERTAIN, dtype=object)
        if self.lower is not None:
            out[p < self.lower] = SUSCEPTIBLE
        if self.upper is not None:
            out[p > self.upper] = RESISTANT
        return out

    def one(self, prob: float) -> str:
        return str(self.label(prob)[0])

    def to_dict(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "lower": self.lower, "upper": self.upper,
                "rule": asdict(self.rule), "fitted_on": self.fitted_on, "n_fitted": self.n_fitted}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Zones:
        missing = {"threshold", "lower", "upper"} - set(data)
        if missing:
            raise UncertaintyError(f"Not a zones record: {sorted(missing)} missing.")
        lower, upper = data["lower"], data["upper"]
        zones = cls(float(data["threshold"]), None if lower is None else float(lower),
                    None if upper is None else float(upper), ZoneRule.from_config(data.get("rule") or {}),
                    str(data.get("fitted_on", "validation")), int(data.get("n_fitted", 0)))
        if zones.lower is not None and zones.upper is not None and zones.lower > zones.upper:
            raise UncertaintyError(f"lower edge {zones.lower} is above upper edge {zones.upper}.")
        return zones


def fit_zones(y: np.ndarray, prob: np.ndarray, threshold: float,
              rule: ZoneRule) -> tuple[Zones, SideResult, SideResult, list[dict[str, Any]]]:
    """Fit both edges on the part given (validation, by the protocol) and return the trade-off curves too."""
    y = np.asarray(y).astype(np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    if y.shape != prob.shape or y.size == 0:
        raise UncertaintyError(f"Need one probability per label, got {prob.shape} for {y.shape}.")
    if np.unique(y).size < 2:
        raise UncertaintyError("Both classes are needed to fit confidence zones.")
    if not 0 <= threshold <= 1:
        raise UncertaintyError(f"The decision threshold must be a probability, got {threshold}.")
    low = fit_side(y, prob, threshold, "susceptible", rule.target_npv, rule.min_coverage)
    high = fit_side(y, prob, threshold, "resistant", rule.target_precision, rule.min_coverage)
    zones = Zones(float(threshold), low.edge, high.edge, rule, "validation", int(y.size))
    curve = side_curve(y, prob, threshold, "susceptible") + side_curve(y, prob, threshold, "resistant")
    return zones, low, high, curve


def zone_metrics(y: np.ndarray, prob: np.ndarray, zones: Zones) -> dict[str, Any]:
    """How the fitted zones behave on a set of labelled probabilities (validation, or stored test ones)."""
    y = np.asarray(y).astype(np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    if y.shape != prob.shape or y.size == 0:
        raise UncertaintyError(f"Need one probability per label, got {prob.shape} for {y.shape}.")
    labels = zones.label(prob)
    # n_resistant is the number of resistant *isolates*; the zone counts are n_zone_* so the two cannot
    # be confused (or silently overwrite each other) in a report row.
    out: dict[str, Any] = {"n": int(y.size), "n_resistant": int((y == 1).sum())}
    for name, key in ((SUSCEPTIBLE, "susceptible"), (UNCERTAIN, "uncertain"), (RESISTANT, "resistant")):
        mask = labels == name
        n = int(mask.sum())
        out[f"n_zone_{key}"] = n
        out[f"share_{key}"] = n / y.size
        out[f"resistant_share_{key}"] = float(y[mask].mean()) if n else float("nan")
    out["npv_susceptible"] = (1 - out["resistant_share_susceptible"]) if out["n_zone_susceptible"] else float("nan")
    out["precision_resistant"] = out["resistant_share_resistant"]
    out["confident_share"] = (out["n_zone_susceptible"] + out["n_zone_resistant"]) / y.size
    confident = labels != UNCERTAIN
    if confident.any():
        called = np.where(labels[confident] == RESISTANT, 1, 0)
        out["accuracy_confident"] = float((called == y[confident]).mean())
    else:
        out["accuracy_confident"] = float("nan")
    return out


def bootstrap_zone_metrics(y: np.ndarray, groups: np.ndarray, prob: np.ndarray, zones: Zones, *,
                           resamples: int = 2000, level: float = 0.95, seed: int = 42,
                           keys: tuple[str, ...] = ("share_susceptible", "npv_susceptible",
                                                    "share_uncertain", "share_resistant",
                                                    "precision_resistant", "confident_share")
                           ) -> dict[str, dict[str, float]]:
    """Percentile intervals for the zone numbers, resampling whole patient groups (protocol section 6)."""
    y = np.asarray(y).astype(np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    groups = np.asarray(groups)
    if groups.size != y.size:
        raise UncertaintyError(f"Need one group per row, got {groups.size} for {y.size}.")
    order = np.argsort(groups, kind="stable")
    _, starts, counts = np.unique(groups[order], return_index=True, return_counts=True)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(resamples):
        idx = group_resample_indices(order, starts, counts, rng.integers(0, starts.size, starts.size))
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        values = zone_metrics(yb, prob[idx], zones)
        for key in keys:
            value = values.get(key, float("nan"))
            if np.isfinite(value):
                draws[key].append(float(value))
    alpha = (1 - level) / 2
    point = zone_metrics(y, prob, zones)
    return {key: {"value": float(point.get(key, float("nan"))),
                  "low": float(np.quantile(v, alpha)) if v else float("nan"),
                  "high": float(np.quantile(v, 1 - alpha)) if v else float("nan"),
                  "draws": len(v)} for key, v in draws.items()}
