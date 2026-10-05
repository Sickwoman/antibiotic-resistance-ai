"""The demo's external-evaluation panel: figures read from committed aggregate results at runtime, never typed in.

Sources (committed; nothing is recomputed):
- `results/metrics/v2.0/marisma_evaluation.json`: the Version 2.0 one-time evaluation on MARISMa (results commit
  `89e73e8`);
- `results/metrics/v0.4/ecoli_ciprofloxacin/test_intervals.json`: the internal DRIAMS-A test of the same model.

One count is derived: the R or I isolates in the confidence zone, which is n_in_zone minus NPV x n_in_zone. It must
come out a whole number. No individual prediction or identifier is read: these files hold aggregates only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path("results/metrics/v2.0/marisma_evaluation.json")
INTERNAL = Path("results/metrics/v0.4/ecoli_ciprofloxacin/test_intervals.json")


class EvidenceError(RuntimeError):
    """The committed aggregates are missing or do not reconcile."""


def _metric(entry: dict[str, Any]) -> dict[str, float]:
    return {"estimate": float(entry["estimate"]), "low": float(entry["low"]), "high": float(entry["high"])}


def _whole(value: float, what: str) -> int:
    if abs(value - round(value)) > 1e-6:
        raise EvidenceError(f"{what} is not a whole number: {value!r}")
    return int(round(value))


def load_evidence(root: Path = ROOT) -> dict[str, Any]:
    raw = (root / REPORT).read_bytes().replace(b"\r\n", b"\n")    # the committed (LF) content, on any checkout
    report = json.loads(raw)
    primary, family, gap = report["primary"], report["holm_family"], report["gap"]
    metrics, zone = primary["metrics"], primary["zone"]
    n_zone = int(zone["n_in_zone"])
    in_zone_s = _whole(metrics["zone_npv"]["estimate"] * n_zone, "the S isolates in the zone")
    internal = json.loads((root / INTERNAL).read_bytes())["random"]["intervals"]["tuned_lightgbm"]
    return {
        "source": {"report": REPORT.as_posix(), "report_sha256": hashlib.sha256(raw).hexdigest(),
                   "written_utc": report["written_utc"], "authorised_commit": report["authorised_commit"][:7],
                   "internal": INTERNAL.as_posix()},
        "population": {"scope": report["scope"], "n": int(primary["n"]), "n_ri": int(primary["n_resistant"]),
                       "n_s": int(primary["n_susceptible"]), "prevalence": float(primary["prevalence"]),
                       "candidates": int(report["populations"]["candidate_scoring_cohort"]),
                       "matched": int(report["populations"]["matched_to_an_amr_record"])},
        "threshold": float(primary["threshold"]),
        "auroc": _metric(metrics["roc_auc"]),
        "holm_adjusted_p": float(family["adjusted"]["ciprofloxacin"]),
        "conclusion": family["conclusions"]["ciprofloxacin"],
        "error_control": family["error_control"],
        "ceftriaxone": family["conclusions"]["ceftriaxone"],
        "sensitivity": _metric(metrics["sensitivity"]),
        "specificity": _metric(metrics["specificity"]),
        "calibration_intercept": _metric(metrics["calibration_intercept"]),
        "zone": {"lower_edge": float(primary["zone_lower_edge"]), "n": n_zone, "ri": n_zone - in_zone_s,
                 "npv": _metric(metrics["zone_npv"]), "point_reading": zone["point"],
                 "interval_reading": zone["interval"]},
        "gap": {"estimate": float(gap["gap"]), "low": float(gap["low"]), "high": float(gap["high"]),
                "demonstrated": bool(gap["demonstrated"]), "internal_auroc": float(gap["internal_auroc"])},
        "internal": {"auroc": _metric(internal["roc_auc"]), "sensitivity": _metric(internal["sensitivity"]),
                     "specificity": _metric(internal["specificity"])},
    }
