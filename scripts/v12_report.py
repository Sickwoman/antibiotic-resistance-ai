"""Version 1.2 - the complete generated report, from the run's saved outputs only.

    python scripts/v12_report.py

The development run (scripts/v12_development.py, run `ba4144c`) generated `tables.md`, which left out three of
the plan's pre-specified secondary endpoints. They were computed and saved; this script tabulates **every**
pre-specified endpoint from those saved files and writes `tables_complete.md` beside `tables.md`, which is kept
exactly as the run wrote it.

**It fits nothing, predicts nothing and loads no data.** It reads only: `primary_result.json`,
`pooled_results.json`, `stability.csv`, `forward_check.csv` and the run's rows of the development log. Every
difference is printed with its operands named ("X minus Y") and what a positive value means, because a sign is
only meaningful with its comparator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tables import md_table  # noqa: E402
from src.utils import ConfigError, get_logger, load_config, project_path, show_path  # noqa: E402

log = get_logger("v12_report")
FITTED = ("B", "C", "D1")
LABELS = {"R0": "R0 no-skill", "B": "B unchanged Version 1.1 procedure", "C": "C candidate",
          "D1": "D1 diagnostic (B's setting, C's procedure)"}
SIGNS = [
    ("Primary: Brier(B) minus Brier(C)", "C's Brier is lower (better)"),
    ("Brier(B) minus Brier(D1)", "D1's Brier is lower (better)"),
    ("Brier(D1) minus Brier(B), as the plan names it", "D1's Brier is higher (worse)"),
    ("Brier(C) minus Brier(D1)", "C's Brier is higher (worse)"),
    ("AUROC or PR-AUC: C minus B, D1 minus B, C minus D1", "the first-named arm ranks better"),
    ("Forward: Brier(B) minus Brier(C)", "C's Brier is lower (better)"),
]
FORWARD_NOTE = (
    "**What the forward check does and does not show.** In this one date-separated split, no arm delivered its "
    "0.90 target. Two further observations sit beside that: the resistance rate was higher in the evaluated year "
    "(171 of 1,506, 11.4 %, against 76 of 915, 8.3 %), and every fitted arm under-predicted it on average "
    "(positive calibration intercepts). **Neither establishes why sensitivity fell.** A change in prevalence "
    "alone — pure label shift, with resistant and susceptible spectra scoring as before — would leave "
    "sensitivity at a fixed cut-off unchanged, and a positive intercept is compatible with several kinds of shift. "
    "The loss may reflect how the later year's resistant spectra scored, the sampling noise of cut-offs chosen on "
    "at most 76 resistant spectra, the cut-off rule's own tendency to fall short, or a mixture; this design cannot "
    "separate them. Patients cannot be linked across years, so the split is date-separated, not "
    "patient-separated.")


def ci(d: dict[str, float], digits: int = 4) -> str:
    return f"{d['estimate']:.{digits}f} ({d['low']:.{digits}f}–{d['high']:.{digits}f})"


def signed(d: dict[str, float], digits: int = 4) -> str:
    return f"{d['estimate']:+.{digits}f} [{d['low']:+.{digits}f}, {d['high']:+.{digits}f}]"


def flip(d: dict[str, float]) -> dict[str, float]:
    """X minus Y from Y minus X: the estimate changes sign and the bounds swap."""
    return {"estimate": -d["estimate"], "low": -d["high"], "high": -d["low"]}


def reading(d: dict[str, float], positive: str, negative: str) -> str:
    if d["low"] > 0:
        return f"{positive}; interval excludes 0"
    if d["high"] < 0:
        return f"{negative}; interval excludes 0"
    return "not demonstrated (interval contains 0)"


def load(report: Path, dev_log: Path) -> dict[str, Any]:
    needed = ["primary_result.json", "pooled_results.json", "stability.csv", "forward_check.csv"]
    missing = [n for n in needed if not (report / n).is_file()]
    if missing or not dev_log.is_file():
        raise ConfigError(f"Saved Version 1.2 outputs are missing ({missing or dev_log}); run the study first.")
    primary = json.loads((report / "primary_result.json").read_text(encoding="utf-8"))
    run_id = str(primary["provenance"]["run_id"])
    logged = pd.read_csv(dev_log)
    logged = logged[(logged["run_id"].astype(str) == run_id) & (logged["status"] == "ok")]
    if logged.empty:
        raise ConfigError(f"The development log holds no rows of run {run_id}.")
    return {"primary": primary, "run_id": run_id, "logged": logged,
            "pooled": json.loads((report / "pooled_results.json").read_text(encoding="utf-8")),
            "stability": pd.read_csv(report / "stability.csv").set_index("arm"),
            "forward": pd.read_csv(report / "forward_check.csv")}


def stability_row(stab: pd.DataFrame, a: str) -> dict[str, str]:
    r = stab.loc[a]
    return {"Arm": a, "Cut-offs supported": f"{int(r['supported_cutoffs'])} of {int(r['folds'])}",
            "Resistant in selection set (mean)": f"{r['selection_resistant_mean']:.0f}",
            "Delivered sensitivity mean (SD)": f"{r['sensitivity_mean']:.3f} ({r['sensitivity_sd']:.3f})",
            "Range": f"{r['sensitivity_min']:.2f}–{r['sensitivity_max']:.2f}",
            "Folds ≥ 0.90 / ≥ 0.85": f"{int(r['folds_at_or_above_0.90'])} / {int(r['folds_at_or_above_0.85'])}",
            "Specificity mean (SD)": f"{r['specificity_mean']:.3f} ({r['specificity_sd']:.3f})",
            "Cut-off mean (SD)": f"{r['threshold_mean']:.4f} ({r['threshold_sd']:.4f})"}


def forward_row(r: pd.Series, fw: dict[str, Any], fw_log: pd.DataFrame) -> dict[str, str]:
    arm, scores, decided = r["arm"], fw["scores"]["intervals"], fw["decisions"]["intervals"]
    fitted = arm != "R0"
    calibration = (f"{fw_log.loc[arm, 'calibration_slope']:.2f} / {fw_log.loc[arm, 'calibration_intercept']:+.2f}"
                   if fitted else "—")
    return {"Arm": arm, "Train n / resistant": f"{int(r['train_n'])} / {int(r['train_resistant'])}",
            "Evaluated n / resistant": f"{int(r['n'])} / {int(r['n_resistant'])}",
            "Brier": ci(scores[arm]["brier"]), "AUROC": ci(scores[arm]["roc_auc"], 3) if fitted else "—",
            "Calibration slope / intercept": calibration, "Cut-off supported": "yes" if r["supported"] else "no",
            "Delivered sensitivity": ci(decided[arm]["sensitivity"], 3),
            "Delivered specificity": ci(decided[arm]["specificity"], 3)}


def diagnostic_row(name: str, d: dict[str, float], positive: str, negative: str) -> dict[str, str]:
    return {"Difference (as the plan names it)": name, "Estimate [95 % interval]": signed(d),
            "Reading": reading(d, positive, negative)}


def build(s: dict[str, Any], dataset: str, primary_seed: int) -> str:
    primary, pooled, logged, run_id = s["primary"], s["pooled"], s["logged"], s["run_id"]
    seed = str(primary_seed)
    main, fw, stab = pooled[seed], pooled["forward"], s["stability"]
    iv, dv = main["scores"]["intervals"], main["decisions"]["intervals"]
    diff = main["scores"]["differences_to_reference"]                      # reference B: "B minus <arm>"
    cv = logged[(logged["experiment"] == f"v1.2/{run_id}/cv") & (logged["seed"] == primary_seed)].set_index("model")
    fw_log = logged[logged["experiment"] == f"v1.2/{run_id}/forward"].set_index("model")
    b_minus_c, b_minus_d1 = diff["C"]["brier"], diff["D1"]["brier"]
    c_minus_b_auc, c_minus_b_pr = flip(diff["C"]["roc_auc"]), flip(diff["C"]["pr_auc"])
    fw_b_minus_c = fw["scores"]["differences_to_reference"]["C"]["brier"]
    repeats = {k: v["estimate"] for k, v in primary["repeats"].items()}
    others = [v for k, v in repeats.items() if k != seed]
    skill = main["brier_skill_vs_R0"]

    out = [f"# Version 1.2 development tables, complete (`{dataset}`) — exploratory", "",
           f"Generated by `scripts/v12_report.py` from the saved outputs of run `{run_id}` only: **no model was "
           "refitted, no prediction regenerated and no data loaded.** It tabulates every endpoint the plan "
           "(`docs/v1.2_calibration_plan.md`, section 8) pre-specified. `tables.md`, which the run generated and "
           "which omits three of them, is kept unchanged beside this file. Development only; every number here "
           "is exploratory. Research only; no clinical claim.", "",
           "## How every difference is signed", "",
           md_table([{"Difference": d, "Positive means": m} for d, m in SIGNS]), "",
           "Brier is a property of the probabilities alone: **no cut-off enters it.** So a Brier difference between "
           "B and D1 — the same setting, fitted on 7/8 against 8/8 of each training fold — can only come from "
           "fitting and calibrating on that extra eighth of the data, never from how the cut-off was chosen.", "",
           f"## Primary endpoint (partition seed {seed})", "",
           f"Brier(B) minus Brier(C): **{signed(b_minus_c)}** — {reading(b_minus_c, 'C better', 'C worse')}. "
           f"Brier B {iv['B']['brier']['estimate']:.5f}, C {iv['C']['brier']['estimate']:.5f}. Verdict under the "
           f"plan: **{primary['verdict']}**; never \"equivalent\".", "",
           "## 8. Partitions 43 and 44: the primary difference repeated", "",
           md_table([{"Partition": k, "Brier(B) minus Brier(C)": signed(v)} for k, v in primary["repeats"].items()]),
           "",
           f"Across the three partitions: mean {sum(repeats.values()) / len(repeats):+.5f}, range "
           f"{min(repeats.values()):+.5f} to {max(repeats.values()):+.5f}; across 43 and 44 only: mean "
           f"{sum(others) / len(others):+.5f}, range {min(others):+.5f} to {max(others):+.5f}. The partitions "
           "re-use the same 2,421 spectra; they are not independent cohorts.", "",
           f"## 1. Brier skill against R0, and 5. delivered sensitivity and specificity (partition {seed})", "",
           md_table([{"Arm": LABELS[a], "Brier": ci(iv[a]["brier"]),
                      "Brier skill vs R0": ci(skill[a], 3) if a in FITTED else "—",
                      "Delivered sensitivity": ci(dv[a]["sensitivity"], 3),
                      "Delivered specificity": ci(dv[a]["specificity"], 3)} for a in ("R0", *FITTED)]), "",
           "R0 flags every isolate (its cut-off is its own constant), so its delivered sensitivity is 1 and its "
           "specificity 0 by construction.", "",
           f"## 2. Discrimination (partition {seed}, pooled) and C minus B", "",
           md_table([{"Arm": LABELS[a], "AUROC": ci(iv[a]["roc_auc"]), "PR-AUC": ci(iv[a]["pr_auc"])}
                     for a in FITTED]), "",
           f"C minus B: AUROC **{signed(c_minus_b_auc)}** — {reading(c_minus_b_auc, 'C better', 'C worse')}; "
           f"PR-AUC **{signed(c_minus_b_pr)}** — {reading(c_minus_b_pr, 'C better', 'C worse')}.", "",
           f"## 3. Calibration and log loss (partition {seed}, pooled; point estimates, as pre-specified)", "",
           md_table([{"Arm": LABELS[a], "Calibration slope": f"{cv.loc[a, 'calibration_slope']:.3f}",
                      "Calibration intercept": f"{cv.loc[a, 'calibration_intercept']:+.3f}",
                      "Log loss": f"{cv.loc[a, 'log_loss']:.4f}"} for a in FITTED]), "",
           f"R0's pooled log loss is {cv.loc['R0', 'log_loss']:.4f}. Its pooled calibration slope "
           f"({cv.loc['R0', 'calibration_slope']:.2f}) and AUROC ({cv.loc['R0', 'roc_auc']:.3f}) in the development "
           "log carry **no meaning**: R0 predicts each fold's own training rate, so pooled over folds it is five "
           "constants rather than one.", "",
           "## 4. Operating-point stability over the 15 held-out folds", "",
           md_table([stability_row(stab, a) for a in FITTED]), "",
           f"The plan's descriptive rule \"C more stable than B\" (smaller SD **and** mean closer to 0.90): "
           f"**{'met' if primary['c_more_stable_descriptive'] else 'not met'}**. It is a description, not a test, "
           "and the 15 folds re-use one pool of spectra.", "",
           "## 6. Forward in time (fitted before 2017, evaluated once on 2017)", "",
           md_table([forward_row(r, fw, fw_log) for _, r in s["forward"].iterrows()]), "",
           f"Brier(B) minus Brier(C), forward: {signed(fw_b_minus_c)} — "
           f"{reading(fw_b_minus_c, 'C better', 'C worse')}.", "",
           FORWARD_NOTE, "",
           f"## 7. The diagnostics (partition {seed}, paired)", "",
           md_table([diagnostic_row("D1 minus B, Brier", flip(diff["D1"]["brier"]), "D1 worse", "D1 better"),
                     diagnostic_row("D1 minus B, AUROC", flip(diff["D1"]["roc_auc"]), "D1 better", "D1 worse"),
                     diagnostic_row("C minus D1, Brier", main["C_minus_D1"]["brier"], "C worse", "C better"),
                     diagnostic_row("C minus D1, AUROC", main["C_minus_D1"]["roc_auc"], "C better", "C worse")]),
           "",
           f"The +0.0020 quoted earlier is **Brier(B) minus Brier(D1)** = {signed(b_minus_d1)}: D1's Brier was "
           "lower. B and D1 share a setting and differ only in fitting on 7/8 against 8/8 of each training fold "
           "(Brier involves no cut-off), so this is the effect of about one-eighth more fitting data — not of the "
           "cross-fitted cut-off. Its lower bound is barely above zero, it is one of several unadjusted exploratory "
           "comparisons, and B's and D1's setting was selected in Version 1.1 on these same labels.", "",
           "All intervals: 2,000 paired patient-group bootstrap resamples of the pooled held-out predictions, which "
           "treat each fold's fitted model as fixed and so leave out fitting variability.", ""]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Version 1.2: the complete report from saved outputs only.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    try:
        vc = load_config(args.config)["v12_development"]
        report = project_path(vc["report_dir"]) / vc["dataset"]
        saved = load(report, project_path(vc["development_log"]))
        text = build(saved, vc["dataset"], int(vc["primary_partition_seed"]))
        (report / "tables_complete.md").write_text(text, encoding="utf-8", newline="\n")
        log.info("wrote %s from saved outputs only", show_path(report / "tables_complete.md"))
        return 0
    except ConfigError as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
