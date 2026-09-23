"""Markdown tables of the Version 0.7 generalisation experiments, generated from the saved reports.

Run from the project root with the virtual environment active, after scripts/measure_generalisation.py:

    python scripts/generalisation_tables.py

Writes <generalisation.report_dir>/<dataset>/tables.md and prints it. Nothing here computes a result:
every number comes from the files the run wrote, as the evaluation protocol requires.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tables import md_table, number  # noqa: E402
from src.utils import ConfigError, load_config, project_path  # noqa: E402

# Printed under the temporal row wherever it appears. DRIAMS-A re-hashes patient IDs every year, so a
# returning patient cannot be detected across the boundary (protocol amendment 1, point 1). Generated
# rather than typed, so it cannot be dropped from a report.
TEMPORAL_CAVEAT = ("The temporal result is a date-separated evaluation with incomplete patient linkage, "
                   "never a patient-level generalisation result: DRIAMS-A re-hashes patient identifiers "
                   "every year, so a patient who returns in a later year cannot be detected.")
EXTERNAL_CAVEAT = ("External-site intervals are sample-level with unknown within-patient dependence: "
                   "DRIAMS-B, -C and -D carry no patient identifiers, so repeated isolates of one patient "
                   "count as independent and these intervals are expected to be too narrow.")
NOT_SOLVED = ("These numbers measure how much is lost across sites and time. They do not show that the "
              "problem is solved, and no adaptation is attempted in this version.")


def read(folder: Path, name: str) -> pd.DataFrame | None:
    path = folder / name
    frame = pd.read_csv(path) if path.is_file() else None
    return frame if frame is not None and len(frame) else None


def read_json(folder: Path, name: str) -> Any | None:
    path = folder / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def verdict(low: float, high: float) -> str:
    """The pre-registered reading: a gap counts only when its interval excludes zero."""
    if low > 0 or high < 0:
        return "shown"
    return "not demonstrated"


def headline_section(test: pd.DataFrame, seed: int) -> str:
    """One row per experiment and site, for the seed the intervals are reported for."""
    models = test[test["model"] != "prevalence"]
    rows = []
    for r in models[models["seed"] == seed].itertuples(index=False):
        rows.append({"Experiment": r.split, "Trained on": r.train_sites, "Tested on": r.site,
                     "n": int(r.n), "Resistant": int(r.n_resistant),
                     "AUROC": number(float(r.roc_auc), 3), "PR-AUC": number(float(r.pr_auc), 3),
                     "Brier": number(float(r.brier), 3),
                     "Sensitivity": number(float(r.sensitivity), 2),
                     "Specificity": number(float(r.specificity), 2),
                     "Model": r.model})
    return md_table(rows)


def reference_section(test: pd.DataFrame) -> str:
    """What a model that always predicts the training resistance rate scores on the same rows."""
    rows = []
    for r in test[test["model"] == "prevalence"].itertuples(index=False):
        rows.append({"Experiment": r.split, "Tested on": r.site, "n": int(r.n),
                     "Resistant share": number(float(r.n_resistant) / float(r.n), 3),
                     "AUROC": number(float(r.roc_auc), 3), "PR-AUC": number(float(r.pr_auc), 3),
                     "Brier": number(float(r.brier), 3)})
    return md_table(rows)


def seed_section(test: pd.DataFrame) -> str:
    """Mean and range over the seeds, so a single seed is never mistaken for the result."""
    models = test[test["model"] != "prevalence"]
    rows = []
    for (split, site), group in models.groupby(["split", "site"], sort=False):
        if group["seed"].nunique() < 2:
            continue
        rows.append({"Experiment": split, "Tested on": site, "Seeds": int(group["seed"].nunique()),
                     "Mean AUROC": number(float(group["roc_auc"].mean()), 3),
                     "Lowest": number(float(group["roc_auc"].min()), 3),
                     "Highest": number(float(group["roc_auc"].max()), 3)})
    return md_table(rows)


def gap_section(gaps: pd.DataFrame) -> str:
    rows = []
    for r in gaps.itertuples(index=False):
        rows.append({"Comparison": r.comparison, "n": int(r.n), "Resistant": int(r.n_resistant),
                     "AUROC there": number(float(r.other_roc_auc), 3),
                     "Gap": number(float(r.gap), 3),
                     "95 % interval": f"[{number(float(r.low), 3)}, {number(float(r.high), 3)}]",
                     "Verdict": verdict(float(r.low), float(r.high))})
    return md_table(rows)


def paired_section(paired: Any) -> str:
    entries = paired if isinstance(paired, list) else [paired]
    rows = []
    for p in entries:
        rows.append({"Comparison": p["comparison"], "Tested on": p.get("part", "–"), "n": int(p["n"]),
                     "Fewer sites": f"{p['fewer_sites']['experiment']} "
                                    f"({'+'.join(p['fewer_sites']['train_sites'])})",
                     "AUROC": number(float(p["fewer_sites_roc_auc"]), 3),
                     "More sites": f"{p['more_sites']['experiment']} "
                                   f"({'+'.join(p['more_sites']['train_sites'])})",
                     "AUROC ": number(float(p["more_sites_roc_auc"]), 3),
                     "Difference": number(float(p["difference"]), 3),
                     "95 % interval": f"[{number(float(p['low']), 3)}, {number(float(p['high']), 3)}]",
                     "Verdict": verdict(float(p["low"]), float(p["high"]))})
    return md_table(rows)


def zone_section(transfer: pd.DataFrame) -> str:
    rows = []
    for r in transfer.itertuples(index=False):
        rows.append({"Tested on": r.part, "Model": r.model,
                     "Clean site test": "yes" if bool(r.pure_site_test) else "no (also recalibrated)",
                     "n": int(r.n), "Covered": number(float(r.share_susceptible), 3),
                     "Correct there (NPV)": number(float(r.npv_susceptible), 3),
                     "95 % interval": f"[{number(float(r.npv_low), 3)}, {number(float(r.npv_high), 3)}]",
                     f"Reaches {number(float(r.target_npv), 2)}": "yes" if bool(r.transfers) else "no"})
    return md_table(rows)


def shift_section(shift: pd.DataFrame) -> str:
    rows = []
    for r in shift.itertuples(index=False):
        rows.append({"Tested on": r.part, "n": int(r.n),
                     "Median absolute SMD over all bins": number(float(r.median_abs_smd), 3),
                     "Share of bins above 0.5": number(float(r.share_bins_abs_smd_over_half), 3),
                     f"Median absolute SMD, lowest {int(r.lowest_bins)} bins":
                         number(float(r.median_abs_smd_lowest_bins), 3),
                     "Empty there, training site": number(float(r.empty_share_lowest_bins_reference), 2),
                     "Empty there, this site": number(float(r.empty_share_lowest_bins_part), 2)})
    return md_table(rows)


def build_tables(report_dir: Path) -> str:
    run = read_json(report_dir, "run_config.json")
    if run is None:
        raise ConfigError(f"{report_dir} has no run_config.json; run scripts/measure_generalisation.py first.")
    test = read(report_dir, "test_metrics.csv")
    if test is None:
        raise ConfigError(f"{report_dir} has no test_metrics.csv.")
    main_seed = int(run["seeds"][0])
    faithful = run.get("refit_faithful") or {}

    parts: list[str] = [
        f"<!-- generated by scripts/generalisation_tables.py from {report_dir.as_posix()} -->",
        f"**What is compared:** the setting of {run['reused_setting_from']}, refitted unchanged on each "
        f"experiment's own training part. Intervals are for seed {main_seed}; "
        f"{len(run['seeds'])} seeds were fitted.",
        f"**The refit reproduces the saved model** on the source split's validation part to "
        f"{number(float(faithful.get('difference', float('nan'))), 12)} AUROC, checked before any test "
        "part was scored.",
        "**How well does it work where it was not trained?**",
        headline_section(test, main_seed),
        TEMPORAL_CAVEAT,
        EXTERNAL_CAVEAT,
    ]
    references = reference_section(test)
    if references:
        parts += ["**What no-skill scores on the same rows** (always predicts the training resistance rate)",
                  references]
    seeds = seed_section(test)
    if seeds:
        parts += ["**Across the fitted seeds**", seeds]
    gaps = read(report_dir, "generalisation_gaps.csv")
    if gaps is not None:
        parts += ["**How much is lost against the random split** (different rows, so the interval is the "
                  "wider unpaired one)", gap_section(gaps)]
    paired = read_json(report_dir, "two_sites_versus_one.json")
    if paired:
        parts += ["**Does a second training site help at a third?** (the same rows both times, so this "
                  "comparison is paired)", paired_section(paired)]
    transfer = read(report_dir, "zone_transfer.csv")
    if transfer is not None:
        parts += ["**Does the Version 0.6 confidence zone still hold?** The edge is applied unchanged, "
                  "never refitted. Only the rows marked as a clean site test isolate the change of site; "
                  "the others also carry that experiment's own calibration step.",
                  zone_section(transfer)]
    shift = read(report_dir, "shift_diagnostic.csv")
    if shift is not None:
        parts += ["**How different are the spectra themselves?** No AST label is used here; this describes "
                  "the shift, it does not correct for it.", shift_section(shift)]
    parts.append(NOT_SOLVED)
    return "\n\n".join(p for p in parts if p) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Markdown tables for the Version 0.7 experiments.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        dataset = args.dataset or config["dataset"]["name"]
        report_dir = project_path(config["generalisation"]["report_dir"]) / dataset
        text = build_tables(report_dir)
        (report_dir / "tables.md").write_text(text, encoding="utf-8")
        print(text)
        return 0
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
