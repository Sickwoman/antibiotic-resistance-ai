"""Markdown tables of the tuned-model results, generated from the saved reports (never typed by hand).

Run from the project root with the virtual environment active, after the matching test run:

    python scripts/tuned_tables.py                  # Version 0.4 (classical families)
    python scripts/tuned_tables.py --section deep   # Version 0.5 (neural networks)

Writes <report_dir>/<dataset>/tables.md and prints it. The comparison table asked for in the project plan
(accuracy, precision, recall, F1, ROC-AUC, PR-AUC, training time, inference time) holds every version
side by side.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tables import md_table, model_name, number, seconds, signed_ci, with_ci  # noqa: E402
from src.utils import ConfigError, load_config, project_path  # noqa: E402

# Which config section holds the current version, and the earlier versions it is compared with (oldest first).
SECTIONS = {
    "tuning": {"version": "0.4", "previous": [("0.3", "baselines")],
               "note": "Training time: Version 0.3 is one fit; Version 0.4 covers the cross-fitted models used "
                       "for calibration plus the final fit, and excludes the search. Inference is the batch time "
                       "per spectrum on the test part (Version 0.3 with all CPU threads, Version 0.4 with one). "
                       "End-to-end prediction from a raw file: {medians}, median."},
    "deep": {"version": "0.5", "previous": [("0.3", "baselines"), ("0.4", "tuning")],
             "note": "Training time: Version 0.3 is one fit; Versions 0.4 and 0.5 cover the cross-fitted models "
                     "used for calibration plus the final fit, and exclude the search. Inference is the batch "
                     "time per spectrum on the test part (Version 0.3 with all CPU threads, later versions with "
                     "one). End-to-end prediction from a raw file: {medians}, median."},
}


def seed_rows(table: pd.DataFrame, experiment: str, seed: int) -> pd.DataFrame:
    rows = table[table["experiment"] == experiment]
    return rows[(rows["seed"] == seed) | rows["seed"].isna()]


def search_section(search: pd.DataFrame, main: str) -> str:
    rows = [{"Family": model_name(r.family), "Settings tried": f"{int(r.settings):,}",
             "Best cross-validated AUROC": f"{r.cv_roc_auc_mean:.3f} ± {r.cv_roc_auc_sd:.3f}",
             "PR-AUC": number(r.cv_pr_auc_mean), "Search time": seconds(r.search_seconds),
             "Chosen setting": f"`{r.best_setting}`"}
            for r in search[search["experiment"] == main].itertuples()]
    return md_table(rows)


def validation_section(validation: pd.DataFrame, main: str, seed: int) -> str:
    rows = []
    for r in seed_rows(validation, main, seed).itertuples():
        rows.append({"Model": model_name(r.model), "Calibrated": "yes" if r.calibrated else "no",
                     "AUROC": number(r.roc_auc), "PR-AUC": number(r.pr_auc), "Brier": number(r.brier),
                     "Calibration slope": number(getattr(r, "calibration_slope", None), 2),
                     "Cut-off": f"{r.threshold:.3g}", "Sensitivity": number(r.sensitivity),
                     "Specificity": number(r.specificity)})
    return md_table(rows)


def test_section(intervals: dict[str, Any], experiment: str, chosen: str, older_version: str = "0.3") -> str:
    entry = intervals[experiment]
    # Version 0.4 wrote `differences_to_version_0_3`; from Version 0.5 the key names the previous section.
    to_older = entry.get("differences_to_reference_model") or entry.get("differences_to_version_0_3", {})
    rows = []
    for model, m in entry["intervals"].items():
        row = {"Model": model_name(model) + (" (saved)" if model == chosen else ""),
               "AUROC": with_ci(m["roc_auc"]), "PR-AUC": with_ci(m["pr_auc"]),
               "Sensitivity": with_ci(m["sensitivity"]), "Specificity": with_ci(m["specificity"])}
        best = entry.get("differences_to_reference", {}).get(model, {}).get("roc_auc")
        older = to_older.get(model, {}).get("roc_auc")
        row["AUROC: saved model minus this"] = "–" if best is None else signed_ci(best)
        row[f"AUROC: this minus Version {older_version}"] = "–" if older is None else signed_ci(older)
        rows.append(row)
    return md_table(rows)


@dataclass
class Report:
    """One version's saved report: what the tables are built from, never typed by hand."""
    version: str
    directory: Path
    test: pd.DataFrame
    timing: dict[str, Any]
    search: pd.DataFrame | None        # Version 0.3 had fixed settings, so it has no search summary

    @classmethod
    def load(cls, version: str, directory: Path) -> Report:
        search = directory / "search_summary.csv"
        return cls(version, directory, pd.read_csv(directory / "test_metrics.csv"),
                   json.loads((directory / "inference_timing.json").read_text(encoding="utf-8")),
                   pd.read_csv(search) if search.is_file() else None)


def comparison_section(reports: list[Report], main: str, seed: int, note: str) -> str:
    """The table the project plan asks for, with every version's models side by side."""
    rows = []
    for report in reports:
        search_time = {} if report.search is None else {
            r.family: r.search_seconds for r in report.search[report.search["experiment"] == main].itertuples()}
        for r in seed_rows(report.test, main, seed).itertuples():
            if str(r.model).startswith("v0."):
                continue                      # a model carried in from another version: listed from its own report
            family = str(r.model).removeprefix("tuned_")
            rows.append({
                "Version": report.version, "Model": model_name(r.model), "Accuracy": number(r.accuracy),
                "Precision": number(r.precision), "Recall (= sensitivity)": number(r.recall),
                "F1": number(r.f1), "ROC-AUC": number(r.roc_auc), "PR-AUC": number(r.pr_auc),
                "Search time": seconds(search_time[family]) if family in search_time else "–",
                "Training time": seconds(getattr(r, "fit_seconds", None)),
                "Inference (ms/spectrum, batch)": number(getattr(r, "predict_ms_per_sample", None), 3)})
    medians = [f"{report.timing['total_ms']['median']} ms (Version {report.version})" for report in reports]
    joined = " and ".join(medians) if len(medians) < 3 else f"{', '.join(medians[:-1])} and {medians[-1]}"
    return md_table(rows) + "\n" + f"\n{note.format(medians=joined)}\n"


def build_tables(report_dir: Path, previous: list[tuple[str, Path]], version: str = "0.4",
                 note: str = "", label: str | None = None) -> str:
    """`previous` names the earlier versions' report folders, oldest first."""
    card = json.loads((report_dir / "best_model_card.json").read_text(encoding="utf-8"))
    main, seed, chosen = card["split"]["experiment"], card["seed"], card["model"]
    search = pd.read_csv(report_dir / "search_summary.csv")
    validation = pd.read_csv(report_dir / "validation_metrics.csv")
    test = pd.read_csv(report_dir / "test_metrics.csv")
    intervals = json.loads((report_dir / "test_intervals.json").read_text(encoding="utf-8"))
    timing = json.loads((report_dir / "inference_timing.json").read_text(encoding="utf-8"))
    older = [Report.load(name, folder) for name, folder in previous]
    reports = [*older, Report(version, report_dir, test, timing, search)]
    first = test[test["experiment"] == main].iloc[0]

    out = [f"<!-- generated by scripts/tuned_tables.py from {label or report_dir.name} -->", ""]
    out += [f"**Settings searched** on the `{main}` training part "
            f"({card['calibration']['folds']}-fold patient-grouped cross-validation, training data only)", ""]
    out += [search_section(search, main), ""]
    out += [f"**Validation, `{main}` split** (chooses the cut-off and the saved model; calibrated and "
            "uncalibrated rows are the same fitted model)", ""]
    out += [validation_section(validation, main, seed), ""]
    out += [f"**Test, `{main}` split** ({int(first['n']):,} samples, {int(first['n_resistant'])} resistant; "
            f"seed {seed}; 95 % intervals from {intervals[main]['used']} patient-level resamples)", ""]
    out += [test_section(intervals, main, chosen, older[-1].version), ""]
    out += ["**All models compared** (test part, seed 42)", ""]
    out += [comparison_section(reports, main, seed, note), ""]

    overlap_path = report_dir / "patient_overlap_check.csv"
    if overlap_path.is_file():
        overlap = pd.read_csv(overlap_path)
        out += ["**Patient-overlap check** (the chosen family re-tuned on each training part)", ""]
        rows = [{"Comparison": r.comparison, "Note": r.note,
                 "AUROC difference": f"{r.roc_auc_difference:+.3f} "
                                     f"[{r.roc_auc_difference_low:+.3f}, {r.roc_auc_difference_high:+.3f}]",
                 "PR-AUC difference": f"{r.pr_auc_difference:+.3f} "
                                      f"[{r.pr_auc_difference_low:+.3f}, {r.pr_auc_difference_high:+.3f}]"}
                for r in overlap.itertuples()]
        out += [md_table(rows), ""]

    seed_path = report_dir / "seed_variation.csv"
    if seed_path.is_file() and pd.read_csv(seed_path).shape[0]:
        variation = pd.read_csv(seed_path)
        out += ["**Seed variation** (test part; mean and range over the seeds)", ""]
        rows = [{"Experiment": r.experiment, "Model": model_name(r.model), "Seeds": int(r.seeds),
                 "AUROC": f"{r.roc_auc_mean:.3f} ({r.roc_auc_min:.3f}–{r.roc_auc_max:.3f})",
                 "PR-AUC": f"{r.pr_auc_mean:.3f} ({r.pr_auc_min:.3f}–{r.pr_auc_max:.3f})",
                 "Sensitivity": f"{r.sensitivity_mean:.3f} ({r.sensitivity_min:.3f}–{r.sensitivity_max:.3f})"}
                for r in variation.itertuples()]
        out += [md_table(rows), ""]

    out += [f"**Prediction time** of the saved model ({timing['samples']} validation spectra read from raw files, "
            f"{timing['single_spectrum_threads']} thread, milliseconds)", ""]
    out += [md_table([{"Step": step, "Median": timing[key]["median"], "95th percentile": timing[key]["p95"],
                       "Maximum": timing[key]["max"]}
                      for step, key in (("Read + preprocess", "preprocessing_ms"), ("Model", "inference_ms"),
                                        ("Total", "total_ms"))]), ""]
    out += [f"Batch of {timing['batch_size']} spectra with all CPU threads: "
            f"{timing['batch_inference_ms_per_sample']} ms per spectrum (model step only). "
            f"File-based predictions match the stored features: {timing['matches_stored_features']}.", ""]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Markdown tables of the tuned-model results.")
    parser.add_argument("--section", choices=sorted(SECTIONS), default="tuning",
                        help="which results to tabulate (default: tuning = Version 0.4)")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    meta = SECTIONS[args.section]
    dataset = args.dataset or config["dataset"]["name"]
    relative = f"{config[args.section]['report_dir']}/{dataset}"
    report_dir = project_path(relative)
    previous = [(version, project_path(config[section]["report_dir"]) / dataset)
                for version, section in meta["previous"]]
    for folder in (report_dir, *(p for _, p in previous)):
        if not (folder / "test_metrics.csv").is_file():
            print(f"Error: {folder / 'test_metrics.csv'} is missing; run the test evaluation first.",
                  file=sys.stderr)
            return 1
    text = build_tables(report_dir, previous, meta["version"], meta["note"], relative)
    (report_dir / "tables.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
