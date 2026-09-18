"""Markdown tables of the Version 0.4 results, generated from the saved reports (never typed by hand).

Run from the project root with the virtual environment active, after
`python scripts/tune_models.py --evaluate-test`:

    python scripts/tuned_tables.py

Writes results/metrics/v0.4/<dataset>/tables.md and prints it. The comparison table asked for in the
project plan (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, training time, inference time) holds the
Version 0.3 baselines and the Version 0.4 models side by side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tables import md_table, model_name, number, seconds, signed_ci, with_ci  # noqa: E402
from src.utils import ConfigError, load_config, project_path  # noqa: E402


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


def test_section(intervals: dict[str, Any], experiment: str, chosen: str) -> str:
    entry = intervals[experiment]
    rows = []
    for model, m in entry["intervals"].items():
        row = {"Model": model_name(model) + (" (saved)" if model == chosen else ""),
               "AUROC": with_ci(m["roc_auc"]), "PR-AUC": with_ci(m["pr_auc"]),
               "Sensitivity": with_ci(m["sensitivity"]), "Specificity": with_ci(m["specificity"])}
        best = entry.get("differences_to_reference", {}).get(model, {}).get("roc_auc")
        older = entry.get("differences_to_version_0_3", {}).get(model, {}).get("roc_auc")
        row["AUROC: saved model minus this"] = "–" if best is None else signed_ci(best)
        row["AUROC: this minus Version 0.3"] = "–" if older is None else signed_ci(older)
        rows.append(row)
    return md_table(rows)


def comparison_section(v03: pd.DataFrame, v04: pd.DataFrame, search: pd.DataFrame, main: str, seed: int,
                       v03_timing: dict[str, Any], v04_timing: dict[str, Any]) -> str:
    """The table the project plan asks for, with the Version 0.3 and Version 0.4 models side by side."""
    search_time = {r.family: r.search_seconds for r in search[search["experiment"] == main].itertuples()}
    rows = []
    for version, table in (("0.3", v03), ("0.4", v04)):
        for r in seed_rows(table, main, seed).itertuples():
            if version == "0.4" and str(r.model).startswith("v0.3_"):
                continue                      # the Version 0.3 rows are listed from their own report
            family = str(r.model).removeprefix("tuned_")
            rows.append({
                "Version": version, "Model": model_name(r.model), "Accuracy": number(r.accuracy),
                "Precision": number(r.precision), "Recall (= sensitivity)": number(r.recall),
                "F1": number(r.f1), "ROC-AUC": number(r.roc_auc), "PR-AUC": number(r.pr_auc),
                "Search time": seconds(search_time.get(family)) if version == "0.4" else "–",
                "Training time": seconds(getattr(r, "fit_seconds", None)),
                "Inference (ms/spectrum, batch)": number(getattr(r, "predict_ms_per_sample", None), 3)})
    note = (f"\nTraining time: Version 0.3 is one fit; Version 0.4 covers the cross-fitted models used for "
            f"calibration plus the final fit, and excludes the search. Inference is the batch time per spectrum "
            f"on the test part (Version 0.3 with all CPU threads, Version 0.4 with one). End-to-end prediction "
            f"from a raw file: {v03_timing['total_ms']['median']} ms (Version 0.3) and "
            f"{v04_timing['total_ms']['median']} ms (Version 0.4), median.\n")
    return md_table(rows) + "\n" + note


def build_tables(report_dir: Path, v03_dir: Path, label: str | None = None) -> str:
    card = json.loads((report_dir / "best_model_card.json").read_text(encoding="utf-8"))
    main, seed, chosen = card["split"]["experiment"], card["seed"], card["model"]
    search = pd.read_csv(report_dir / "search_summary.csv")
    validation = pd.read_csv(report_dir / "validation_metrics.csv")
    test = pd.read_csv(report_dir / "test_metrics.csv")
    intervals = json.loads((report_dir / "test_intervals.json").read_text(encoding="utf-8"))
    timing = json.loads((report_dir / "inference_timing.json").read_text(encoding="utf-8"))
    v03_test = pd.read_csv(v03_dir / "test_metrics.csv")
    v03_timing = json.loads((v03_dir / "inference_timing.json").read_text(encoding="utf-8"))
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
    out += [test_section(intervals, main, chosen), ""]
    out += ["**All models compared** (test part, seed 42)", ""]
    out += [comparison_section(v03_test, test, search, main, seed, v03_timing, timing), ""]

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
    parser = argparse.ArgumentParser(description="Markdown tables of the Version 0.4 results.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    dataset = args.dataset or config["dataset"]["name"]
    relative = f"{config['tuning']['report_dir']}/{dataset}"
    report_dir, v03_dir = project_path(relative), project_path(config["baselines"]["report_dir"]) / dataset
    for folder, name in ((report_dir, "test_metrics.csv"), (v03_dir, "test_metrics.csv")):
        if not (folder / name).is_file():
            print(f"Error: {folder / name} is missing; run the test evaluation first.", file=sys.stderr)
            return 1
    text = build_tables(report_dir, v03_dir, relative)
    (report_dir / "tables.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
