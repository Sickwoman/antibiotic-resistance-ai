"""Version 0.3 - baseline models: prevalence only, logistic regression, random forest, LightGBM.

Run from the project root with the virtual environment active:

    python scripts/train_baselines.py                   # train + validation only; test parts untouched
    python scripts/train_baselines.py --evaluate-test   # also score the allowed test parts (logged)

Rules from docs/evaluation_protocol.md:
- Every learned step is fitted on training rows only.
- Thresholds and the choice of the saved model come from validation data.
- Test parts are scored only with --evaluate-test, and every evaluation is appended to the test log.
- The test parts of evaluation.locked_test_splits (temporal, external) stay locked until Version 0.7.

Writes:
- results/metrics/v0.3/<dataset>/* and results/plots/v0.3/* (committed; no identifiers)
- results/experiments/test_evaluations.csv (committed, append-only)
- models/v0.3/<dataset>/best_random.joblib and its .json card (git-ignored)
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import exploration as ex  # noqa: E402  (plot style only)
from src.data_loader import DataError  # noqa: E402
from src.dataset import load_dataset, resolve_relpath  # noqa: E402
from src.evaluate import (  # noqa: E402
    EvaluationError,
    append_test_log,
    reliability_table,
    summarize_bootstrap,
    unpaired_difference,
)
from src.model_plots import plot_confusion  # noqa: E402
from src.predict import (  # noqa: E402
    BUNDLE_FORMAT,
    ModelError,
    card,
    load_bundle,
    predict_features,
    predict_spectrum_file,
    save_bundle,
)
from src.splits import SplitError, assert_usable, load_splits  # noqa: E402
from src.train import (  # noqa: E402
    RunResult,
    TrainingError,
    grouped_subsample,
    load_rows,
    model_specs,
    run_experiment,
    select_best,
)
from src.utils import (  # noqa: E402
    ConfigError,
    driams_root,
    get_logger,
    git_commit,
    keep_awake,
    load_config,
    project_path,
    set_seed,
    show_path,
)

log = get_logger("baselines")
STAGE = "v0.3-baselines"
DISPLAY = {"prevalence": "Prevalence only", "logistic_regression": "Logistic regression",
           "random_forest": "Random forest", "lightgbm": "LightGBM"}
SHOWN = ["experiment", "model", "seed", "train_size", "fit_seconds", "threshold", "roc_auc", "pr_auc", "sensitivity",
         "specificity", "precision", "f1", "accuracy", "brier", "calibration_slope"]


def section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def show(df: pd.DataFrame) -> None:
    """Print a table; thresholds keep 3 significant digits because some are tiny (e.g. 2e-05)."""
    formatters = {"threshold": "{:.3g}".format} if "threshold" in df.columns else None
    with pd.option_context("display.max_columns", 40, "display.width", 250, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False, formatters=formatters))


def versions() -> dict[str, str]:
    import joblib
    import lightgbm
    import sklearn

    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "joblib": joblib.__version__}


# ------------------------------------------------------------------------------------------------ figures

def plot_roc_pr(results: list[RunResult], y: np.ndarray, title: str, path: Path) -> Path:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    ex.apply_style()
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))
    prevalence = float(y.mean())
    colors = iter(ex.SERIES)
    for r in results:
        if r.model == "prevalence":
            continue
        color = next(colors)
        fpr, tpr, _ = roc_curve(y, r.test_prob)
        ax_roc.plot(fpr, tpr, color=color, lw=1.8, label=f"{DISPLAY.get(r.model, r.model)} ({r.test['roc_auc']:.3f})")
        precision, recall, _ = precision_recall_curve(y, r.test_prob)
        ax_pr.plot(recall, precision, color=color, lw=1.8, drawstyle="steps-post",
                   label=f"{DISPLAY.get(r.model, r.model)} ({r.test['pr_auc']:.3f})")
        ax_roc.plot(1 - r.test["specificity"], r.test["sensitivity"], "o", ms=6, color=color,
                    markeredgecolor=ex.SURFACE, markeredgewidth=1.5)
    ax_roc.plot([0, 1], [0, 1], ls="--", lw=1, color=ex.MUTED, label="No skill (0.500)")
    ax_pr.axhline(prevalence, ls="--", lw=1, color=ex.MUTED, label=f"No skill ({prevalence:.3f})")
    ax_roc.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="1 - specificity", ylabel="Sensitivity (resistant flagged)")
    ax_pr.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall (sensitivity)", ylabel="Precision")
    ax_roc.set_title("ROC curve", pad=8)
    ax_pr.set_title("Precision-recall curve", pad=8)
    ax_roc.legend(title="Model (AUROC)", loc="lower right")
    ax_pr.legend(title="Model (PR-AUC)", loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.text(0.01, 0.905, "Dots: operating point at the threshold chosen on validation (sensitivity >= 0.90 there)",
             fontsize=9, color=ex.INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return ex._save(fig, path)


def plot_calibration(results: list[RunResult], y: np.ndarray, n_bins: int, title: str, path: Path) -> Path:
    import matplotlib.pyplot as plt

    ex.apply_style()
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=ex.MUTED, label="Perfect calibration")
    colors = iter(ex.SERIES)
    for r in results:
        if r.model == "prevalence":
            continue
        table = reliability_table(y, r.test_prob, n_bins)
        ax.plot(table["mean_predicted"], table["observed_rate"], "-o", color=next(colors), lw=1.8, ms=6,
                markeredgecolor=ex.SURFACE, markeredgewidth=1.5,
                label=f"{DISPLAY.get(r.model, r.model)} (Brier {r.test['brier']:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted resistance probability", ylabel="Observed resistant share")
    ax.set_title(title, pad=22)
    ex._subtitle(ax, f"{n_bins} bins with equal numbers of test samples")
    ax.legend(loc="upper left")
    return ex._save(fig, path)


# ------------------------------------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the Version 0.3 baseline models.")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="score the test parts once (appended to the test log); without it only validation is used")
    parser.add_argument("--allow-rescore", action="store_true",
                        help="with --evaluate-test: score again although test rows for this dataset are logged")
    parser.add_argument("--dataset", default=None, help="dataset folder name (default: dataset.name)")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        seed = int(config["project"]["random_seed"])
        set_seed(seed)
        ev, bl = config["evaluation"], config["baselines"]
        dataset_name = args.dataset or config["dataset"]["name"]
        data_dir = project_path(config["dataset"]["output_dir"]) / dataset_name
        X, meta, summary = load_dataset(data_dir, verify_x=True)
        splits = load_splits(data_dir / "splits", meta)
        requested = list(bl["splits"])
        sm = bl.get("size_matched")
        needed = requested + ([sm["split"], sm["size_of"]] if sm else [])
        missing = sorted({s for s in needed if s not in splits})
        if missing:
            raise SplitError(f"Split(s) {missing} are not saved for {dataset_name}; rebuild the dataset.")
        locked = set(ev["locked_test_splits"]) & set(needed)
        if locked and args.evaluate_test:
            raise SplitError(f"The test parts of {sorted(locked)} are locked until Version 0.7 "
                             "(docs/evaluation_protocol.md); remove them from baselines.")
        log_path = project_path(ev["test_log"])
        if args.evaluate_test and log_path.is_file() and log_path.stat().st_size:
            done = pd.read_csv(log_path)
            done = done[(done["stage"] == STAGE) & (done["dataset"] == dataset_name)]
            if len(done) and not args.allow_rescore:
                raise EvaluationError(f"{len(done)} Version 0.3 test evaluations of {dataset_name} are already "
                                      "logged. Scoring again is a second look at the test data; pass "
                                      "--allow-rescore only on purpose (the rows are logged as another run).")
        specs = model_specs(config)
        seeds = [int(s) for s in bl["seeds"]]
        y = meta["label"].to_numpy().astype(np.int64)

        for name in requested:
            assert_usable(meta, splits[name])          # no empty or single-class part reaches a model
        experiments = [(name, name, splits[name].train, splits[name].validation, splits[name].test)
                       for name in requested]
        if sm:
            ref = splits[sm["split"]]
            target = len(splits[sm["size_of"]].train)
            rows = grouped_subsample(meta, ref.train, target, seed)
            experiments.append((f"{sm['split']}_size_matched", sm["split"], rows, ref.validation, ref.test))

        section(f"Version 0.3 baselines on {dataset_name} ({summary['n_samples']:,} samples, rows fingerprint "
                f"{summary['row_fingerprint']}); test parts {'WILL' if args.evaluate_test else 'will NOT'} be scored")
        for name, split, tr, va, te in experiments:
            print(f"{name:<22} split {split:<12} train {len(tr):>5} ({int(y[tr].sum())} resistant)  "
                  f"validation {len(va):>4}  test {len(te):>4}{'' if args.evaluate_test else ' (not used)'}")

        started = time.monotonic()
        results: list[RunResult] = []
        with keep_awake():
            for name, split, tr, va, te in experiments:
                results += run_experiment(X, meta, experiment=name, split=split, train_rows=tr, validation_rows=va,
                                          test_rows=te if args.evaluate_test else None, specs=specs, seeds=seeds,
                                          threshold_rule=ev["threshold"], log=log)
        train_seconds = round(time.monotonic() - started, 1)

        report_dir = project_path(bl["report_dir"]) / dataset_name
        plot_dir = project_path(bl["plot_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        validation = pd.DataFrame([r.row("validation") for r in results])
        validation.to_csv(report_dir / "validation_metrics.csv", index=False, lineterminator="\n")
        section("Validation results (thresholds chosen here; seed 42 shown)")
        show(validation.loc[validation["seed"] == seeds[0], [c for c in SHOWN if c in validation.columns]])

        primary = ev["primary_metric"]
        main_name = requested[0]
        main_runs = [r for r in results if r.experiment == main_name]
        best = select_best(main_runs, primary, seeds[0])
        print(f"\nSaved model = highest validation {primary} on '{main_name}': {DISPLAY.get(best.model, best.model)} "
              f"({best.validation[primary]:.3f})")

        test_table = None
        intervals: dict[str, Any] = {}
        if args.evaluate_test:
            commit = git_commit()
            append_test_log(project_path(ev["test_log"]), [           # log first: a later failure cannot hide it
                {"logged_at": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": commit, "stage": STAGE,
                 "dataset": dataset_name, "dataset_fingerprint": summary["row_fingerprint"],
                 "x_sha256": summary["x_sha256"],
                 **{k: v for k, v in r.row("test").items() if k not in ("fit_seconds", "predict_ms_per_sample")}}
                for r in results], locked=ev["locked_test_splits"])
            print(f"\n{len(results)} test evaluations appended to {ev['test_log']}")
            test_table = pd.DataFrame([r.row("test") for r in results])
            test_table.to_csv(report_dir / "test_metrics.csv", index=False, lineterminator="\n")
            section("Test results (each test part scored once; seed 42 shown)")
            show(test_table.loc[test_table["seed"] == seeds[0], [c for c in SHOWN if c in test_table.columns]])

            b = ev["bootstrap"]
            samples: dict[str, Any] = {}
            for name, _split, _tr, _va, te in experiments:
                runs = [r for r in results if r.experiment == name and r.seed == seeds[0]]
                summary_ci, samples[name] = summarize_bootstrap(
                    y[te], meta["group_id"].to_numpy()[te], {r.model: r.test_prob for r in runs},
                    {r.model: r.threshold for r in runs}, resamples=int(b["resamples"]), level=float(b["level"]),
                    seed=int(b["seed"]), reference=best.model)
                intervals[name] = summary_ci
            (report_dir / "test_intervals.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
            section(f"95 % intervals (patient-level bootstrap, {b['resamples']} resamples), seed 42")
            ci_rows = []
            for name, s in intervals.items():
                for model, m in s["intervals"].items():
                    row = {"experiment": name, "model": model}
                    for metric in ("roc_auc", "pr_auc", "sensitivity", "specificity"):
                        row[metric] = f"{m[metric]['estimate']:.3f} [{m[metric]['low']:.3f}, {m[metric]['high']:.3f}]"
                    diff = s["differences_to_reference"].get(model, {}).get("roc_auc")
                    row[f"AUROC {best.model} minus this"] = (
                        "-" if diff is None else f"{diff['estimate']:+.3f} [{diff['low']:+.3f}, {diff['high']:+.3f}]")
                    ci_rows.append(row)
                print(f"{name}: {s['used']} resamples used, {s['skipped_single_class']} skipped (one class only)")
            ci_table = pd.DataFrame(ci_rows)
            ci_table.to_csv(report_dir / "test_intervals.csv", index=False, lineterminator="\n")
            show(ci_table)

            stochastic = {spec.name for spec in specs if spec.stochastic}
            seed_rows = []
            for (name, model), g in test_table[test_table["model"].isin(stochastic)].groupby(["experiment", "model"]):
                row = {"experiment": name, "model": model, "seeds": len(g)}
                for metric in ("roc_auc", "pr_auc", "sensitivity", "specificity"):
                    row.update({f"{metric}_mean": g[metric].mean(), f"{metric}_min": g[metric].min(),
                                f"{metric}_max": g[metric].max()})
                seed_rows.append(row)
            seed_table = pd.DataFrame(seed_rows)
            seed_table.to_csv(report_dir / "seed_variation.csv", index=False, lineterminator="\n")
            section(f"Seed variation of stochastic models on test (seeds {seeds})")
            show(seed_table)

            if sm:
                matched = f"{sm['split']}_size_matched"
                overlap_rows = []
                level = float(b["level"])
                for spec in specs:
                    row = {"model": spec.name}
                    for metric in ("roc_auc", "pr_auc"):
                        w = intervals[sm["size_of"]]["intervals"][spec.name][metric]
                        m = intervals[matched]["intervals"][spec.name][metric]
                        low, high = unpaired_difference(samples[matched][spec.name][metric],
                                                        samples[sm["size_of"]][spec.name][metric], level)
                        row.update({f"{metric}_{matched}": m["estimate"], f"{metric}_{sm['size_of']}": w["estimate"],
                                    f"{metric}_difference": m["estimate"] - w["estimate"],
                                    f"{metric}_difference_low": low, f"{metric}_difference_high": high})
                    overlap_rows.append(row)
                overlap = pd.DataFrame(overlap_rows)
                overlap.to_csv(report_dir / "patient_overlap_check.csv", index=False, lineterminator="\n")
                section(f"Patient-overlap check: {matched} (years pooled) minus {sm['size_of']} (one year), "
                        "same training size; difference with 95 % interval")
                show(overlap)

            main_te = splits[main_name].test
            main_seed = [r for r in main_runs if r.seed == seeds[0]]
            title = (f"{summary['species']} + {summary['antibiotic']}: baselines, {main_name} split, test part "
                     f"(n={len(main_te):,}, {int(y[main_te].sum())} resistant)")
            plots = [plot_roc_pr(main_seed, y[main_te], title, plot_dir / f"{dataset_name}_{main_name}_roc_pr.png"),
                     plot_calibration(main_seed, y[main_te], int(ev["calibration_bins"]),
                                      f"Calibration, {main_name} split (test)",
                                      plot_dir / f"{dataset_name}_{main_name}_calibration.png"),
                     plot_confusion(best.threshold, best.test,
                                    f"{DISPLAY.get(best.model, best.model)}, {main_name} split (test)",
                                    plot_dir / f"{dataset_name}_{main_name}_confusion_{best.model}.png")]
            for p in plots:
                print(f"saved {show_path(p)}")

        # -------------------------------------------------------------------- save the chosen model
        spec = next(s for s in specs if s.name == best.model)
        train_rows = splits[best.split].train
        bundle = {
            "format": BUNDLE_FORMAT,
            "model_version": f"v{config['project']['version']}-{best.model}-{best.experiment}-seed{best.seed}",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": git_commit(),
            "pipeline": best.pipeline, "model": best.model, "model_kind": spec.kind, "params": spec.params,
            "scaled": spec.scale, "seed": best.seed,
            "species": summary["species"], "antibiotic": summary["antibiotic"],
            "label_map": {"0": summary["label_definition"]["0"], "1": summary["label_definition"]["1"]},
            "threshold": best.threshold, "threshold_rule": ev["threshold"],
            "preprocessing": summary["preprocessing"], "feature_fingerprint": summary["feature_fingerprint"],
            "n_features": int(summary["n_features"]),
            "dataset": {"name": dataset_name, "row_fingerprint": summary["row_fingerprint"],
                        "x_sha256": summary["x_sha256"], "sites": summary["sites"],
                        "intermediate_as": summary["intermediate_as"]},
            "split": {"name": best.split, "experiment": best.experiment, "train_size": best.train_size,
                      "dataset_fingerprint": splits[best.split].dataset_fingerprint,
                      "train_sites": sorted(meta["site"].iloc[train_rows].unique().tolist())},
            "selection": {"metric": primary, "part": "validation",
                          "candidates": {r.model: r.validation[primary] for r in main_runs if r.seed == seeds[0]}},
            "metrics": {"validation": best.validation, "test": best.test},
            "versions": versions(),
        }
        model_path = save_bundle(bundle, project_path(bl["model_dir"]) / dataset_name / f"best_{main_name}.joblib")
        (report_dir / "best_model_card.json").write_text(json.dumps(card(bundle), indent=2, default=str),
                                                         encoding="utf-8")
        print(f"\nSaved {show_path(model_path)} ({model_path.stat().st_size / 1e6:.2f} MB)")

        # -------------------------------------------------------------------- timing with the saved file
        loaded = load_bundle(model_path)
        rng = np.random.default_rng(seed)
        val_rows = splits[main_name].validation
        timed_rows = rng.choice(val_rows, size=min(int(ev["timing_samples"]), val_rows.size), replace=False)
        root = driams_root(config)
        predict_spectrum_file(loaded, resolve_relpath(root, meta["spectrum_relpath"].iloc[timed_rows[0]]))  # warm-up
        timings = pd.DataFrame([predict_spectrum_file(loaded, resolve_relpath(root, rel)).to_dict()
                                for rel in meta["spectrum_relpath"].iloc[timed_rows]])
        agrees = bool(np.allclose(timings["resistance_probability"],
                                  np.round(predict_features(loaded, load_rows(X, timed_rows))[0], 4), atol=1e-4))
        batch_bundle = load_bundle(model_path, n_jobs=None)          # saved setting (all CPU threads)
        batch = load_rows(X, val_rows)
        predict_features(batch_bundle, batch[:10])                   # warm-up
        started = time.perf_counter()
        predict_features(batch_bundle, batch)
        batch_ms = (time.perf_counter() - started) * 1000 / len(batch)
        timing = {"samples": len(timings), "model": loaded["model_version"],
                  "matches_stored_features": agrees,
                  "single_spectrum_threads": loaded["pipeline"].steps[-1][1].get_params().get("n_jobs"),
                  "batch_threads": batch_bundle["pipeline"].steps[-1][1].get_params().get("n_jobs"),
                  "batch_size": len(batch), "batch_inference_ms_per_sample": round(batch_ms, 4)}
        for col in ("preprocessing_ms", "inference_ms", "total_ms"):
            timing[col] = {"median": round(float(timings[col].median()), 2),
                           "p95": round(float(timings[col].quantile(0.95)), 2),
                           "max": round(float(timings[col].max()), 2)}
        (report_dir / "inference_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        section(f"Prediction time with the saved model ({len(timings)} validation spectra, read from raw files)")
        print(json.dumps(timing, indent=2))

        run_config = {"stage": STAGE, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": git_commit(),
                      "dataset": dataset_name, "rows_fingerprint": summary["row_fingerprint"],
                      "x_sha256": summary["x_sha256"], "test_parts_scored": args.evaluate_test,
                      "experiments": {n: {"split": s, "train": len(tr), "validation": len(va), "test": len(te)}
                                      for n, s, tr, va, te in experiments},
                      "evaluation": ev, "baselines": bl, "training_seconds": train_seconds, "versions": versions()}
        (report_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        print(f"\nReports: {report_dir}\nThis is a research prototype; predictions are not clinical results.")
        return 0
    except (DataError, ConfigError, SplitError, TrainingError, EvaluationError, ModelError) as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:                                 # noqa: BLE001 - a run must never end silently
        log.exception("The run stopped with an unexpected error: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
