"""Version 0.4 - tuned and calibrated models, following docs/v0.4_search_plan.md.

Run from the project root with the virtual environment active (keep the laptop plugged in, lid open):

    python scripts/tune_models.py                   # searches, calibration, validation; test parts untouched
    python scripts/tune_models.py --evaluate-test   # also score the planned test parts once (logged)
    python scripts/tune_models.py --families logistic_regression   # development run on some families only

Searches and fitted models are cached in models/v0.4/cache, keyed by the data rows, the settings, the
tuning code and the library versions, so the --evaluate-test run reuses the development run's work.

Writes:
- results/metrics/v0.4/<dataset>/* and results/plots/v0.4/* (committed; no identifiers)
- results/experiments/test_evaluations.csv (append-only)
- models/v0.4/<dataset>/best_random.joblib and its .json card (git-ignored)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Single fits inside the searches may warn (e.g. convergence); the chosen models are checked explicitly.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::FutureWarning")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import exploration as ex  # noqa: E402  (plot style only)
from src.data_loader import DataError  # noqa: E402
from src.dataset import load_dataset, resolve_relpath  # noqa: E402
from src.evaluate import (  # noqa: E402
    EvaluationError,
    append_test_log,
    choose_threshold,
    classification_metrics,
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
from src.splits import SplitError, load_splits  # noqa: E402
from src.train import TrainingError, grouped_subsample, load_rows  # noqa: E402
from src.tuning import (  # noqa: E402
    FamilySpec,
    TuningError,
    cache_key,
    cached,
    code_fingerprint,
    converged,
    family_specs,
    fit_calibrated,
    grouped_folds,
    run_search,
    set_threads,
    uncalibrated,
)
from src.utils import (  # noqa: E402
    PROJECT_ROOT,
    ConfigError,
    driams_root,
    get_logger,
    git_commit,
    keep_awake,
    load_config,
    project_path,
    set_seed,
)

log = get_logger("tuning")
STAGE = "v0.4-tuned"
DISPLAY = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
           "lightgbm": "LightGBM", "svm_rbf": "SVM (RBF)"}
SHOWN = ["experiment", "model", "seed", "calibrated", "train_size", "fit_seconds", "threshold", "roc_auc", "pr_auc",
         "sensitivity", "specificity", "precision", "f1", "accuracy", "brier", "calibration_slope"]


def section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def show(df: pd.DataFrame) -> None:
    formatters = {"threshold": "{:.3g}".format} if "threshold" in df.columns else None
    with pd.option_context("display.max_columns", 40, "display.width", 260, "display.max_colwidth", 90,
                           "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False, formatters=formatters))


def versions() -> dict[str, str]:
    import joblib
    import lightgbm
    import sklearn

    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "joblib": joblib.__version__}


def rows_hash(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()[:16]


def readable(setting: dict[str, Any]) -> str:
    return json.dumps({k: setting[k] for k in sorted(setting)}, default=str)


@dataclass
class Finalist:
    experiment: str
    split: str
    family: str
    seed: int
    setting: dict[str, Any]
    model: Any
    train_size: int
    fit_seconds: float
    threshold: float
    validation: dict[str, Any]
    validation_uncalibrated: dict[str, Any] | None
    validation_prob: np.ndarray
    converged: bool | None
    test_prob: np.ndarray | None = None
    test: dict[str, Any] | None = None
    predict_ms_per_sample: float = float("nan")

    @property
    def name(self) -> str:
        return f"tuned_{self.family}"

    def row(self, part: str, calibrated: bool = True) -> dict[str, Any]:
        metrics = (self.validation if calibrated else self.validation_uncalibrated) if part == "validation" \
            else self.test
        return {"experiment": self.experiment, "split": self.split, "model": self.name, "seed": self.seed,
                "calibrated": calibrated, "setting": readable(self.setting), "train_size": self.train_size,
                "fit_seconds": round(self.fit_seconds, 2), "converged": self.converged,
                "predict_ms_per_sample": round(self.predict_ms_per_sample, 4), **(metrics or {})}


class Context:
    """Data, settings and cache locations shared by all experiments of one run."""

    def __init__(self, config: dict[str, Any], dataset_name: str, evaluate_test: bool):
        self.config = config
        self.tc, self.ev, self.bl = config["tuning"], config["evaluation"], config["baselines"]
        self.dataset_name = dataset_name
        self.evaluate_test = evaluate_test
        data_dir = project_path(config["dataset"]["output_dir"]) / dataset_name
        self.X, self.meta, self.summary = load_dataset(data_dir, verify_x=True)
        self.splits = load_splits(data_dir / "splits", self.meta)
        self.y = self.meta["label"].to_numpy().astype(np.int64)
        self.groups = self.meta["group_id"].to_numpy()
        self.seed = int(config["project"]["random_seed"])
        self.cv_seed = int(self.tc["cv_seed"])
        self.seeds = [int(s) for s in self.tc["seeds"]]
        self.model_dir = project_path(self.tc["model_dir"]) / dataset_name
        self.cache_dir = project_path(self.tc["model_dir"]) / "cache" / dataset_name
        self.report_dir = project_path(self.tc["report_dir"]) / dataset_name
        self.plot_dir = project_path(self.tc["plot_dir"])
        self.code = code_fingerprint([PROJECT_ROOT / "src" / "tuning.py", PROJECT_ROOT / "src" / "train.py"])
        self.versions = versions()

    def experiment(self, name: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        """(split name, train, validation, test rows) of a named experiment."""
        if name in self.splits:
            s = self.splits[name]
            return name, s.train, s.validation, s.test
        sm = self.bl["size_matched"]
        if name == f"{sm['split']}_size_matched":
            ref = self.splits[sm["split"]]
            rows = grouped_subsample(self.meta, ref.train, len(self.splits[sm["size_of"]].train), self.seed)
            return sm["split"], rows, ref.validation, ref.test
        raise SplitError(f"Unknown experiment {name!r}")


def tune_experiment(ctx: Context, name: str, specs: list[FamilySpec]) -> tuple[list[dict[str, Any]], list[Finalist]]:
    split, tr, va, te = ctx.experiment(name)
    locked = set(ctx.ev["locked_test_splits"])
    if ctx.evaluate_test and split in locked:
        raise SplitError(f"The {split} test part is locked until Version 0.7 (docs/evaluation_protocol.md).")
    X_tr, y_tr = load_rows(ctx.X, tr), ctx.y[tr]
    X_va, y_va = load_rows(ctx.X, va), ctx.y[va]
    X_te = load_rows(ctx.X, te) if ctx.evaluate_test else None
    folds = grouped_folds(y_tr, ctx.groups[tr], int(ctx.tc["cv_folds"]), ctx.cv_seed)
    base_key = {"dataset": ctx.summary["row_fingerprint"], "experiment": name, "train": rows_hash(tr),
                "folds": int(ctx.tc["cv_folds"]), "cv_seed": ctx.cv_seed, "metric": ctx.tc["search_metric"],
                "code": ctx.code, "versions": ctx.versions}
    summary_rows, finalists = [], []
    ctx.report_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        key = cache_key(**base_key, family=spec.fingerprint())
        log.info("%s / %s: searching %s", name, spec.name, "(or loading the cached search)")
        search = cached(ctx.cache_dir / name / spec.name / "search.joblib", key,
                        lambda spec=spec: run_search(spec, X_tr, y_tr, folds, ctx.cv_seed, ctx.tc["search_metric"]),
                        log)
        search.table.to_csv(ctx.report_dir / f"search_{name}_{spec.name}.csv", index=False, lineterminator="\n")
        best = search.table.loc[search.best_index]
        summary_rows.append({"experiment": name, "family": spec.name, "settings": len(search.table),
                             "best_setting": readable(search.best_setting),
                             "cv_roc_auc_mean": best["cv_roc_auc_mean"], "cv_roc_auc_sd": best["cv_roc_auc_sd"],
                             "cv_pr_auc_mean": best["cv_pr_auc_mean"], "search_seconds": round(search.seconds, 1)})
        log.info("%s / %s: best of %d settings %s, CV AUROC %.3f (search %.0f s)", name, spec.name,
                 len(search.table), readable(search.best_setting), best["cv_roc_auc_mean"], search.seconds)
        for seed in (ctx.seeds if spec.stochastic else ctx.seeds[:1]):
            fit_key = cache_key(search=key, seed=seed, calibration=ctx.tc["calibration"])

            def fit(spec=spec, setting=search.best_setting, seed=seed):
                started = time.perf_counter()
                model = fit_calibrated(spec, setting, X_tr, y_tr, folds, seed, ctx.tc["calibration"])
                return model, time.perf_counter() - started

            model, fit_seconds = cached(ctx.cache_dir / name / spec.name / f"fit_seed{seed}.joblib", fit_key, fit, log)
            set_threads(model, 1)            # single-thread prediction: bit-identical results in every run
            val_prob = model.predict_proba(X_va)[:, 1]
            threshold = choose_threshold(y_va, val_prob, ctx.ev["threshold"])
            inner = uncalibrated(model)
            validation_uncal = None
            if hasattr(inner, "predict_proba"):
                raw = inner.predict_proba(X_va)[:, 1]
                validation_uncal = classification_metrics(y_va, raw, choose_threshold(y_va, raw, ctx.ev["threshold"]))
            f = Finalist(name, split, spec.name, seed, search.best_setting, model, int(len(tr)), fit_seconds,
                         threshold, classification_metrics(y_va, val_prob, threshold), validation_uncal, val_prob,
                         converged(model))
            if X_te is not None:
                started = time.perf_counter()
                f.test_prob = model.predict_proba(X_te)[:, 1]
                f.predict_ms_per_sample = (time.perf_counter() - started) * 1000 / len(te)
                f.test = classification_metrics(ctx.y[te], f.test_prob, threshold)
            log.info("%s / %s (seed %d): fitted in %.0f s, validation AUROC %.3f", name, spec.name, seed,
                     fit_seconds, f.validation["roc_auc"])
            finalists.append(f)
    return summary_rows, finalists


# ------------------------------------------------------------------------------------------------ figures

def plot_search(summary_tables: dict[str, pd.DataFrame], title: str, path: Path) -> Path:
    import matplotlib.pyplot as plt

    ex.apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    rng = np.random.default_rng(0)
    for i, table in enumerate(summary_tables.values()):
        scores = table["cv_roc_auc_mean"].to_numpy()
        x = i + rng.uniform(-0.18, 0.18, scores.size)
        ax.scatter(x, scores, s=22, color=ex.SERIES[i], alpha=0.75, edgecolor=ex.SURFACE, linewidth=0.8)
        best = int(np.nanargmax(scores))
        ax.scatter([i], [scores[best]], s=90, marker="D", color=ex.SERIES[i], edgecolor=ex.INK, linewidth=1,
                   zorder=3)
        ax.text(i + 0.26, scores[best], f"{scores[best]:.3f}", va="center", fontsize=9, color=ex.INK_2)
    ax.set_xticks(range(len(summary_tables)), [DISPLAY.get(f, f) for f in summary_tables])
    ax.set_xlim(-0.6, len(summary_tables) - 0.2)
    ax.set_ylabel("Cross-validated AUROC (mean of 5 folds)")
    ax.grid(axis="x", visible=False)
    ax.set_title(title, pad=22)
    ex._subtitle(ax, "One dot per setting; diamond = chosen setting (training part only)")
    return ex._save(fig, path)


def plot_roc_pr(curves: list[tuple[str, np.ndarray, dict[str, Any]]], y: np.ndarray, title: str, path: Path) -> Path:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    ex.apply_style()
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.8))
    for i, (label, prob, metrics) in enumerate(curves):
        color, style = (ex.MUTED, ":") if label.startswith("Version 0.3") else (ex.SERIES[i], "-")
        fpr, tpr, _ = roc_curve(y, prob)
        ax_roc.plot(fpr, tpr, style, color=color, lw=1.8, label=f"{label} ({metrics['roc_auc']:.3f})")
        precision, recall, _ = precision_recall_curve(y, prob)
        ax_pr.plot(recall, precision, style, color=color, lw=1.8, drawstyle="steps-post",
                   label=f"{label} ({metrics['pr_auc']:.3f})")
    ax_roc.plot([0, 1], [0, 1], ls="--", lw=1, color=ex.AXIS, label="No skill (0.500)")
    ax_pr.axhline(float(y.mean()), ls="--", lw=1, color=ex.AXIS, label=f"No skill ({y.mean():.3f})")
    ax_roc.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="1 - specificity", ylabel="Sensitivity (resistant flagged)")
    ax_pr.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall (sensitivity)", ylabel="Precision")
    ax_roc.set_title("ROC curve", pad=8)
    ax_pr.set_title("Precision-recall curve", pad=8)
    ax_roc.legend(title="Model (AUROC)", loc="lower right", fontsize=8.5)
    ax_pr.legend(title="Model (PR-AUC)", loc="upper right", fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return ex._save(fig, path)


def plot_calibration(panels: list[tuple[str, np.ndarray, list[tuple[str, np.ndarray, dict[str, Any]]]]],
                     n_bins: int, path: Path) -> Path:
    """One reliability panel per (title, labels, [(series label, probabilities, metrics)])."""
    import matplotlib.pyplot as plt

    ex.apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.9))
    for ax, (title, y, series) in zip(np.atleast_1d(axes), panels, strict=True):
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=ex.AXIS, label="Perfect calibration")
        for i, (label, prob, metrics) in enumerate(series):
            t = reliability_table(y, prob, n_bins)
            ax.plot(t["mean_predicted"], t["observed_rate"], "-o", color=ex.SERIES[i], lw=1.8, ms=6,
                    markeredgecolor=ex.SURFACE, markeredgewidth=1.5,
                    label=f"{label} (Brier {metrics['brier']:.3f}, slope {metrics['calibration_slope']:.2f})")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted resistance probability",
               ylabel="Observed resistant share")
        ax.set_title(title, pad=8)
        ax.legend(loc="upper left", fontsize=8.5)
    fig.suptitle(f"Calibration ({n_bins} bins with equal numbers of samples)", x=0.01, ha="left", fontsize=13,
                 fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return ex._save(fig, path)


# ------------------------------------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser(description="Version 0.4: tuned and calibrated models.")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="score the planned test parts once (appended to the test log)")
    parser.add_argument("--families", nargs="+", default=None, help="development only: tune these families")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        set_seed(int(config["project"]["random_seed"]))
        specs = family_specs(config)
        if args.families:
            unknown = sorted(set(args.families) - {s.name for s in specs})
            if unknown:
                raise TuningError(f"Unknown families {unknown}")
            if args.evaluate_test:
                raise TuningError("--evaluate-test needs the full search plan (no --families).")
            specs = [s for s in specs if s.name in args.families]
        ctx = Context(config, args.dataset or config["dataset"]["name"], args.evaluate_test)
        main_name = ctx.tc["split"]
        section(f"Version 0.4 on {ctx.dataset_name} (rows fingerprint {ctx.summary['row_fingerprint']}); "
                f"test parts {'WILL' if args.evaluate_test else 'will NOT'} be scored; families: "
                f"{', '.join(s.name for s in specs)}")
        started = time.monotonic()
        with keep_awake():
            summary_rows, finalists = tune_experiment(ctx, main_name, specs)
            main_seed = [f for f in finalists if f.seed == ctx.seeds[0]]
            chosen = max(main_seed, key=lambda f: f.validation[ctx.tc["search_metric"]])
            section(f"Chosen on validation ({ctx.tc['search_metric']}): {DISPLAY[chosen.family]} "
                    f"({chosen.validation[ctx.tc['search_metric']]:.3f}); re-tuning it for the patient-overlap check")
            chosen_spec = next(s for s in specs if s.name == chosen.family)
            retune = []
            for name in ctx.tc["retune_experiments"]:
                rows, found = tune_experiment(ctx, name, [chosen_spec])
                summary_rows += rows
                retune += found
        seconds = round(time.monotonic() - started, 1)

        # ---------------------------------------------------------------- reports: search and validation
        search_summary = pd.DataFrame(summary_rows)
        search_summary.to_csv(ctx.report_dir / "search_summary.csv", index=False, lineterminator="\n")
        section("Search results (cross-validation inside the training part)")
        show(search_summary)
        everything = finalists + retune
        va_rows = [f.row("validation") for f in everything]
        va_rows += [f.row("validation", calibrated=False) for f in everything if f.validation_uncalibrated]
        validation = pd.DataFrame(va_rows)
        validation.to_csv(ctx.report_dir / "validation_metrics.csv", index=False, lineterminator="\n")
        section("Validation (cut-offs chosen here; seed 42; calibrated=False rows show the same model before "
                "calibration)")
        show(validation.loc[validation["seed"] == ctx.seeds[0], [c for c in SHOWN if c in validation.columns]])
        for f in everything:
            if f.converged is False:
                log.warning("%s / %s (seed %d) stopped at its iteration limit", f.experiment, f.family, f.seed)

        # ---------------------------------------------------------------- save the chosen model
        split_rows = ctx.splits[chosen.split].train
        bundle = {
            "format": BUNDLE_FORMAT,
            "model_version": f"v{config['project']['version']}-{chosen.name}-{chosen.experiment}-seed{chosen.seed}",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": git_commit(),
            "pipeline": chosen.model, "model": chosen.name, "model_kind": chosen.family,
            "params": chosen.setting, "seed": chosen.seed,
            "calibration": {"method": ctx.tc["calibration"],
                            "fitted_on": f"out-of-fold predictions of {ctx.tc['cv_folds']} patient-grouped folds of "
                                         "the training part"},
            "species": ctx.summary["species"], "antibiotic": ctx.summary["antibiotic"],
            "label_map": {"0": ctx.summary["label_definition"]["0"], "1": ctx.summary["label_definition"]["1"]},
            "threshold": chosen.threshold, "threshold_rule": ctx.ev["threshold"],
            "preprocessing": ctx.summary["preprocessing"], "feature_fingerprint": ctx.summary["feature_fingerprint"],
            "n_features": int(ctx.summary["n_features"]),
            "dataset": {"name": ctx.dataset_name, "row_fingerprint": ctx.summary["row_fingerprint"],
                        "x_sha256": ctx.summary["x_sha256"], "sites": ctx.summary["sites"],
                        "intermediate_as": ctx.summary["intermediate_as"]},
            "split": {"name": chosen.split, "experiment": chosen.experiment, "train_size": chosen.train_size,
                      "dataset_fingerprint": ctx.splits[chosen.split].dataset_fingerprint,
                      "train_sites": sorted(ctx.meta["site"].iloc[split_rows].unique().tolist())},
            "selection": {"metric": ctx.tc["search_metric"], "part": "validation",
                          "candidates": {f.name: f.validation[ctx.tc["search_metric"]] for f in main_seed}},
            "metrics": {"validation": chosen.validation, "validation_uncalibrated": chosen.validation_uncalibrated,
                        "test": chosen.test},
            "versions": ctx.versions,
        }
        model_path = save_bundle(bundle, ctx.model_dir / f"best_{main_name}.joblib")
        (ctx.report_dir / "best_model_card.json").write_text(json.dumps(card(bundle), indent=2, default=str),
                                                             encoding="utf-8")
        print(f"\nSaved {model_path.relative_to(project_path('.'))} ({model_path.stat().st_size / 1e6:.2f} MB)")

        # ---------------------------------------------------------------- test parts
        if args.evaluate_test:
            b = ctx.ev["bootstrap"]
            test_rows = pd.DataFrame([f.row("test") for f in everything])
            _, _, _, main_te = ctx.experiment(main_name)
            y_te = ctx.y[main_te]

            v03_path = project_path(ctx.bl["model_dir"]) / ctx.dataset_name / f"best_{main_name}.joblib"
            v03 = load_bundle(v03_path)
            v03_prob = predict_features(v03, load_rows(ctx.X, main_te))[0]
            v03_metrics = classification_metrics(y_te, v03_prob, float(v03["threshold"]))
            logged = pd.read_csv(project_path(ctx.bl["report_dir"]) / ctx.dataset_name / "test_metrics.csv")
            logged = logged[(logged["experiment"] == main_name) & (logged["model"] == v03["model"])
                            & (logged["seed"] == v03["seed"])]
            # The logged value was computed with all CPU threads; a forest adds its trees in thread order, so
            # the last bits of tied probabilities can differ. 1e-4 is far below any meaningful AUROC change.
            v03_gap = abs(float(logged["roc_auc"].iloc[0]) - v03_metrics["roc_auc"]) if len(logged) == 1 else None
            if v03_gap is None or v03_gap > 1e-4:
                raise EvaluationError("The saved Version 0.3 model does not reproduce its logged test AUROC.")
            print(f"Version 0.3 model re-scored: AUROC {v03_metrics['roc_auc']:.6f} "
                  f"(logged {float(logged['roc_auc'].iloc[0]):.6f}, difference {v03_gap:.1e})")
            v03_row = {"experiment": main_name, "split": main_name, "model": f"v0.3_{v03['model']}",
                       "seed": v03["seed"], "calibrated": False, "setting": "Version 0.3 fixed settings",
                       "train_size": v03["split"]["train_size"], "reproduces_logged_auroc_within": v03_gap,
                       **v03_metrics}
            test_rows = pd.concat([test_rows, pd.DataFrame([v03_row])], ignore_index=True)
            test_rows.to_csv(ctx.report_dir / "test_metrics.csv", index=False, lineterminator="\n")
            section("Test results (each model scored once; seed 42 shown)")
            show(test_rows.loc[test_rows["seed"] == ctx.seeds[0], [c for c in SHOWN if c in test_rows.columns]])

            intervals, samples = {}, {}
            probs = {f.name: f.test_prob for f in main_seed}
            probs[v03_row["model"]] = v03_prob
            thresholds = {f.name: f.threshold for f in main_seed}
            thresholds[v03_row["model"]] = float(v03["threshold"])
            intervals[main_name], samples[main_name] = summarize_bootstrap(
                y_te, ctx.groups[main_te], probs, thresholds, resamples=int(b["resamples"]), level=float(b["level"]),
                seed=int(b["seed"]), reference=chosen.name)
            for name in ctx.tc["retune_experiments"]:
                _, _, _, te = ctx.experiment(name)
                f = next(x for x in retune if x.experiment == name and x.seed == ctx.seeds[0])
                intervals[name], samples[name] = summarize_bootstrap(
                    ctx.y[te], ctx.groups[te], {f.name: f.test_prob}, {f.name: f.threshold},
                    resamples=int(b["resamples"]), level=float(b["level"]), seed=int(b["seed"]))
            (ctx.report_dir / "test_intervals.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
            ci_rows = []
            for name, s in intervals.items():
                for model, m in s["intervals"].items():
                    row = {"experiment": name, "model": model}
                    for metric in ("roc_auc", "pr_auc", "sensitivity", "specificity"):
                        row[metric] = f"{m[metric]['estimate']:.3f} [{m[metric]['low']:.3f}, {m[metric]['high']:.3f}]"
                    diff = s["differences_to_reference"].get(model, {}).get("roc_auc")
                    row[f"AUROC {chosen.name} minus this"] = (
                        "-" if diff is None else f"{diff['estimate']:+.3f} [{diff['low']:+.3f}, {diff['high']:+.3f}]")
                    ci_rows.append(row)
            ci_table = pd.DataFrame(ci_rows)
            ci_table.to_csv(ctx.report_dir / "test_intervals.csv", index=False, lineterminator="\n")
            section(f"95 % intervals (patient-level bootstrap, {b['resamples']} resamples)")
            show(ci_table)

            stochastic = {f"tuned_{s.name}" for s in specs if s.stochastic}
            seed_rows = []
            for (name, model), g in test_rows[test_rows["model"].isin(stochastic)].groupby(["experiment", "model"]):
                row = {"experiment": name, "model": model, "seeds": len(g)}
                for metric in ("roc_auc", "pr_auc", "sensitivity", "specificity"):
                    row.update({f"{metric}_mean": g[metric].mean(), f"{metric}_min": g[metric].min(),
                                f"{metric}_max": g[metric].max()})
                seed_rows.append(row)
            pd.DataFrame(seed_rows).to_csv(ctx.report_dir / "seed_variation.csv", index=False, lineterminator="\n")
            section("Seed variation on test")
            show(pd.DataFrame(seed_rows))

            names = list(ctx.tc["retune_experiments"])
            if len(names) == 2:
                w_name = next(n for n in names if n in ctx.splits)
                m_name = next(n for n in names if n != w_name)
                level = float(b["level"])
                overlap = {"model": chosen.name}
                for metric in ("roc_auc", "pr_auc"):
                    m_est = intervals[m_name]["intervals"][chosen.name][metric]["estimate"]
                    w_est = intervals[w_name]["intervals"][chosen.name][metric]["estimate"]
                    low, high = unpaired_difference(samples[m_name][chosen.name][metric],
                                                    samples[w_name][chosen.name][metric], level)
                    overlap.update({f"{metric}_{m_name}": m_est, f"{metric}_{w_name}": w_est,
                                    f"{metric}_difference": m_est - w_est, f"{metric}_difference_low": low,
                                    f"{metric}_difference_high": high})
                pd.DataFrame([overlap]).to_csv(ctx.report_dir / "patient_overlap_check.csv", index=False,
                                               lineterminator="\n")
                section(f"Patient-overlap check ({m_name} minus {w_name}, both re-tuned)")
                show(pd.DataFrame([overlap]))

            commit = git_commit()
            logged_at = time.strftime("%Y-%m-%d %H:%M:%S")
            base = {"logged_at": logged_at, "git_commit": commit, "dataset": ctx.dataset_name,
                    "dataset_fingerprint": ctx.summary["row_fingerprint"], "x_sha256": ctx.summary["x_sha256"]}
            log_rows = [{**base, "stage": STAGE, **f.row("test")} for f in everything]
            log_rows.append({**base, "stage": "v0.4-reference", **v03_row})
            append_test_log(project_path(ctx.ev["test_log"]), log_rows)
            print(f"\n{len(log_rows)} test evaluations appended to {ctx.ev['test_log']}")

            name_of = {f.family: DISPLAY.get(f.family, f.family) for f in main_seed}
            curves = [(name_of[f.family], f.test_prob, f.test) for f in main_seed]
            curves.append((f"Version 0.3 {DISPLAY.get(v03['model'], v03['model']).lower()}", v03_prob, v03_metrics))
            val_rows = ctx.splits[main_name].validation
            before_after = []
            inner = uncalibrated(chosen.model)
            if hasattr(inner, "predict_proba"):
                before_after.append(("Before calibration", inner.predict_proba(load_rows(ctx.X, val_rows))[:, 1],
                                     chosen.validation_uncalibrated))
            before_after.append(("After sigmoid calibration", chosen.validation_prob, chosen.validation))
            panels = [(f"{name_of[chosen.family]}: validation part", ctx.y[val_rows], before_after),
                      ("Calibrated models: test part", y_te,
                       [(name_of[f.family], f.test_prob, f.test) for f in main_seed])]
            plots = [
                plot_search({s.name: pd.read_csv(ctx.report_dir / f"search_{main_name}_{s.name}.csv") for s in specs},
                            f"Settings searched on the {main_name} training part", ctx.plot_dir /
                            f"{ctx.dataset_name}_search_overview.png"),
                plot_roc_pr(curves, y_te, f"Tuned and calibrated models, {main_name} split, test part "
                                          f"(n={len(main_te):,}, {int(y_te.sum())} resistant)",
                            ctx.plot_dir / f"{ctx.dataset_name}_{main_name}_roc_pr.png"),
                plot_calibration(panels, int(ctx.ev["calibration_bins"]),
                                 ctx.plot_dir / f"{ctx.dataset_name}_{main_name}_calibration.png"),
                plot_confusion(chosen.threshold, chosen.test,
                               f"{name_of[chosen.family]} (tuned, calibrated), {main_name} split (test)",
                               ctx.plot_dir / f"{ctx.dataset_name}_{main_name}_confusion_{chosen.name}.png"),
            ]
            for p in plots:
                print(f"saved {p.relative_to(project_path('.'))}")

        # ---------------------------------------------------------------- timing with the saved file
        loaded = load_bundle(model_path)
        rng = np.random.default_rng(ctx.seed)
        val_rows = ctx.splits[main_name].validation
        timed = rng.choice(val_rows, size=min(int(ctx.ev["timing_samples"]), val_rows.size), replace=False)
        root = driams_root(config)
        predict_spectrum_file(loaded, resolve_relpath(root, ctx.meta["spectrum_relpath"].iloc[timed[0]]))
        timings = pd.DataFrame([predict_spectrum_file(loaded, resolve_relpath(root, rel)).to_dict()
                                for rel in ctx.meta["spectrum_relpath"].iloc[timed]])
        agrees = bool(np.allclose(timings["resistance_probability"],
                                  np.round(predict_features(loaded, load_rows(ctx.X, timed))[0], 4), atol=1e-4))
        batch_bundle = load_bundle(model_path, n_jobs=-1)          # all CPU threads for a batch
        batch = load_rows(ctx.X, val_rows)
        predict_features(batch_bundle, batch[:10])
        t0 = time.perf_counter()
        predict_features(batch_bundle, batch)
        timing = {"samples": len(timings), "model": loaded["model_version"], "matches_stored_features": agrees,
                  "single_spectrum_threads": 1, "batch_threads": "all", "batch_size": len(batch),
                  "batch_inference_ms_per_sample": round((time.perf_counter() - t0) * 1000 / len(batch), 4)}
        for col in ("preprocessing_ms", "inference_ms", "total_ms"):
            timing[col] = {"median": round(float(timings[col].median()), 2),
                           "p95": round(float(timings[col].quantile(0.95)), 2),
                           "max": round(float(timings[col].max()), 2)}
        (ctx.report_dir / "inference_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        section("Prediction time with the saved model")
        print(json.dumps(timing, indent=2))

        run_config = {"stage": STAGE, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": git_commit(),
                      "code_fingerprint": ctx.code, "dataset": ctx.dataset_name,
                      "rows_fingerprint": ctx.summary["row_fingerprint"], "x_sha256": ctx.summary["x_sha256"],
                      "test_parts_scored": args.evaluate_test, "families": [s.name for s in specs],
                      "chosen_family": chosen.family, "seconds": seconds, "tuning": ctx.tc,
                      "evaluation": ctx.ev, "versions": ctx.versions}
        (ctx.report_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str),
                                                        encoding="utf-8")
        print(f"\nReports: {ctx.report_dir}\nThis is a research prototype; predictions are not clinical results.")
        return 0
    except (DataError, ConfigError, SplitError, TrainingError, TuningError, EvaluationError, ModelError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
