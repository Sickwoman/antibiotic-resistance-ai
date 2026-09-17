"""Version 0.4 - tuned and calibrated models, following docs/v0.4_search_plan.md.

Run from the project root with the virtual environment active (keep the laptop plugged in, lid open):

    python scripts/tune_models.py                   # searches, calibration, validation; test parts untouched
    python scripts/tune_models.py --evaluate-test   # then: score the planned test parts once (logged)
    python scripts/tune_models.py --families logistic_regression   # development run on some families only

Searches and fitted models are cached in models/v0.4/cache, keyed by the data rows, the settings, the
tuning code and the library versions. The --evaluate-test run must reuse those cached models (it stops
instead of refitting unless --allow-refit is given) and must reproduce the development run's validation
results before any test row is scored.

Order of the --evaluate-test run: checks -> fitting/validation (cached) -> comparison with the development
run -> test scoring -> test log -> reports. A failure after scoring cannot skip the log.

Writes:
- results/metrics/v0.4/<dataset>/* and results/plots/v0.4/* (committed; no identifiers)
- results/experiments/test_evaluations.csv (append-only)
- models/v0.4/<dataset>/best_random.joblib, its .json card and test_probabilities.npz (git-ignored)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import exploration as ex  # noqa: E402  (plot style only)
from src.data_loader import DataError  # noqa: E402
from src.dataset import load_dataset, resolve_relpath  # noqa: E402
from src.evaluate import (  # noqa: E402
    TEST_LOG_COLUMNS,
    EvaluationError,
    append_test_log,
    choose_threshold,
    classification_metrics,
    interval,
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
REFERENCE_STAGE = "v0.4-reference"
DISPLAY = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
           "lightgbm": "LightGBM", "svm_rbf": "SVM (RBF)"}
SHOWN = ["experiment", "model", "seed", "calibrated", "train_size", "fit_seconds", "threshold", "roc_auc", "pr_auc",
         "sensitivity", "specificity", "precision", "f1", "accuracy", "brier", "calibration_slope"]
METRICS_WITH_CI = ("roc_auc", "pr_auc", "sensitivity", "specificity")


def section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def show(df: pd.DataFrame) -> None:
    formatters = {"threshold": "{:.3g}".format} if "threshold" in df.columns else None
    with pd.option_context("display.max_columns", 40, "display.width", 260, "display.max_colwidth", 90,
                           "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False, formatters=formatters))


def versions() -> dict[str, str]:
    """Library versions that are part of every cache key (do not add entries: that would change the keys)."""
    import joblib
    import lightgbm
    import sklearn

    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "joblib": joblib.__version__}


def rows_hash(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()[:16]


def readable(setting: dict[str, Any]) -> str:
    return json.dumps({k: setting[k] for k in sorted(setting)}, default=str)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


@dataclass
class Finalist:
    experiment: str
    split: str
    family: str
    seed: int
    setting: dict[str, Any]
    model: Any
    train_size: int
    fit_seconds: float            # 5 cross-fitted models for calibration + the final fit
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

    def __init__(self, config: dict[str, Any], dataset_name: str, evaluate_test: bool, require_cache: bool):
        self.config = config
        self.tc, self.ev, self.bl = config["tuning"], config["evaluation"], config["baselines"]
        self.dataset_name = dataset_name
        self.evaluate_test = evaluate_test
        self.require_cache = require_cache
        self.commit = git_commit()                   # read once: the code that runs is the code at start-up
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
        self.report_dir.mkdir(parents=True, exist_ok=True)

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

    def check_cache_info(self) -> None:
        """The cache keys cover the data rows but not the feature values: a side file records them."""
        import scipy

        info = {"x_sha256": self.summary["x_sha256"], "feature_fingerprint": self.summary["feature_fingerprint"],
                "scipy": scipy.__version__}
        path = self.cache_dir / "cache_info.json"
        if path.is_file():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored != info:
                raise TuningError(f"The cache in {self.cache_dir} was made with other features or scipy "
                                  f"({stored} vs {info}); delete that folder to recompute.")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    def get(self, path: Path, key: str, compute):
        """Cached result; in --evaluate-test mode a missing cache entry stops the run (unless --allow-refit)."""
        if self.require_cache:
            def refuse():
                raise TuningError(f"{path.relative_to(self.cache_dir)} is not cached for these settings/code. "
                                  "The test run must reuse the development run's models; run without "
                                  "--evaluate-test first (or pass --allow-refit on purpose).")
            return cached(path, key, refuse, log)
        return cached(path, key, compute, log)


def write_run_config(ctx: Context, specs: list[FamilySpec], status: str, **extra: Any) -> None:
    record = {"stage": STAGE, "status": status, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
              "git_commit": ctx.commit, "code_fingerprint": ctx.code, "dataset": ctx.dataset_name,
              "rows_fingerprint": ctx.summary["row_fingerprint"], "x_sha256": ctx.summary["x_sha256"],
              "feature_fingerprint": ctx.summary["feature_fingerprint"], "test_parts_scored": ctx.evaluate_test,
              "families": [s.name for s in specs], "versions": ctx.versions, **extra, "config": ctx.config}
    (ctx.report_dir / "run_config.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def tune_experiment(ctx: Context, name: str, specs: list[FamilySpec]) -> tuple[list[dict[str, Any]], list[Finalist]]:
    """Search, calibrate and validate on one experiment. Test rows are never loaded here."""
    split, tr, va, _ = ctx.experiment(name)
    X_tr, y_tr = load_rows(ctx.X, tr), ctx.y[tr]
    X_va, y_va = load_rows(ctx.X, va), ctx.y[va]
    folds = grouped_folds(y_tr, ctx.groups[tr], int(ctx.tc["cv_folds"]), ctx.cv_seed)
    base_key = {"dataset": ctx.summary["row_fingerprint"], "experiment": name, "train": rows_hash(tr),
                "folds": int(ctx.tc["cv_folds"]), "cv_seed": ctx.cv_seed, "metric": ctx.tc["search_metric"],
                "code": ctx.code, "versions": ctx.versions}
    summary_rows, finalists = [], []
    for spec in specs:
        key = cache_key(**base_key, family=spec.fingerprint())
        log.info("%s / %s: searching (or loading the cached search)", name, spec.name)
        search = ctx.get(ctx.cache_dir / name / spec.name / "search.joblib", key,
                         lambda spec=spec: run_search(spec, X_tr, y_tr, folds, ctx.cv_seed, ctx.tc["search_metric"]))
        write_csv(search.table, ctx.report_dir / f"search_{name}_{spec.name}.csv")
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

            model, fit_seconds = ctx.get(ctx.cache_dir / name / spec.name / f"fit_seed{seed}.joblib", fit_key, fit)
            set_threads(model, 1)            # single-thread prediction: bit-identical results in every run
            val_prob = model.predict_proba(X_va)[:, 1]
            threshold = choose_threshold(y_va, val_prob, ctx.ev["threshold"])
            inner = uncalibrated(model)
            validation_uncal = None
            if hasattr(inner, "predict_proba"):      # SVC without probability=True has none
                raw = inner.predict_proba(X_va)[:, 1]
                validation_uncal = classification_metrics(y_va, raw, choose_threshold(y_va, raw, ctx.ev["threshold"]))
            f = Finalist(name, split, spec.name, seed, search.best_setting, model, int(len(tr)), fit_seconds,
                         threshold, classification_metrics(y_va, val_prob, threshold), validation_uncal, val_prob,
                         converged(model))
            log.info("%s / %s (seed %d): fitted in %.0f s, validation AUROC %.3f", name, spec.name, seed,
                     fit_seconds, f.validation["roc_auc"])
            finalists.append(f)
    return summary_rows, finalists


def preflight(ctx: Context, allow_rescore: bool) -> dict[str, Any]:
    """Everything the test stage needs, checked before any test row is loaded."""
    names = [ctx.tc["split"], *ctx.tc["retune_experiments"]]
    locked = set(ctx.ev["locked_test_splits"])
    for name in names:
        split = ctx.experiment(name)[0]
        if split in locked:
            raise SplitError(f"Experiment {name!r} uses the {split} test part, which is locked until Version 0.7.")
    v03_path = project_path(ctx.bl["model_dir"]) / ctx.dataset_name / f"best_{ctx.tc['split']}.joblib"
    v03 = load_bundle(v03_path)
    if v03["dataset"]["row_fingerprint"] != ctx.summary["row_fingerprint"]:
        raise ModelError(f"{v03_path} was trained on another build of {ctx.dataset_name}.")
    v03_report = project_path(ctx.bl["report_dir"]) / ctx.dataset_name
    for name in ("test_metrics.csv", "validation_metrics.csv"):
        if not (v03_report / name).is_file():
            raise EvaluationError(f"Version 0.3 report {v03_report / name} is missing.")
    logged = pd.read_csv(v03_report / "test_metrics.csv")
    logged = logged[(logged["experiment"] == ctx.tc["split"]) & (logged["model"] == v03["model"])
                    & (logged["seed"] == v03["seed"])]
    if len(logged) != 1:
        raise EvaluationError("The Version 0.3 test report must contain exactly one row for the saved model.")
    log_path = project_path(ctx.ev["test_log"])
    if log_path.is_file() and log_path.stat().st_size:
        existing = pd.read_csv(log_path)
        if existing.columns.tolist() != TEST_LOG_COLUMNS:
            raise EvaluationError(f"{log_path} has unexpected columns.")
        done = existing[(existing["stage"] == STAGE) & (existing["dataset"] == ctx.dataset_name)]
        if len(done) and not allow_rescore:
            raise EvaluationError(f"{len(done)} Version 0.4 test evaluations of {ctx.dataset_name} are already "
                                  "logged. Scoring again is a second look at the test data; pass --allow-rescore "
                                  "only on purpose (the new rows are logged as an additional run).")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8"):
            pass
    except OSError as exc:
        raise EvaluationError(f"The test log {log_path} cannot be written ({exc}); close programs using it.") from exc
    return {"bundle": v03, "logged_roc_auc": float(logged["roc_auc"].iloc[0]), "report_dir": v03_report}


def compare_with_previous(previous: pd.DataFrame | None, current: pd.DataFrame) -> str:
    """The test run must reproduce the development run's validation results exactly."""
    if previous is None:
        return "no earlier validation results to compare with"
    keys = ["experiment", "model", "seed", "calibrated"]
    merged = current.merge(previous, on=keys, how="left", suffixes=("", "_before"), indicator=True)
    if (merged["_merge"] != "both").any():
        missing = merged.loc[merged["_merge"] != "both", keys].to_dict("records")
        raise TuningError(f"The development run has no validation results for {missing}; run it again first.")
    for col in ("roc_auc", "pr_auc", "threshold", "brier", "sensitivity", "specificity"):
        gap = np.nanmax(np.abs(merged[col].to_numpy(float) - merged[f"{col}_before"].to_numpy(float)))
        if gap > 1e-12:
            raise TuningError(f"Validation {col} differs from the development run by {gap:.3g}; not scoring test.")
    return f"{len(merged)} validation rows identical to the development run"


def reference_rows(ctx: Context, part: str, experiments: list[str]) -> pd.DataFrame:
    """Prevalence-only rows of the Version 0.3 run (same training data; not scored again)."""
    path = project_path(ctx.bl["report_dir"]) / ctx.dataset_name / f"{part}_metrics.csv"
    if not path.is_file():
        log.warning("No Version 0.3 %s report (%s): the prevalence-only reference rows are left out.", part, path)
        return pd.DataFrame()
    table = pd.read_csv(path)
    rows = table[(table["model"] == "prevalence") & table["experiment"].isin(experiments)].copy()
    rows["model"] = "v0.3_prevalence"
    rows["calibrated"] = False
    rows["setting"] = "always the training resistance rate (from the Version 0.3 run)"
    return rows


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


# ------------------------------------------------------------------------------------------------ test stage

def score_and_log(ctx: Context, everything: list[Finalist], ref: dict[str, Any]) -> dict[str, Any]:
    """Score every final model once on its test part, then append the log before anything else is written."""
    main_name = ctx.tc["split"]
    for f in everything:
        _, _, _, te = ctx.experiment(f.experiment)
        X_te = load_rows(ctx.X, te)
        started = time.perf_counter()
        f.test_prob = f.model.predict_proba(X_te)[:, 1]
        f.predict_ms_per_sample = (time.perf_counter() - started) * 1000 / len(te)
        f.test = classification_metrics(ctx.y[te], f.test_prob, f.threshold)
    _, _, _, main_te = ctx.experiment(main_name)
    v03 = ref["bundle"]
    v03_prob = predict_features(v03, load_rows(ctx.X, main_te))[0]
    v03_metrics = classification_metrics(ctx.y[main_te], v03_prob, float(v03["threshold"]))
    # The logged value was computed with all CPU threads; a forest adds its trees in thread order, so the last
    # bits of tied probabilities can differ. 1e-4 is far below any meaningful AUROC change.
    gap = abs(ref["logged_roc_auc"] - v03_metrics["roc_auc"])
    v03_row = {"experiment": main_name, "split": main_name, "model": f"v0.3_{v03['model']}", "seed": v03["seed"],
               "calibrated": False, "setting": "Version 0.3 fixed settings (saved model)",
               "train_size": v03["split"]["train_size"], "reproduces_logged_auroc_within": gap, **v03_metrics}

    base = {"logged_at": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": ctx.commit, "dataset": ctx.dataset_name,
            "dataset_fingerprint": ctx.summary["row_fingerprint"], "x_sha256": ctx.summary["x_sha256"]}
    rows = [{**base, "stage": STAGE, **f.row("test")} for f in everything]
    rows.append({**base, "stage": REFERENCE_STAGE, **v03_row})
    append_test_log(project_path(ctx.ev["test_log"]), rows)
    print(f"\n{len(rows)} test evaluations appended to {ctx.ev['test_log']}")
    if gap > 1e-4:
        raise EvaluationError(f"The saved Version 0.3 model gives test AUROC {v03_metrics['roc_auc']:.6f}, "
                              f"logged {ref['logged_roc_auc']:.6f} (logged above; check the model file).")
    print(f"Version 0.3 model re-scored: AUROC {v03_metrics['roc_auc']:.6f} (logged {ref['logged_roc_auc']:.6f}, "
          f"difference {gap:.1e})")

    arrays = {f"{f.experiment}__{f.name}__seed{f.seed}": f.test_prob for f in everything}
    arrays[f"{main_name}__{v03_row['model']}"] = v03_prob
    for name in {f.experiment for f in everything}:
        arrays[f"{name}__rows"] = ctx.experiment(name)[3]
    ctx.model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ctx.model_dir / "test_probabilities.npz", **arrays)
    return {"row": v03_row, "prob": v03_prob, "metrics": v03_metrics}


def differences(samples: dict[str, dict[str, np.ndarray]], points: dict[str, dict[str, dict[str, float]]],
                minus: str, level: float) -> dict[str, dict[str, dict[str, float]]]:
    """model minus `minus` for every other model, from the same (paired) resamples."""
    out = {}
    for model in samples:
        if model == minus:
            continue
        out[model] = {}
        for m in METRICS_WITH_CI:
            low, high = interval(samples[model][m] - samples[minus][m], level)
            out[model][m] = {"estimate": points[model][m]["estimate"] - points[minus][m]["estimate"],
                             "low": low, "high": high}
    return out


# ------------------------------------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser(description="Version 0.4: tuned and calibrated models.")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="score the planned test parts once (appended to the test log)")
    parser.add_argument("--allow-refit", action="store_true",
                        help="with --evaluate-test: fit models that are not in the cache (normally refused)")
    parser.add_argument("--allow-rescore", action="store_true",
                        help="with --evaluate-test: score again although Version 0.4 test rows are logged")
    parser.add_argument("--families", nargs="+", default=None, help="development only: tune these families")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        set_seed(int(config["project"]["random_seed"]))
        if config["tuning"]["search_metric"] != config["evaluation"]["primary_metric"]:
            raise TuningError("tuning.search_metric must equal evaluation.primary_metric.")
        specs = family_specs(config)
        if args.families:
            unknown = sorted(set(args.families) - {s.name for s in specs})
            if unknown:
                raise TuningError(f"Unknown families {unknown}")
            if args.evaluate_test:
                raise TuningError("--evaluate-test needs the full search plan (no --families).")
            specs = [s for s in specs if s.name in args.families]
        ctx = Context(config, args.dataset or config["dataset"]["name"], args.evaluate_test,
                      require_cache=args.evaluate_test and not args.allow_refit)
        ctx.check_cache_info()
        main_name = ctx.tc["split"]
        previous_path = ctx.report_dir / "validation_metrics.csv"
        previous = pd.read_csv(previous_path) if previous_path.is_file() else None
        ref = preflight(ctx, args.allow_rescore) if args.evaluate_test else None
        write_run_config(ctx, specs, "running")
        section(f"Version 0.4 on {ctx.dataset_name} (rows fingerprint {ctx.summary['row_fingerprint']}); "
                f"test parts {'WILL' if args.evaluate_test else 'will NOT'} be scored; families: "
                f"{', '.join(s.name for s in specs)}")
        started = time.monotonic()
        with keep_awake():
            summary_rows, finalists = tune_experiment(ctx, main_name, specs)
            main_seed = [f for f in finalists if f.seed == ctx.seeds[0]]
            chosen = max(main_seed, key=lambda f: f.validation[ctx.tc["search_metric"]])
            section(f"Chosen on validation ({ctx.tc['search_metric']}): {DISPLAY.get(chosen.family, chosen.family)} "
                    f"({chosen.validation[ctx.tc['search_metric']]:.3f}); re-tuning it for the patient-overlap check")
            chosen_spec = next(s for s in specs if s.name == chosen.family)
            retune = []
            for name in ctx.tc["retune_experiments"]:
                rows, found = tune_experiment(ctx, name, [chosen_spec])
                summary_rows += rows
                retune += found
        seconds = round(time.monotonic() - started, 1)
        everything = finalists + retune

        # ---------------------------------------------------------------- search and validation reports
        search_summary = pd.DataFrame(summary_rows)
        write_csv(search_summary, ctx.report_dir / "search_summary.csv")
        section("Search results (cross-validation inside the training part)")
        show(search_summary)
        va_rows = [f.row("validation") for f in everything]
        va_rows += [f.row("validation", calibrated=False) for f in everything if f.validation_uncalibrated]
        validation = pd.DataFrame(va_rows)
        if args.evaluate_test:
            print(compare_with_previous(previous, validation))
        experiments = [main_name, *ctx.tc["retune_experiments"]]
        validation = pd.concat([validation, reference_rows(ctx, "validation", experiments)], ignore_index=True)
        write_csv(validation, previous_path)
        section("Validation (cut-offs chosen here; seed 42; calibrated=False: same model before calibration; "
                "v0.3_prevalence: no-skill reference)")
        show(validation.loc[validation["seed"] == ctx.seeds[0], [c for c in SHOWN if c in validation.columns]])
        for f in everything:
            if f.converged is False:
                log.warning("%s / %s (seed %d) stopped at its iteration limit", f.experiment, f.family, f.seed)

        # ---------------------------------------------------------------- the chosen model (no test metrics yet)
        split_rows = ctx.splits[chosen.split].train
        bundle = {
            "format": BUNDLE_FORMAT,
            "model_version": f"v{config['project']['version']}-{chosen.name}-{chosen.experiment}-seed{chosen.seed}",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": ctx.commit, "code_fingerprint": ctx.code,
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
                        "test": None},
            "versions": ctx.versions,
        }
        model_path = ctx.model_dir / f"best_{main_name}.joblib"

        def save_chosen() -> None:
            save_bundle(bundle, model_path)
            (ctx.report_dir / "best_model_card.json").write_text(json.dumps(card(bundle), indent=2, default=str),
                                                                 encoding="utf-8")

        save_chosen()
        print(f"\nSaved {model_path.relative_to(project_path('.'))} ({model_path.stat().st_size / 1e6:.2f} MB)")

        # ---------------------------------------------------------------- test stage
        if args.evaluate_test:
            v03 = score_and_log(ctx, everything, ref)
            b = ctx.ev["bootstrap"]
            level = float(b["level"])
            _, _, _, main_te = ctx.experiment(main_name)
            y_te = ctx.y[main_te]
            test_rows = pd.concat([pd.DataFrame([f.row("test") for f in everything]), pd.DataFrame([v03["row"]]),
                                   reference_rows(ctx, "test", experiments)], ignore_index=True)
            write_csv(test_rows, ctx.report_dir / "test_metrics.csv")
            section("Test results (each model scored once; seed 42 shown)")
            show(test_rows.loc[test_rows["seed"] == ctx.seeds[0], [c for c in SHOWN if c in test_rows.columns]])

            intervals, samples = {}, {}
            probs = {f.name: f.test_prob for f in main_seed}
            probs[v03["row"]["model"]] = v03["prob"]
            thresholds = {f.name: f.threshold for f in main_seed}
            thresholds[v03["row"]["model"]] = float(ref["bundle"]["threshold"])
            intervals[main_name], samples[main_name] = summarize_bootstrap(
                y_te, ctx.groups[main_te], probs, thresholds, resamples=int(b["resamples"]), level=level,
                seed=int(b["seed"]), reference=chosen.name)
            intervals[main_name]["differences_to_version_0_3"] = differences(
                samples[main_name], intervals[main_name]["intervals"], v03["row"]["model"], level)
            for name in ctx.tc["retune_experiments"]:
                _, _, _, te = ctx.experiment(name)
                f = next(x for x in retune if x.experiment == name and x.seed == ctx.seeds[0])
                intervals[name], samples[name] = summarize_bootstrap(
                    ctx.y[te], ctx.groups[te], {f.name: f.test_prob}, {f.name: f.threshold},
                    resamples=int(b["resamples"]), level=level, seed=int(b["seed"]))
            (ctx.report_dir / "test_intervals.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
            ci_rows = []
            for name, s in intervals.items():
                for model, m in s["intervals"].items():
                    row = {"experiment": name, "model": model}
                    for metric in METRICS_WITH_CI:
                        row[metric] = f"{m[metric]['estimate']:.3f} [{m[metric]['low']:.3f}, {m[metric]['high']:.3f}]"
                    for label, table, sign in ((f"AUROC {chosen.name} minus this", s["differences_to_reference"], 1),
                                               ("AUROC this minus Version 0.3",
                                                s.get("differences_to_version_0_3", {}), 1)):
                        d = table.get(model, {}).get("roc_auc")
                        row[label] = "-" if d is None else (
                            f"{sign * d['estimate']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]")
                    ci_rows.append(row)
            ci_table = pd.DataFrame(ci_rows)
            write_csv(ci_table, ctx.report_dir / "test_intervals.csv")
            section(f"95 % intervals (patient-level bootstrap, {b['resamples']} resamples)")
            show(ci_table)

            stochastic = {f"tuned_{s.name}" for s in specs if s.stochastic}
            seed_rows = []
            tuned = test_rows[test_rows["model"].isin(stochastic)]
            for (name, model), g in tuned.groupby(["experiment", "model"]):
                row = {"experiment": name, "model": model, "seeds": len(g)}
                for metric in METRICS_WITH_CI:
                    row.update({f"{metric}_mean": g[metric].mean(), f"{metric}_min": g[metric].min(),
                                f"{metric}_max": g[metric].max()})
                seed_rows.append(row)
            write_csv(pd.DataFrame(seed_rows), ctx.report_dir / "seed_variation.csv")
            section("Seed variation on test")
            show(pd.DataFrame(seed_rows))

            names = list(ctx.tc["retune_experiments"])
            w_name = next((n for n in names if n in ctx.splits), None)
            m_name = next((n for n in names if n != w_name), None)
            if w_name and m_name:
                overlap = []
                for label, other, note in (
                        (f"{m_name} minus {w_name}", m_name, "same training size (the planned check)"),
                        (f"{main_name} minus {w_name}", main_name, "full training part (context; sizes differ)")):
                    row = {"comparison": label, "model": chosen.name, "note": note}
                    for metric in ("roc_auc", "pr_auc"):
                        a = intervals[other]["intervals"][chosen.name][metric]["estimate"]
                        w = intervals[w_name]["intervals"][chosen.name][metric]["estimate"]
                        low, high = unpaired_difference(samples[other][chosen.name][metric],
                                                        samples[w_name][chosen.name][metric], level)
                        row.update({f"{metric}_first": a, f"{metric}_{w_name}": w, f"{metric}_difference": a - w,
                                    f"{metric}_difference_low": low, f"{metric}_difference_high": high})
                    overlap.append(row)
                write_csv(pd.DataFrame(overlap), ctx.report_dir / "patient_overlap_check.csv")
                section("Patient-overlap check (chosen family re-tuned on each training part)")
                show(pd.DataFrame(overlap))

            name_of = {f.family: DISPLAY.get(f.family, f.family) for f in main_seed}
            curves = [(name_of[f.family], f.test_prob, f.test) for f in main_seed]
            curves.append((f"Version 0.3 {DISPLAY.get(ref['bundle']['model'], ref['bundle']['model']).lower()}",
                           v03["prob"], v03["metrics"]))
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
                            f"Settings searched on the {main_name} training part",
                            ctx.plot_dir / f"{ctx.dataset_name}_search_overview.png"),
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
            bundle["metrics"]["test"] = chosen.test
            save_chosen()

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

        write_run_config(ctx, specs, "finished", chosen_family=chosen.family, seconds=seconds)
        print(f"\nReports: {ctx.report_dir}\nThis is a research prototype; predictions are not clinical results.")
        return 0
    except (DataError, ConfigError, SplitError, TrainingError, TuningError, EvaluationError, ModelError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
