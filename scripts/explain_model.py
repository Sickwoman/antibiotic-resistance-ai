"""Version 0.6: explain the project's model in m/z terms and fit its confidence zones.

Everything here describes a model that already exists; nothing is trained except one deliberate control
model on shuffled labels, and no model, setting or cut-off may change because of what comes out. The
procedure was fixed in docs/v0.6_explainability_plan.md before the first explanation was computed.

Run from the project root with the virtual environment active:

    python scripts/explain_model.py                       # the Version 0.4 model, the project's model
    python scripts/explain_model.py --skip-cross-model    # without the Version 0.5 MLP comparison
    python scripts/explain_model.py --dataset <name>

**Test rows are never scored here.** Contributions and importances use the validation part; the training
part is used only for the class contrast and the shuffled-label control. Where a test-side number appears
it is derived from the probabilities the one-time test scoring already saved, after checking that they
cover exactly that test part's rows and reproduce both the logged AUROC and the logged Brier score (the
AUROC alone would not: it is unchanged by any monotone rescaling, which the zones are not). The run
asserts that the append-only test log is byte-identical when it finishes.

Writes (paths from config.yaml -> explain):
- results/metrics/v0.6/<dataset>/* and results/plots/v0.6/* (committed; no identifiers)
- models/v0.4/<dataset>/uncertainty.json (a copy next to the model it belongs to; git-ignored)
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
from src.dataset import load_dataset  # noqa: E402
from src.evaluate import EvaluationError, assert_split_allowed  # noqa: E402
from src.explain import (  # noqa: E402
    ExplainError,
    bin_span,
    class_contrast,
    contiguous_blocks,
    contributions,
    explain_one,
    global_importance,
    merge_regions,
    model_columns,
    permutation_importance,
    rank_agreement,
)
from src.model_plots import plot_importance_agreement, plot_regions, plot_uncertainty_zones  # noqa: E402
from src.predict import ModelError, load_bundle  # noqa: E402
from src.preprocessing import PreprocessingConfig  # noqa: E402
from src.splits import SplitError, assert_usable, check_split, load_splits  # noqa: E402
from src.train import load_rows  # noqa: E402
from src.tuning import TuningError, base_pipeline, family_specs, pipeline_params, set_threads  # noqa: E402
from src.uncertainty import (  # noqa: E402
    UncertaintyError,
    ZoneRule,
    Zones,
    bootstrap_zone_metrics,
    fit_zones,
    zone_metrics,
)
from src.utils import (  # noqa: E402
    ConfigError,
    file_hash,
    get_logger,
    git_commit,
    keep_awake,
    load_config,
    project_path,
    set_seed,
)

STAGE = "v0.6-explain"
EXACT = 1e-12          # a saved model re-scored on the same rows must reproduce its logged metric exactly
log = get_logger("explain")


def versions() -> dict[str, str]:
    import joblib
    import lightgbm
    import sklearn

    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "joblib": joblib.__version__}


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


def roc_auc(y: np.ndarray, prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(y).astype(np.int64), np.asarray(prob, dtype=np.float64)))


class Context:
    """The dataset, the saved model being explained, and where the reports go."""

    def __init__(self, config: dict[str, Any], dataset_name: str | None, skip_cross_model: bool):
        self.config = config
        self.xc = config["explain"]
        self.ev = config["evaluation"]
        self.dataset_name = dataset_name or config["dataset"]["name"]
        self.commit = git_commit()
        self.versions = versions()
        self.skip_cross_model = skip_cross_model

        data_dir = project_path(config["dataset"]["output_dir"]) / self.dataset_name
        self.X, self.meta, self.summary = load_dataset(data_dir, verify_x=True)
        self.splits = load_splits(data_dir / "splits", self.meta)
        if str(self.xc.get("rows")) != "validation":
            raise ConfigError(f"explain.rows is {self.xc.get('rows')!r}; the protocol (amendment 2) allows "
                              "explanations on the validation part only.")
        self.split_name = str(self.xc["split"])
        if self.split_name not in self.splits:
            raise SplitError(f"The dataset has no {self.split_name!r} split.")
        # Version 0.6 scores no test row at all, but a locked split must still be refused here, before
        # anything is computed: that is the first of the two refusals the protocol requires.
        assert_split_allowed(self.split_name, self.ev["locked_test_splits"])
        self.split = self.splits[self.split_name]
        check_split(self.meta, self.split)
        assert_usable(self.meta, self.split)

        self.model_root = project_path(self.xc["model_dir"]) / self.dataset_name
        self.bundle_path = self.model_root / f"best_{self.split_name}.joblib"
        self.bundle = load_bundle(self.bundle_path, n_jobs=1)
        self.pipeline = self.bundle["pipeline"]
        set_threads(self.pipeline, 1)                 # predictions on one thread, so a rerun reproduces them
        self._check_bundle_matches_dataset()

        self.report_dir = project_path(self.xc["report_dir"]) / self.dataset_name
        self.plot_dir = project_path(self.xc["plot_dir"])
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.previous_dir = self._reports_of_the_explained_model()
        self.test_log = project_path(self.ev["test_log"])
        self.test_log_hash = file_hash(self.test_log)

        self.pcfg = PreprocessingConfig.from_dict(self.bundle["preprocessing"])
        self.span = bin_span(self.pipeline)
        self.y = self.meta["label"].to_numpy().astype(np.int64)
        self.validation_rows = np.asarray(self.split.validation, dtype=np.int64)
        self.train_rows = np.asarray(self.split.train, dtype=np.int64)

    def _reports_of_the_explained_model(self) -> Path:
        """Where the model being explained wrote its own reports (its config section owns both paths).

        The `explain` section is skipped: it names the same `model_dir` as the section that produced the
        model, so matching it would point this at Version 0.6's own reports and silently lose both the
        per-seed verification and the whole test-side summary.
        """
        wanted = str(self.xc["model_dir"]).rstrip("/")
        for name, section in self.config.items():
            if name == "explain" or not isinstance(section, dict) or "report_dir" not in section:
                continue
            if str(section.get("model_dir", "")).rstrip("/") == wanted:
                return project_path(section["report_dir"]) / self.dataset_name
        raise ConfigError(f"No config section other than 'explain' has model_dir {wanted!r}, so the reports "
                          "of the model being explained cannot be found. explain.model_dir must name a "
                          "section that produced a model.")

    def _check_bundle_matches_dataset(self) -> None:
        """The saved model must belong to this dataset build, or the explanation describes other data."""
        dataset = self.bundle.get("dataset") or {}
        if dataset.get("row_fingerprint") != self.summary["row_fingerprint"]:
            raise ModelError(f"{self.bundle_path} was trained on dataset rows "
                             f"{dataset.get('row_fingerprint')!r}, but this build is "
                             f"{self.summary['row_fingerprint']!r}.")
        if self.bundle["feature_fingerprint"] != self.summary["feature_fingerprint"]:
            raise ModelError("The model's preprocessing settings differ from the dataset's.")
        if int(self.bundle["n_features"]) != int(self.X.shape[1]):
            raise ModelError(f"The model expects {self.bundle['n_features']} bins, the dataset has "
                             f"{self.X.shape[1]}.")

    def record(self, status: str, **extra: Any) -> dict[str, Any]:
        return {"stage": STAGE, "status": status, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": time.strftime("%Z%z"), "git_commit": self.commit, "dataset": self.dataset_name,
                "rows_fingerprint": self.summary["row_fingerprint"], "x_sha256": self.summary["x_sha256"],
                "feature_fingerprint": self.summary["feature_fingerprint"],
                "explained_model": self.bundle["model_version"], "split": self.split_name,
                "rows_explained": self.xc["rows"], "n_rows_explained": int(self.validation_rows.size),
                "test_parts_scored": False, "versions": self.versions, **extra, "config": self.config}

    def write_status(self, status: str, **extra: Any) -> None:
        (self.report_dir / "run_status.json").write_text(json.dumps(self.record(status, **extra), indent=2,
                                                                    default=str), encoding="utf-8")

    def write_config(self, **extra: Any) -> None:
        """The record of a finished run. The status file is finished too, so a leftover 'running' always
        means the run really did stop early."""
        record = self.record("finished", **extra)
        (self.report_dir / "run_config.json").write_text(json.dumps(record, indent=2, default=str),
                                                         encoding="utf-8")
        (self.report_dir / "run_status.json").write_text(json.dumps(record, indent=2, default=str),
                                                         encoding="utf-8")

    def assert_test_log_untouched(self) -> None:
        if file_hash(self.test_log) != self.test_log_hash:
            raise EvaluationError(f"{self.test_log} changed during a Version 0.6 run. Nothing here may score "
                                  "a test row; the run is not trustworthy and its reports should be discarded.")


# --- the model this describes ------------------------------------------------------------------------------------

def validation_probabilities(ctx: Context) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Score the validation rows and check the model reproduces the AUROC it was reported with."""
    X_val = load_rows(ctx.X, ctx.validation_rows)
    y_val = ctx.y[ctx.validation_rows]
    prob = ctx.pipeline.predict_proba(X_val)[:, 1]
    logged = float(((ctx.bundle.get("metrics") or {}).get("validation") or {}).get("roc_auc", float("nan")))
    achieved = roc_auc(y_val, prob)
    if np.isfinite(logged) and abs(achieved - logged) > EXACT:
        raise ModelError(f"The saved model scores {achieved:.12f} on validation but was reported as "
                         f"{logged:.12f}. It is not the model the reports describe; stopping.")
    log.info("validation rows %d (%d resistant); model reproduces its logged AUROC %.6f",
             len(y_val), int(y_val.sum()), achieved)
    return X_val, y_val, prob, achieved


# --- what the model used -----------------------------------------------------------------------------------------

def has_exact_contributions(ctx: Context) -> bool:
    """Whether this model can be explained exactly rather than approximately.

    Only gradient-boosted trees give exact per-prediction contributions here. The project's model is one,
    so the full analysis runs; a different saved model still gets the model-agnostic parts, and the run
    records which parts were possible instead of quietly substituting an approximation.
    """
    try:
        contributions(ctx.pipeline, load_rows(ctx.X, ctx.validation_rows[:2]))
    except ExplainError as exc:
        log.warning("no exact contributions for this model (%s); TreeSHAP, the region table, the seed "
                    "stability check and the shuffled-label control are left out of this run", exc)
        return False
    return True


def importance_and_regions(ctx: Context, X_val: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    contrib, _ = contributions(ctx.pipeline, X_val)
    importance = global_importance(contrib, ctx.span, ctx.pcfg, model_columns(ctx.pipeline, X_val))
    regions = merge_regions(importance, top_bins=int(ctx.xc["regions"]["top_bins"]),
                            merge_gap=int(ctx.xc["regions"]["merge_gap_bins"]),
                            keep=int(ctx.xc["regions"]["report_regions"]))
    log.info("contributions: %d columns, %d ever used; %d regions reported",
             contrib.shape[1], int((np.abs(contrib) > 0).any(axis=0).sum()), len(regions))
    return importance, regions, contrib


def use_all_threads(ctx: Context, model: Any) -> bool:
    """Let the permutation work use every CPU thread, but only after proving it changes nothing.

    Permuting 1,000 blocks five times is thousands of predictions, and single-threaded they take the best
    part of an hour. The saved model is documented as giving the same probabilities at any thread count;
    that claim is checked here on real rows rather than assumed, and the model is put back on one thread
    if it does not hold exactly.
    """
    sample = load_rows(ctx.X, ctx.validation_rows[:64])
    set_threads(model, 1)
    one = model.predict_proba(sample)[:, 1]
    set_threads(model, -1)
    if np.array_equal(one, model.predict_proba(sample)[:, 1]):
        return True
    set_threads(model, 1)
    log.info("predictions differ between thread counts; staying on one thread")
    return False


def permutation_tables(ctx: Context, X_val: np.ndarray, y_val: np.ndarray,
                       importance: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    settings = ctx.xc["permutation"]
    threaded = use_all_threads(ctx, ctx.pipeline)
    blocks = contiguous_blocks(int(ctx.X.shape[1]), int(settings["block_bins"]))
    log.info("block permutation importance: %d blocks x %d repeats", len(blocks), int(settings["repeats"]))
    per_block = permutation_importance(ctx.pipeline, X_val, y_val, blocks, repeats=int(settings["repeats"]),
                                       seed=int(settings["seed"]), pcfg=ctx.pcfg, progress=200)
    if importance is None:                     # the top bins to re-test one at a time come from TreeSHAP
        set_threads(ctx.pipeline, 1)
        return per_block, None
    top = importance.nlargest(int(settings["top_single_bins"]), "mean_abs")["column"].to_numpy()
    single = permutation_importance(ctx.pipeline, X_val, y_val, [(int(c), int(c) + 1) for c in top],
                                    repeats=int(settings["repeats"]), seed=int(settings["seed"]), pcfg=ctx.pcfg)
    if threaded:
        set_threads(ctx.pipeline, 1)           # back to one thread: everything else here is single-threaded
    return per_block, single


def block_shap(importance: pd.DataFrame, blocks: list[tuple[int, int]], span: int = 1) -> np.ndarray:
    """SHAP importance summed to the same blocks the permutation used, so the two can be compared.

    The blocks are ranges of raw input bins; the importances are per model column, which covers `span`
    raw bins. Mapping one to the other is what keeps the two measures aligned when the pipeline coarsens.
    """
    by_column = importance.sort_values("column")["mean_abs"].to_numpy()
    return np.array([by_column[start // span:-(-end // span)].sum() for start, end in blocks])


# --- is it stable? -------------------------------------------------------------------------------------------------

def cached_seed_models(ctx: Context) -> list[tuple[int, Any]]:
    """The per-seed fits of the same setting, from the Version 0.4 cache, each verified before it is used."""
    import joblib

    family = str(ctx.bundle["model"]).removeprefix("tuned_")
    cache = project_path(ctx.xc["model_dir"]) / "cache" / ctx.dataset_name / ctx.split_name / family
    logged = pd.read_csv(ctx.previous_dir / "validation_metrics.csv")
    logged = logged[(logged["model"] == ctx.bundle["model"]) & (logged["experiment"] == ctx.split_name)]
    if "calibrated" in logged.columns:
        logged = logged[logged["calibrated"]]
    X_val, y_val = load_rows(ctx.X, ctx.validation_rows), ctx.y[ctx.validation_rows]
    out: list[tuple[int, Any]] = []
    for seed in [int(s) for s in ctx.xc["seeds"]]:
        path = cache / f"fit_seed{seed}.joblib"
        row = logged[logged["seed"] == seed]
        if not path.is_file() or row.empty:
            log.warning("seed %d: no cached fit or no logged validation row; left out of the stability check", seed)
            continue
        stored = joblib.load(path)
        model = stored["value"][0] if isinstance(stored, dict) and "value" in stored else None
        if model is None:
            log.warning("seed %d: the cache file is not a fitted model; left out", seed)
            continue
        set_threads(model, 1)
        achieved = roc_auc(y_val, model.predict_proba(X_val)[:, 1])
        expected = float(row["roc_auc"].iloc[0])
        if abs(achieved - expected) > EXACT:
            log.warning("seed %d: cached fit scores %.12f but %.12f was logged; left out", seed, achieved, expected)
            continue
        out.append((seed, model))
    log.info("stability: %d of %d cached fits verified", len(out), len(ctx.xc["seeds"]))
    return out


def seed_stability(ctx: Context, X_val: np.ndarray, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank agreement of the bin importances between the per-seed fits of the same setting."""
    columns = model_columns(ctx.pipeline, X_val)
    per_seed: dict[int, np.ndarray] = {}
    rows = []
    for seed, model in cached_seed_models(ctx):
        contrib, _ = contributions(model, X_val)
        frame = global_importance(contrib, ctx.span, ctx.pcfg, columns)
        per_seed[seed] = frame.sort_values("column")["mean_abs"].to_numpy()
        regions = merge_regions(frame, top_bins=int(ctx.xc["regions"]["top_bins"]),
                                merge_gap=int(ctx.xc["regions"]["merge_gap_bins"]),
                                keep=int(ctx.xc["regions"]["report_regions"]))
        shared = len(set(regions["peak_mz"].round(0)) & set(reference["peak_mz"].round(0)))
        rows.append({"seed": seed, "n_regions": len(regions), "regions_shared_with_seed42": shared,
                     "strongest_region_mz_start": float(regions["mz_start"].iloc[0]) if len(regions) else np.nan,
                     "strongest_region_mz_end": float(regions["mz_end"].iloc[0]) if len(regions) else np.nan})
    pairs = []
    seeds = sorted(per_seed)
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            agreement = rank_agreement(per_seed[a], per_seed[b], top_k=int(ctx.xc["regions"]["top_bins"]))
            pairs.append({"seed_a": a, "seed_b": b, **agreement})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def null_control(ctx: Context, X_val: np.ndarray, regions: pd.DataFrame) -> dict[str, Any]:
    """Refit the same setting on shuffled training labels: how strong does a region look by chance?"""
    spec = next((s for s in family_specs(ctx.config, "tuning")
                 if s.name == str(ctx.bundle["model"]).removeprefix("tuned_")), None)
    if spec is None:
        raise TuningError(f"No family in the tuning section matches {ctx.bundle['model']!r}.")
    seed = int(ctx.xc["null_control"]["label_shuffle_seed"])
    setting = {k: v for k, v in (ctx.bundle.get("params") or {}).items()}
    pipeline = base_pipeline(spec, seed, n_jobs=-1)
    pipeline.set_params(**pipeline_params(spec, setting, seed))
    X_train = load_rows(ctx.X, ctx.train_rows)
    shuffled = np.random.default_rng(seed).permutation(ctx.y[ctx.train_rows])
    started = time.perf_counter()
    pipeline.fit(X_train, shuffled)
    set_threads(pipeline, 1)
    contrib, _ = contributions(pipeline, X_val)
    strength = np.abs(contrib).mean(axis=0)
    # Like for like: the control's strongest region is found by exactly the same merging procedure as the
    # real one, so a contiguous region is compared with a contiguous region rather than with the strongest
    # scattered bins (which would flatter the real model).
    null_importance = global_importance(contrib, ctx.span, ctx.pcfg, model_columns(pipeline, X_val))
    null_regions = merge_regions(null_importance, top_bins=int(ctx.xc["regions"]["top_bins"]),
                                 merge_gap=int(ctx.xc["regions"]["merge_gap_bins"]), keep=1)
    null_region = float(null_regions["total_abs"].iloc[0]) if len(null_regions) else float("nan")
    real = float(regions["total_abs"].iloc[0]) if len(regions) else float("nan")
    # The control has no calibration step, so only magnitudes on the margin are compared, never probabilities.
    return {"seed": seed, "fit_seconds": round(time.perf_counter() - started, 1),
            "shuffled_labels": True, "n_train": int(ctx.train_rows.size),
            "null_strongest_bin": float(strength.max()), "null_mean_bin": float(strength.mean()),
            "null_strongest_region": null_region,
            "null_region_columns": int(null_regions["n_columns"].iloc[0]) if len(null_regions) else 0,
            "real_strongest_region": real,
            "real_region_columns": int(regions["n_columns"].iloc[0]) if len(regions) else 0,
            "real_over_null": real / null_region if null_region and np.isfinite(null_region) else float("nan")}


def cross_model(ctx: Context, X_val: np.ndarray, y_val: np.ndarray,
                per_block: pd.DataFrame) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """The same permutation importance on the Version 0.5 MLP: do two model families agree?"""
    path = project_path(ctx.xc["cross_model"]) / ctx.dataset_name / f"best_{ctx.split_name}.joblib"
    if ctx.skip_cross_model or not path.is_file():
        log.info("cross-model check skipped (%s)", "asked for" if ctx.skip_cross_model else f"no {path}")
        return None, None
    other = load_bundle(path, n_jobs=1)
    if int(other["n_features"]) != int(ctx.X.shape[1]):
        log.warning("cross-model check skipped: %s expects %s bins", path.name, other["n_features"])
        return None, None
    settings = ctx.xc["permutation"]
    blocks = contiguous_blocks(int(ctx.X.shape[1]), int(settings["block_bins"]))
    log.info("cross-model permutation importance on %s", other["model_version"])
    use_all_threads(ctx, other["pipeline"])
    frame = permutation_importance(other["pipeline"], X_val, y_val, blocks, repeats=int(settings["repeats"]),
                                   seed=int(settings["seed"]), pcfg=ctx.pcfg, progress=200)
    order = per_block.sort_values("bin_start")["auroc_drop_mean"].to_numpy()
    agreement = rank_agreement(order, frame.sort_values("bin_start")["auroc_drop_mean"].to_numpy(), top_k=20)
    agreement["model"] = other["model_version"]
    agreement["baseline_roc_auc"] = float(frame["baseline_roc_auc"].iloc[0])
    return frame, agreement


# --- confidence zones ------------------------------------------------------------------------------------------

def fit_and_write_zones(ctx: Context, y_val: np.ndarray, prob_val: np.ndarray) -> tuple[Zones, dict[str, Any]]:
    """Fit the zones on validation and write them out *before* anything reads a stored test prediction."""
    rule = ZoneRule.from_config(ctx.xc["uncertainty"])
    zones, low, high, curve = fit_zones(y_val, prob_val, float(ctx.bundle["threshold"]), rule)
    write_csv(pd.DataFrame(curve), ctx.report_dir / "uncertainty_curve.csv")
    record = {**zones.to_dict(), "model_version": ctx.bundle["model_version"],
              "sides": {side.side: {"target": side.target, "edge": side.edge, "n_covered": side.n_covered,
                                    "coverage": side.coverage, "achieved": side.achieved,
                                    "exists": side.exists, "best_achieved": side.best_achieved,
                                    "best_edge": side.best_edge, "best_coverage": side.best_coverage}
                        for side in (low, high)}}
    for path in (ctx.report_dir / "uncertainty.json", ctx.model_root / "uncertainty.json"):
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    for side in (low, high):
        if side.exists:
            log.info("%s zone: edge %.4f, covers %.1f%% of validation, achieves %.3f (target %.2f)",
                     side.side, side.edge, 100 * side.coverage, side.achieved, side.target)
        else:
            log.info("%s zone: no cut reaches the target %.2f at %.0f%% coverage; the best it reaches is "
                     "%.3f (at %.1f%% coverage). Reported as 'no high-confidence zone'.", side.side,
                     side.target, 100 * rule.min_coverage, side.best_achieved, 100 * side.best_coverage)
    return zones, record


def stored_test_summary(ctx: Context, zones: Zones) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Zone behaviour on the test part, derived from the probabilities the one-time scoring already saved.

    Nothing is scored here: the stored array must first reproduce the AUROC that was logged for it, which
    is what makes it the same evaluation rather than a new one.
    """
    path = ctx.model_root / "test_probabilities.npz"
    metrics_path = ctx.previous_dir / "test_metrics.csv"
    if not path.is_file() or not metrics_path.is_file():
        log.warning("no stored test probabilities (%s); the test-side zone summary is left out", path.name)
        return None, None
    stored = np.load(path)
    model, seed = str(ctx.bundle["model"]), int(ctx.bundle["seed"])
    key, rows_key = f"{ctx.split_name}__{model}__seed{seed}", f"{ctx.split_name}__rows"
    if key not in stored or rows_key not in stored:
        log.warning("%s has no entry %r; the test-side zone summary is left out", path.name, key)
        return None, None
    rows = np.asarray(stored[rows_key], dtype=np.int64)
    prob = np.asarray(stored[key], dtype=np.float64)
    logged = pd.read_csv(metrics_path)
    logged = logged[(logged["model"] == model) & (logged["seed"] == seed) &
                    (logged["experiment"] == ctx.split_name)]
    if "calibrated" in logged.columns:
        logged = logged[logged["calibrated"]]
    if logged.empty:
        log.warning("no logged test row for %s seed %d; the test-side zone summary is left out", model, seed)
        return None, None
    # AUROC alone is not enough: it is invariant under any monotone transform of the probabilities, while
    # every zone number depends on their absolute values. The Brier score is not invariant, and the row
    # order has to be the test part's own, so both are checked as well.
    if not np.array_equal(rows, np.asarray(ctx.split.test, dtype=np.int64)):
        raise EvaluationError(f"The stored rows in {path.name} are not this dataset's {ctx.split_name} test "
                              "part (order included); refusing to derive anything from them.")
    achieved, expected = roc_auc(ctx.y[rows], prob), float(logged["roc_auc"].iloc[0])
    brier = float(np.mean((prob - ctx.y[rows]) ** 2))
    logged_brier = float(logged["brier"].iloc[0])
    for name, got, want in (("AUROC", achieved, expected), ("Brier score", brier, logged_brier)):
        if abs(got - want) > EXACT:
            raise EvaluationError(f"The stored test probabilities give {name} {got:.12f} but {want:.12f} was "
                                  f"logged. They are not the scored predictions; refusing to derive from them.")
    log.info("stored test predictions verified against the log (AUROC %.12f, Brier %.12f, same %d rows); "
             "deriving zone behaviour only", achieved, brier, rows.size)
    summary = {"reproduces_logged_auroc": achieved, "logged_auroc": expected,
               "reproduces_logged_brier": brier, "logged_brier": logged_brier, "n": int(rows.size),
               **zone_metrics(ctx.y[rows], prob, zones)}
    intervals = bootstrap_zone_metrics(ctx.y[rows], ctx.meta["group_id"].to_numpy()[rows], prob, zones,
                                       resamples=int(ctx.ev["bootstrap"]["resamples"]),
                                       level=float(ctx.ev["bootstrap"]["level"]),
                                       seed=int(ctx.ev["bootstrap"]["seed"]))
    return summary, intervals


def example_explanations(ctx: Context, X_val: np.ndarray, y_val: np.ndarray, prob: np.ndarray,
                         zones: Zones) -> pd.DataFrame:
    """One explained spectrum per zone, named by its position in the validation part and nothing else."""
    chosen = {"lowest probability": int(np.argmin(prob)), "closest to the cut-off":
              int(np.argmin(np.abs(prob - float(ctx.bundle["threshold"])))), "highest probability":
              int(np.argmax(prob))}
    rows = []
    for description, position in chosen.items():
        regions = explain_one(ctx.pipeline, X_val[position], ctx.pcfg, top_bins=40,
                              merge_gap=int(ctx.xc["regions"]["merge_gap_bins"]), keep=5)
        for region in regions.itertuples(index=False):
            rows.append({"example": description, "validation_position": position,
                         "resistance_probability": round(float(prob[position]), 6),
                         "true_label": int(y_val[position]), "output": zones.one(float(prob[position])),
                         "region_rank": int(region.rank), "mz_start": float(region.mz_start),
                         "mz_end": float(region.mz_end),
                         "contribution": round(float(region.total_signed), 6),
                         "pushed_towards": region.towards})
    return pd.DataFrame(rows)


# --- figures -------------------------------------------------------------------------------------------------------

def figures(ctx: Context, X_val: np.ndarray, y_val: np.ndarray, prob: np.ndarray,
            regions: pd.DataFrame | None, importance: pd.DataFrame | None, per_block: pd.DataFrame,
            curve: pd.DataFrame, zones: Zones) -> list[Path]:
    ex.apply_style()
    stem = f"{ctx.dataset_name}_{ctx.split_name}"
    mz = np.asarray(ctx.pcfg.bin_centers, dtype=np.float64)
    blocks = contiguous_blocks(int(ctx.X.shape[1]), int(ctx.xc["permutation"]["block_bins"]))
    ordered = per_block.sort_values("bin_start")
    out = [plot_uncertainty_zones(curve, zones, prob, y_val,
                                  "Version 0.6: confidence zones fitted on the validation part",
                                  ctx.plot_dir / f"{stem}_uncertainty_zones.png")]
    if regions is not None and importance is not None:
        out.insert(0, plot_regions(regions, X_val[y_val == 1].mean(axis=0), X_val[y_val == 0].mean(axis=0),
                                   mz, f"Version 0.6: influential m/z regions ({ctx.bundle['model_version']})",
                                   ctx.plot_dir / f"{stem}_regions.png"))
        out.append(plot_importance_agreement(
            block_shap(importance, blocks, ctx.span), ordered["auroc_drop_mean"].to_numpy(),
            mz[[start for start, _ in blocks]],
            "Version 0.6: contribution size against AUROC loss, per 18 Da block",
            ctx.plot_dir / f"{stem}_importance_agreement.png"))
    return out


# --- the run ---------------------------------------------------------------------------------------------------

def run(ctx: Context) -> None:
    started = time.perf_counter()
    ctx.write_status("running")
    X_val, y_val, prob_val, validation_auroc = validation_probabilities(ctx)

    exact = has_exact_contributions(ctx)
    importance = regions = method_agreement = control = None
    if exact:
        importance, regions, _ = importance_and_regions(ctx, X_val)
        write_csv(importance, ctx.report_dir / "global_importance.csv")
        write_csv(regions, ctx.report_dir / "regions.csv")
        contrast = class_contrast(load_rows(ctx.X, ctx.train_rows), ctx.y[ctx.train_rows], regions,
                                  span=ctx.span)
        write_csv(contrast, ctx.report_dir / "region_contrast.csv")

    per_block, single = permutation_tables(ctx, X_val, y_val, importance)
    write_csv(per_block, ctx.report_dir / "permutation_blocks.csv")
    if single is not None:
        write_csv(single, ctx.report_dir / "permutation_single_bins.csv")
    blocks = contiguous_blocks(int(ctx.X.shape[1]), int(ctx.xc["permutation"]["block_bins"]))

    if exact:
        method_agreement = rank_agreement(block_shap(importance, blocks, ctx.span),
                                          per_block.sort_values("bin_start")["auroc_drop_mean"].to_numpy(),
                                          top_k=20)
        stability, pairs = seed_stability(ctx, X_val, regions)
        write_csv(stability, ctx.report_dir / "seed_stability.csv")
        write_csv(pairs, ctx.report_dir / "seed_agreement.csv")

        control = null_control(ctx, X_val, regions)
        (ctx.report_dir / "null_control.json").write_text(json.dumps(control, indent=2), encoding="utf-8")
        log.info("shuffled-label control: strongest real region %.4f against %.4f by chance (ratio %.1f)",
                 control["real_strongest_region"], control["null_strongest_region"],
                 control["real_over_null"])

    other_blocks, cross = cross_model(ctx, X_val, y_val, per_block)
    if other_blocks is not None:
        write_csv(other_blocks, ctx.report_dir / "permutation_blocks_cross_model.csv")

    zones, zone_record = fit_and_write_zones(ctx, y_val, prob_val)
    validation_zones = zone_metrics(y_val, prob_val, zones)
    test_zones, test_intervals = stored_test_summary(ctx, zones)
    write_csv(pd.DataFrame([{"part": "validation", **validation_zones}] +
                           ([{"part": "test (from stored predictions)", **test_zones}] if test_zones else [])),
              ctx.report_dir / "zone_metrics.csv")
    if test_intervals:
        (ctx.report_dir / "zone_intervals.json").write_text(json.dumps(test_intervals, indent=2),
                                                            encoding="utf-8")

    if exact:
        write_csv(example_explanations(ctx, X_val, y_val, prob_val, zones), ctx.report_dir / "examples.csv")

    curve = pd.read_csv(ctx.report_dir / "uncertainty_curve.csv")
    made = figures(ctx, X_val, y_val, prob_val, regions, importance, per_block, curve, zones)
    log.info("figures: %s", ", ".join(p.name for p in made))

    ctx.assert_test_log_untouched()
    ctx.write_config(seconds=round(time.perf_counter() - started, 1),
                     validation_roc_auc=validation_auroc, exact_contributions=exact,
                     method_agreement=method_agreement,
                     cross_model_agreement=cross, null_control=control, zones=zone_record,
                     test_log_sha256=ctx.test_log_hash, plots=[p.name for p in made])
    log.info("Version 0.6 reports written to %s", ctx.report_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain the project's saved model and fit its confidence "
                                                 "zones (validation rows only; no test row is scored).")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--skip-cross-model", action="store_true",
                        help="leave out the Version 0.5 network comparison")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    ctx = None
    try:
        config = load_config(args.config)
        set_seed(int(config["project"]["random_seed"]))
        if "explain" not in config:
            raise ConfigError("config.yaml has no 'explain' section (Version 0.6 settings).")
        ctx = Context(config, args.dataset, args.skip_cross_model)
        with keep_awake():
            started = time.perf_counter()
            run(ctx)
            log.info("finished in %.1f minutes", (time.perf_counter() - started) / 60)
    except (ConfigError, DataError, SplitError, ModelError, TuningError, EvaluationError, ExplainError,
            UncertaintyError) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        if ctx is not None:
            ctx.write_status("failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    except Exception as exc:                       # unexpected: record why, then fail loudly
        log.exception("unexpected error")
        if ctx is not None:
            ctx.write_status("failed", error=f"{type(exc).__name__}: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
