"""Version 0.7 - how much does the model lose at another hospital, or in a later year?

Run from the project root with the virtual environment active:

    python scripts/measure_generalisation.py

This is the version protocol decision 7 released the `temporal` and `external` test parts for, so it is
the one script in the project that spends them. Everything it may do was fixed first in
docs/v0.7_generalisation_plan.md and protocol amendment 3, committed before this file existed.

**No setting is chosen here.** The Version 0.4 winning LightGBM setting is read from its saved model card
and refitted unchanged on each experiment's own training part, so a gap between experiments can only come
from the data. Before anything is scored, the refit is proved faithful: the same setting refitted on the
`random` training part must reproduce the saved model's logged validation AUROC.

Writes (paths from config.yaml -> generalisation):
- results/metrics/v0.7/<dataset>/* and results/plots/v0.7/*  (committed; no identifiers)
- models/v0.7/<dataset>/test_probabilities.npz               (git-ignored; lets later versions reuse
                                                              these scores instead of re-scoring)
- one appended row per test evaluation in results/experiments/test_evaluations.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import DataError  # noqa: E402
from src.dataset import load_dataset  # noqa: E402
from src.evaluate import (  # noqa: E402
    EvaluationError,
    append_test_log,
    bootstrap,
    choose_threshold,
    classification_metrics,
    summarize_bootstrap,
    unpaired_difference,
)
from src.predict import ModelError, load_bundle, predict_features  # noqa: E402
from src.splits import SplitError, assert_usable, check_split, load_splits  # noqa: E402
from src.train import load_rows  # noqa: E402
from src.tuning import (  # noqa: E402,E501
    FamilySpec,
    cache_key,
    cached,
    code_fingerprint,
    fit_calibrated,
    grouped_folds,
    set_threads,
)
from src.uncertainty import Zones, bootstrap_zone_metrics, zone_metrics  # noqa: E402
from src.utils import (  # noqa: E402
    ConfigError,
    file_hash,
    get_logger,
    git_commit,
    keep_awake,
    load_config,
    project_path,
    set_seed,
    show_path,
)

log = get_logger("generalise")

STAGE = "v0.7-generalisation"
REFERENCE_STAGE = "v0.7-reference"
# The refit must reproduce the saved model's logged validation AUROC. A calibrated LightGBM fitted from
# the same rows, setting and seed is deterministic, so this is exact rather than approximate; the
# tolerance only absorbs the decimals the report was written with.
FAITHFUL = 1e-9
PREVALENCE = "prevalence"


def versions() -> dict[str, str]:
    import joblib
    import lightgbm
    import sklearn

    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "joblib": joblib.__version__}


class Context:
    """Everything the run needs, with every provenance check done in the constructor."""

    def __init__(self, config: dict[str, Any], dataset_name: str | None) -> None:
        self.config = config
        self.gc = config["generalisation"]
        self.ev = config["evaluation"]
        self.tc = config["tuning"]
        self.dataset_name = dataset_name or config["dataset"]["name"]
        self.commit = git_commit()
        self.versions = versions()

        data_dir = project_path(config["dataset"]["output_dir"]) / self.dataset_name
        self.X, self.meta, self.summary = load_dataset(data_dir, verify_x=True)
        self.splits = load_splits(data_dir / "splits", self.meta)
        self.y = self.meta["label"].to_numpy().astype(np.int64)
        self.groups = self.meta["group_id"].to_numpy()
        self.seed = int(config["project"]["random_seed"])
        self.seeds = [int(s) for s in self.gc["seeds"]]

        self.source_split = str(self.gc["source_split"])
        self.experiments = [str(name) for name in self.gc["experiments"]]
        for name in (self.source_split, *self.experiments):
            if name not in self.splits:
                raise SplitError(f"The dataset has no {name!r} split; rebuild the splits "
                                 "(python scripts/build_dataset.py --splits-only).")
            check_split(self.meta, self.splits[name])
            assert_usable(self.meta, self.splits[name])

        model_root = project_path(self.gc["source_model"]) / self.dataset_name
        self.bundle_path = model_root / f"best_{self.source_split}.joblib"
        self.bundle = load_bundle(self.bundle_path, n_jobs=1)
        self._check_bundle_matches_dataset()
        # Checked here, before a single row is fitted or scored: whether the saved model is allowed on the
        # parts it is asked about is a property of the data, so there is no reason to find out late.
        for name in [str(s) for s in self.gc["score_saved_model_on"]]:
            if name not in self.splits:
                raise SplitError(f"generalisation.score_saved_model_on names {name!r}, which is not a split.")
            self.assert_saved_model_may_be_scored_on(name)
        self.card = self._model_card()
        self.setting = dict(self.card["params"])
        self.spec = self._family_spec()

        self.report_dir = project_path(self.gc["report_dir"]) / self.dataset_name
        self.plot_dir = project_path(self.gc["plot_dir"])
        self.model_dir = project_path(self.gc["model_dir"]) / self.dataset_name
        self.cache_dir = project_path(self.gc["model_dir"]) / "cache" / self.dataset_name
        for folder in (self.report_dir, self.plot_dir, self.model_dir, self.cache_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.test_log = project_path(self.ev["test_log"])
        self.test_log_hash = file_hash(self.test_log)
        # src/train.py is not in the key: nothing here goes through it. src/tuning.py is, because
        # fit_calibrated and base_pipeline decide what a refit actually is.
        self.code = code_fingerprint([Path(__file__).resolve().parents[1] / "src" / "tuning.py"])

    def _check_bundle_matches_dataset(self) -> None:
        """The saved model must belong to this dataset build, or every comparison below is meaningless."""
        stored = self.bundle.get("dataset") or {}
        for key, ours in (("row_fingerprint", self.summary["row_fingerprint"]),
                          ("x_sha256", self.summary["x_sha256"])):
            theirs = stored.get(key)
            if theirs and theirs != ours:
                raise ModelError(f"{show_path(self.bundle_path)} was made for a dataset whose {key} is "
                                 f"{theirs}, but this one is {ours}.")
        if self.bundle["feature_fingerprint"] != self.summary["feature_fingerprint"]:
            raise ModelError(f"{show_path(self.bundle_path)} was made with other preprocessing settings "
                             f"({self.bundle['feature_fingerprint']} against "
                             f"{self.summary['feature_fingerprint']}).")

    def _model_card(self) -> dict[str, Any]:
        """The card of the model whose setting is reused, found through its own config section."""
        wanted = str(self.gc["source_model"]).rstrip("/")
        for name, section in self.config.items():
            if name == "generalisation" or not isinstance(section, dict) or "report_dir" not in section:
                continue
            if str(section.get("model_dir", "")).rstrip("/") == wanted:
                path = project_path(section["report_dir"]) / self.dataset_name / "best_model_card.json"
                if not path.is_file():
                    raise ConfigError(f"{show_path(path)} does not exist, so the setting to reuse cannot "
                                      "be read. Run that version first.")
                return json.loads(path.read_text(encoding="utf-8"))
        raise ConfigError(f"No config section other than 'generalisation' has model_dir {wanted!r}, so the "
                          "reports of the model whose setting is reused cannot be found.")

    def _family_spec(self) -> FamilySpec:
        """The family of the reused setting, from the section that searched it."""
        kind = str(self.card["model_kind"])
        for name, cfg in self.tc["families"].items():
            if str(cfg.get("kind")) == kind:
                return FamilySpec.from_config(name, cfg)
        raise ConfigError(f"No tuning family has kind {kind!r}, so the reused setting cannot be rebuilt.")

    def rows(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = self.splits[name]
        return (np.asarray(s.train, dtype=np.int64), np.asarray(s.validation, dtype=np.int64),
                np.asarray(s.test, dtype=np.int64))

    def assert_saved_model_may_be_scored_on(self, name: str) -> None:
        """The saved model may only be scored where it has seen neither the rows nor the patients.

        For the `temporal` part this refuses: that part is date-separated from its own training part, but
        the saved model was trained on the source split's rows, which reach into the later year.
        """
        source_train, _, _ = self.rows(self.source_split)
        _, _, test = self.rows(name)
        rows = np.intersect1d(source_train, test).size
        groups = np.intersect1d(np.unique(self.groups[source_train]), np.unique(self.groups[test])).size
        if rows or groups:
            raise EvaluationError(
                f"The saved model's training part shares {rows} row(s) and {groups} patient group(s) with "
                f"the {name} test part, so scoring it there would be leakage, not generalisation "
                "(protocol amendment 3, point 3). Nothing was fitted or scored.")

    def sites_of(self, rows: np.ndarray) -> list[str]:
        return sorted(set(self.meta["site"].to_numpy()[rows].tolist()))

    def record(self, status: str, **extra: Any) -> dict[str, Any]:
        return {"stage": STAGE, "status": status, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": time.strftime("%Z%z"), "git_commit": self.commit,
                "code_fingerprint": self.code, "dataset": self.dataset_name,
                "rows_fingerprint": self.summary["row_fingerprint"], "x_sha256": self.summary["x_sha256"],
                "feature_fingerprint": self.summary["feature_fingerprint"],
                "reused_setting_from": self.card["model_version"], "setting": self.setting,
                "experiments": self.experiments, "seeds": self.seeds,
                "versions": self.versions, **extra, "config": self.config}

    def write_status(self, status: str, **extra: Any) -> None:
        (self.report_dir / "run_status.json").write_text(
            json.dumps(self.record(status, **extra), indent=2, default=str), encoding="utf-8")

    def write_config(self, **extra: Any) -> None:
        record = self.record("finished", **extra)
        for name in ("run_config.json", "run_status.json"):
            (self.report_dir / name).write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    def assert_test_log_only_grew(self, before: pd.DataFrame) -> None:
        """The log is append-only: every row that was there must still be there, unchanged."""
        after = pd.read_csv(self.test_log) if self.test_log.is_file() else pd.DataFrame()
        if len(after) < len(before):
            raise EvaluationError(f"{show_path(self.test_log)} lost rows during this run.")
        if len(before):
            pd.testing.assert_frame_equal(after.iloc[:len(before)].reset_index(drop=True),
                                          before.reset_index(drop=True))


# --- fitting the one fixed setting -----------------------------------------------------------------------------

def refit(ctx: Context, name: str, seed: int) -> tuple[Any, float, dict[str, Any], np.ndarray]:
    """Refit the reused setting on one experiment's training part and take its threshold from validation.

    Cached on the rows, the setting, the seed and the text of src/tuning.py, so a rerun of this script does
    not refit anything, and editing what a refit means invalidates the cache instead of silently reusing it.
    """
    tr, va, _ = ctx.rows(name)
    X_tr, y_tr = load_rows(ctx.X, tr), ctx.y[tr]
    folds = grouped_folds(y_tr, ctx.groups[tr], int(ctx.tc["cv_folds"]), int(ctx.tc["cv_seed"]))
    key = cache_key(dataset=ctx.summary["row_fingerprint"], experiment=name, train=tr.tolist(),
                    setting=ctx.setting, seed=seed, folds=int(ctx.tc["cv_folds"]),
                    cv_seed=int(ctx.tc["cv_seed"]), calibration=str(ctx.tc["calibration"]),
                    code=ctx.code, versions=ctx.versions)

    def compute():
        started = time.perf_counter()
        fitted = fit_calibrated(ctx.spec, ctx.setting, X_tr, y_tr, folds, seed,
                                method=str(ctx.tc["calibration"]))
        return {"model": fitted, "fit_seconds": round(time.perf_counter() - started, 1)}

    stored = cached(ctx.cache_dir / f"{name}_seed{seed}.joblib", key, compute, log)
    model = stored["model"]
    set_threads(model, 1)              # predictions on one thread, so a rerun reproduces them exactly
    X_va, y_va = load_rows(ctx.X, va), ctx.y[va]
    val_prob = model.predict_proba(X_va)[:, 1]
    threshold = choose_threshold(y_va, val_prob, ctx.ev["threshold"])
    validation = classification_metrics(y_va, val_prob, threshold)
    return model, threshold, {"fit_seconds": stored["fit_seconds"], **validation}, val_prob


def prove_the_refit_is_faithful(ctx: Context) -> dict[str, Any]:
    """Refit the reused setting on the source experiment and check it reproduces the saved model.

    This runs before any locked test part is touched. If it fails, the refits below would not be the
    Version 0.4 model in another data regime, and the whole comparison would be meaningless.
    """
    _, threshold, validation, _ = refit(ctx, ctx.source_split, int(ctx.card["seed"]))
    logged = float(ctx.card["metrics"]["validation"]["roc_auc"])
    gap = abs(validation["roc_auc"] - logged)
    logged_threshold = float(ctx.card["threshold"])
    if gap > FAITHFUL:
        raise EvaluationError(
            f"Refitting the reused setting on the {ctx.source_split} training part gives validation AUROC "
            f"{validation['roc_auc']:.12f}, but {ctx.card['model_version']} logged {logged:.12f} "
            f"(difference {gap:.1e}). The refit is not the saved model, so no gap measured against it "
            "would mean anything. Nothing was scored.")
    log.info("refit proved faithful: the reused setting reproduces %s on validation "
             "(AUROC %.12f, difference %.1e)", ctx.card["model_version"], validation["roc_auc"], gap)
    return {"experiment": ctx.source_split, "seed": int(ctx.card["seed"]),
            "refit_validation_roc_auc": validation["roc_auc"], "logged_validation_roc_auc": logged,
            "difference": gap, "refit_threshold": threshold, "logged_threshold": logged_threshold,
            "threshold_difference": abs(threshold - logged_threshold)}


# --- scoring ---------------------------------------------------------------------------------------------------

def per_site(ctx: Context, rows: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Test rows grouped by site, plus nothing else: a pooled external number is only ever secondary.

    A single-site test part yields one entry, so `external` reports B and D apart while `temporal` and
    `external_ab` each report one.
    """
    sites = ctx.meta["site"].to_numpy()[rows]
    return [(site, rows[sites == site]) for site in sorted(set(sites.tolist()))]


def score(ctx: Context, name: str, model: Any, threshold: float, rows: np.ndarray) -> tuple[np.ndarray, dict]:
    X_te = load_rows(ctx.X, rows)
    started = time.perf_counter()
    prob = model.predict_proba(X_te)[:, 1]
    ms = (time.perf_counter() - started) * 1000 / max(rows.size, 1)
    metrics = classification_metrics(ctx.y[rows], prob, threshold)
    log.info("%s on %s (%d rows, %d resistant): AUROC %.4f, PR-AUC %.4f, Brier %.4f", name,
             "+".join(ctx.sites_of(rows)), rows.size, int((ctx.y[rows] == 1).sum()),
             metrics["roc_auc"], metrics["pr_auc"], metrics["brier"])
    return prob, {**metrics, "predict_ms_per_sample": round(ms, 4)}


def prevalence_row(ctx: Context, name: str, train_rows: np.ndarray, rows: np.ndarray) -> dict[str, Any]:
    """Protocol section 5: what a model that always predicts the training resistance rate would score."""
    rate = float((ctx.y[train_rows] == 1).mean())
    prob = np.full(rows.size, rate, dtype=np.float64)
    return {"experiment": name, "split": name, "model": PREVALENCE, "seed": ctx.seed,
            "train_size": int(train_rows.size), "setting": f"always predicts {rate:.4f}",
            **classification_metrics(ctx.y[rows], prob, rate)}


def saved_model_rows(ctx: Context, name: str) -> list[dict[str, Any]]:
    """The saved project model, scored on a test part it has never seen, with no refit.

    Its threshold came from the *source* split's validation part, not from this site, which is stated in
    the row itself because it is a real limitation of this comparison (amendment 3, point 3).
    """
    ctx.assert_saved_model_may_be_scored_on(name)      # already checked at start-up; cheap to repeat here
    _, _, test = ctx.rows(name)
    rows_out = []
    for site, site_rows in per_site(ctx, test):
        prob = predict_features(ctx.bundle, load_rows(ctx.X, site_rows))[0]
        metrics = classification_metrics(ctx.y[site_rows], prob, float(ctx.bundle["threshold"]))
        log.info("saved %s on %s (%d rows): AUROC %.4f", ctx.bundle["model_version"], site,
                 site_rows.size, metrics["roc_auc"])
        # The site belongs in the experiment name: the test log has no site column, so two sites scored
        # under one name would be two rows nothing could tell apart.
        rows_out.append({"experiment": f"{name}__{site}__saved_model", "split": name,
                         "model": f"v0.4_{ctx.bundle['model']}", "seed": int(ctx.bundle["seed"]),
                         "train_size": int(ctx.bundle["split"]["train_size"]),
                         "setting": f"saved model, unchanged; threshold from the {ctx.source_split} "
                                    "validation part, not from this site",
                         "site": site, "prob": prob, **metrics})
    return rows_out


# --- reusing the source result instead of scoring it again ------------------------------------------------------

def stored_source_probabilities(ctx: Context) -> dict[int, np.ndarray]:
    """The source split's saved test probabilities, verified against the log. Nothing is scored again.

    The guard Version 0.6 established: the rows must be that test part's own rows in its own order, and the
    probabilities must reproduce both the logged AUROC and the logged Brier score. AUROC alone would not
    do, because it survives any monotone rescaling of the probabilities.
    """
    path = project_path(ctx.gc["source_model"]) / ctx.dataset_name / "test_probabilities.npz"
    if not path.is_file():
        raise EvaluationError(f"{show_path(path)} does not exist, so the {ctx.source_split} result cannot be "
                              "reused and would have to be scored again. Refusing to do that.")
    stored = np.load(path)
    _, _, test = ctx.rows(ctx.source_split)
    if not np.array_equal(stored[f"{ctx.source_split}__rows"], test):
        raise EvaluationError(f"The stored rows in {path.name} are not this dataset's {ctx.source_split} test "
                              "part (order included); refusing to derive anything from them.")
    model = str(ctx.card["model"])
    out: dict[int, np.ndarray] = {}
    for seed in ctx.seeds:
        key = f"{ctx.source_split}__{model}__seed{seed}"
        if key in stored.files:
            out[seed] = np.asarray(stored[key], dtype=np.float64)
        else:
            log.warning("%s holds no %s; that seed is left out of the comparison", path.name, key)
    main_seed = int(ctx.card["seed"])
    if main_seed not in out:
        raise EvaluationError(f"{path.name} holds no probabilities for seed {main_seed}, the seed of the "
                              "saved model, so the comparison would have no reference.")
    logged = pd.read_csv(ctx.test_log)
    row = logged[(logged["split"] == ctx.source_split) & (logged["experiment"] == ctx.source_split)
                 & (logged["model"] == model) & (logged["seed"] == main_seed)]
    if row.empty:
        raise EvaluationError(f"The test log has no {ctx.source_split} row for {model} seed {main_seed}, so "
                              "the stored probabilities cannot be checked against anything.")
    got = classification_metrics(ctx.y[test], out[main_seed], float(ctx.card["threshold"]))
    for metric in ("roc_auc", "brier"):
        want = float(row[metric].iloc[-1])
        if abs(got[metric] - want) > 1e-6:                 # the log stores six decimals
            raise EvaluationError(f"The stored {ctx.source_split} probabilities give {metric} "
                                  f"{got[metric]:.12f} but {want:.12f} was logged. They are not the scored "
                                  "predictions; refusing to derive from them.")
    log.info("reusing the stored %s test probabilities for %d seed(s), verified against the log "
             "(AUROC %.6f, Brier %.6f, same %d rows)", ctx.source_split, len(out), got["roc_auc"],
             got["brier"], test.size)
    return out


# --- the comparisons the plan pre-specified --------------------------------------------------------------------

def auroc_samples(ctx: Context, rows: np.ndarray, prob: np.ndarray, threshold: float) -> np.ndarray:
    """Bootstrap AUROC draws for one test part, resampling whole patient groups."""
    samples, _ = bootstrap(ctx.y[rows], ctx.groups[rows], {"m": prob}, {"m": threshold},
                           resamples=int(ctx.ev["bootstrap"]["resamples"]),
                           seed=int(ctx.ev["bootstrap"]["seed"]), metrics=("roc_auc",))
    return samples["m"]["roc_auc"]


def generalisation_gaps(ctx: Context, source_prob: np.ndarray, source_threshold: float,
                        scored: list[dict[str, Any]]) -> pd.DataFrame:
    """`random` minus each other test part. These test sets share no rows, so the interval is the
    second-level unpaired bootstrap: wider than a paired one, and meant to be (protocol section 6)."""
    _, _, source_rows = ctx.rows(ctx.source_split)
    base = auroc_samples(ctx, source_rows, source_prob, source_threshold)
    source_point = classification_metrics(ctx.y[source_rows], source_prob, source_threshold)["roc_auc"]
    level = float(ctx.ev["bootstrap"]["level"])
    out = []
    for e in scored:
        draws = auroc_samples(ctx, e["rows"], e["prob"], e["threshold"])
        low, high = unpaired_difference(base, draws, level, seed=int(ctx.ev["bootstrap"]["seed"]))
        demonstrated = bool(low > 0 or high < 0)
        out.append({"comparison": f"{ctx.source_split} minus {e['label']}",
                    "kind": "unpaired (different rows)", "source_roc_auc": source_point,
                    "other_roc_auc": e["roc_auc"], "gap": source_point - e["roc_auc"],
                    "low": low, "high": high, "n": int(e["rows"].size),
                    "n_resistant": int((ctx.y[e["rows"]] == 1).sum()), "demonstrated": demonstrated})
        log.info("gap, %s: %.4f [%.4f, %.4f] %s", out[-1]["comparison"], out[-1]["gap"], low, high,
                 "(excludes 0)" if demonstrated else "(includes 0: not demonstrated)")
    return pd.DataFrame(out)


def same_row_pairs(refits: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every pair of different experiments scored on exactly the same rows, fewer training sites first.

    Found from the rows themselves rather than from experiment names, so the comparison the specification
    asks for cannot be silently skipped because a site was named differently.
    """
    out = []
    for i, a in enumerate(refits):
        for b in refits[i + 1:]:
            if a["experiment"] != b["experiment"] and np.array_equal(a["rows"], b["rows"]):
                first, second = sorted((a, b), key=lambda e: (len(e["train_sites"]), e["train_size"]))
                out.append((first, second))
    return out


def two_sites_versus_one(ctx: Context, one: dict[str, Any] | None,
                         two: dict[str, Any] | None) -> dict[str, Any] | None:
    """One training regime minus the other on the *same* rows, so this comparison is paired and tighter."""
    if one is None or two is None:
        log.warning("no two experiments were scored on the same rows; the paired comparison is left out")
        return None
    if not np.array_equal(one["rows"], two["rows"]):
        raise EvaluationError("The two training regimes were not scored on the same rows, so a paired "
                              "comparison would be wrong.")
    rows = one["rows"]
    summary, _ = summarize_bootstrap(
        ctx.y[rows], ctx.groups[rows], {"one_site": one["prob"], "two_sites": two["prob"]},
        {"one_site": one["threshold"], "two_sites": two["threshold"]},
        resamples=int(ctx.ev["bootstrap"]["resamples"]), level=float(ctx.ev["bootstrap"]["level"]),
        seed=int(ctx.ev["bootstrap"]["seed"]), reference="two_sites")
    diff = summary["differences_to_reference"]["one_site"]["roc_auc"]
    demonstrated = bool(diff["low"] > 0 or diff["high"] < 0)
    log.info("two training sites minus one, same %d rows: %.4f [%.4f, %.4f] %s", rows.size,
             diff["estimate"], diff["low"], diff["high"],
             "(excludes 0)" if demonstrated else "(includes 0: not demonstrated)")
    return {"comparison": f"{two['experiment']} minus {one['experiment']}, on the same rows",
            "kind": "paired (same rows)", "part": one["label"],
            "fewer_sites": {"experiment": one["experiment"], "train_sites": one["train_sites"],
                            "train_size": one["train_size"]},
            "more_sites": {"experiment": two["experiment"], "train_sites": two["train_sites"],
                           "train_size": two["train_size"]},
            "n": int(rows.size), "n_resistant": int((ctx.y[rows] == 1).sum()),
            "fewer_sites_roc_auc": summary["intervals"]["one_site"]["roc_auc"]["estimate"],
            "more_sites_roc_auc": summary["intervals"]["two_sites"]["roc_auc"]["estimate"],
            "difference": diff["estimate"], "low": diff["low"], "high": diff["high"],
            "demonstrated": demonstrated}


# --- do the Version 0.6 confidence zones survive a change of site? ---------------------------------------------

def load_zones(ctx: Context) -> Zones | None:
    path = project_path(ctx.gc["zone_transfer"]["zones"]) / ctx.dataset_name / "uncertainty.json"
    if not path.is_file():
        log.warning("%s does not exist; the zone-transfer section is left out", show_path(path))
        return None
    return Zones.from_dict(json.loads(path.read_text(encoding="utf-8")))


def zone_transfer(ctx: Context, zones: Zones, entries: list[dict[str, Any]]) -> pd.DataFrame:
    """Apply the Version 0.6 edge unchanged and report coverage and NPV with intervals.

    `pure_site_test` marks the rows where this is a clean test of *site* transfer: the same saved model,
    the same edge, new spectra. The refits are reported too, but each carries its own calibration step, so
    a difference there mixes the new site with a recalibrated probability scale. That distinction is a
    column rather than a footnote, because the two do not mean the same thing.
    """
    target = float(ctx.gc["zone_transfer"]["target_npv"])
    rows = []
    for e in entries:
        y, prob = ctx.y[e["rows"]], e["prob"]
        metrics = zone_metrics(y, prob, zones)
        intervals = bootstrap_zone_metrics(y, ctx.groups[e["rows"]], prob, zones,
                                           resamples=int(ctx.ev["bootstrap"]["resamples"]),
                                           level=float(ctx.ev["bootstrap"]["level"]),
                                           seed=int(ctx.ev["bootstrap"]["seed"]))
        npv = intervals["npv_susceptible"]
        transfers = bool(np.isfinite(npv["value"]) and (npv["value"] >= target or npv["high"] >= target))
        rows.append({"part": e["label"], "model": e["model"], "pure_site_test": e["pure"],
                     "n": int(e["rows"].size), "n_resistant": int((y == 1).sum()), "edge": zones.lower,
                     "n_zone_susceptible": metrics.get("n_zone_susceptible"),
                     "share_susceptible": metrics.get("share_susceptible"),
                     "npv_susceptible": npv["value"], "npv_low": npv["low"], "npv_high": npv["high"],
                     "target_npv": target, "transfers": transfers})
        log.info("zone transfer, %s (%s): covers %.1f%% at NPV %.3f [%.3f, %.3f] -> %s", e["label"],
                 e["model"], 100 * (metrics.get("share_susceptible") or 0.0), npv["value"], npv["low"],
                 npv["high"], "holds" if transfers else "does NOT hold")
    return pd.DataFrame(rows)


# --- why performance changes: a feature-only diagnostic --------------------------------------------------------

def shift_diagnostic(ctx: Context, test_parts: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe the input shift between the source training site and each test part. No AST label is used.

    Runs after every scoring is logged, so it cannot influence a model or a threshold. It describes the
    shift; it does not correct for it.
    """
    low_bins = int(ctx.gc["shift_diagnostic"]["low_mz_bins"])
    train_rows, _, _ = ctx.rows(ctx.source_split)
    reference = load_rows(ctx.X, train_rows).astype(np.float64)
    ref_mean, ref_var = reference.mean(axis=0), reference.var(axis=0, ddof=1)
    summary = []
    for label, rows in test_parts.items():
        other = load_rows(ctx.X, rows).astype(np.float64)
        pooled = np.sqrt((ref_var + other.var(axis=0, ddof=1)) / 2)
        smd = np.where(pooled > 0, (other.mean(axis=0) - ref_mean) / np.where(pooled > 0, pooled, 1.0), 0.0)
        summary.append({"part": label, "n": int(rows.size),
                        "median_abs_smd": float(np.median(np.abs(smd))),
                        "share_bins_abs_smd_over_half": float(np.mean(np.abs(smd) > 0.5)),
                        "max_abs_smd": float(np.abs(smd).max()),
                        "median_abs_smd_lowest_bins": float(np.median(np.abs(smd[:low_bins]))),
                        "lowest_bins": low_bins,
                        "empty_share_lowest_bins_reference": float(np.mean(reference[:, :low_bins] == 0)),
                        "empty_share_lowest_bins_part": float(np.mean(other[:, :low_bins] == 0))})
        log.info("shift, %s: median |SMD| %.3f over all bins, %.3f over the lowest %d bins; empty share "
                 "there %.2f against %.2f at the training site", label, summary[-1]["median_abs_smd"],
                 summary[-1]["median_abs_smd_lowest_bins"], low_bins,
                 summary[-1]["empty_share_lowest_bins_part"],
                 summary[-1]["empty_share_lowest_bins_reference"])
    return pd.DataFrame(summary), _region_shift(ctx, reference, test_parts)


def _region_shift(ctx: Context, reference: np.ndarray, test_parts: dict[str, np.ndarray]) -> pd.DataFrame:
    """The same measure, restricted to the m/z regions Version 0.6 found the model relies on."""
    path = project_path(ctx.gc["shift_diagnostic"]["regions_from"]) / ctx.dataset_name / "regions.csv"
    if not path.is_file():
        log.warning("%s does not exist; the per-region shift is left out", show_path(path))
        return pd.DataFrame()
    wanted = pd.read_csv(path)
    rows = []
    for r in wanted.itertuples(index=False):
        first, last = int(r.first_column), int(r.last_column) + 1
        ref_total = reference[:, first:last].sum(axis=1)
        for label, part_rows in test_parts.items():
            other = load_rows(ctx.X, part_rows).astype(np.float64)[:, first:last].sum(axis=1)
            pooled = np.sqrt((ref_total.var(ddof=1) + other.var(ddof=1)) / 2)
            rows.append({"rank": int(r.rank), "mz_start": float(r.mz_start), "mz_end": float(r.mz_end),
                         "towards": str(r.towards), "part": label,
                         "smd": float((other.mean() - ref_total.mean()) / pooled) if pooled > 0 else 0.0})
    return pd.DataFrame(rows)


# --- the run ---------------------------------------------------------------------------------------------------

def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def run(ctx: Context) -> None:
    started = time.perf_counter()
    ctx.write_status("running")
    before = pd.read_csv(ctx.test_log) if ctx.test_log.is_file() and ctx.test_log.stat().st_size else pd.DataFrame()

    # Nothing may be scored until the refit is proved to be the saved model in another data regime.
    faithful = prove_the_refit_is_faithful(ctx)
    source = stored_source_probabilities(ctx)
    main_seed = int(ctx.card["seed"])
    model_name = str(ctx.card["model"])

    log_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []          # seed-42 scorings, per experiment and site
    validation_rows: list[dict[str, Any]] = []

    base = {"logged_at": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": ctx.commit,
            "dataset": ctx.dataset_name, "dataset_fingerprint": ctx.summary["row_fingerprint"],
            "x_sha256": ctx.summary["x_sha256"]}

    for name in ctx.experiments:
        tr, _, te = ctx.rows(name)
        arrays[f"{name}__rows"] = te
        for seed in ctx.seeds:
            model, threshold, validation, _ = refit(ctx, name, seed)
            validation_rows.append({"experiment": name, "seed": seed, "train_size": int(tr.size),
                                    "train_sites": "+".join(ctx.sites_of(tr)), **validation})
            for site, site_rows in per_site(ctx, te):
                label = f"{name}__{site}"
                prob, metrics = score(ctx, f"{label} seed {seed}", model, threshold, site_rows)
                arrays[f"{label}__seed{seed}"] = prob
                row = {"experiment": label, "split": name, "model": model_name, "seed": seed,
                       "train_size": int(tr.size), "threshold": threshold, **metrics}
                log_rows.append({**base, "stage": STAGE, **row})
                report_rows.append({**row, "site": site, "train_sites": "+".join(ctx.sites_of(tr)),
                                    "setting": "Version 0.4 winning setting, refitted; threshold from "
                                               "this experiment's own validation part"})
                if seed == main_seed:
                    entries.append({"label": label, "experiment": name, "site": site, "rows": site_rows,
                                    "prob": prob, "threshold": threshold, "roc_auc": metrics["roc_auc"],
                                    "model": f"refit ({name})", "pure": False,
                                    "train_sites": ctx.sites_of(tr), "train_size": int(tr.size)})
        for site, site_rows in per_site(ctx, te):
            row = prevalence_row(ctx, f"{name}__{site}", tr, site_rows)
            log_rows.append({**base, "stage": REFERENCE_STAGE, **row, "split": name})
            report_rows.append({**row, "split": name, "site": site,
                                "train_sites": "+".join(ctx.sites_of(tr))})

    for name in [str(s) for s in ctx.gc["score_saved_model_on"]]:
        for row in saved_model_rows(ctx, name):
            prob, site = row.pop("prob"), row["site"]
            arrays[f"{name}__saved_model__{site}"] = prob
            _, _, te = ctx.rows(name)
            site_rows = te[ctx.meta["site"].to_numpy()[te] == site]
            log_rows.append({**base, "stage": REFERENCE_STAGE, **{k: v for k, v in row.items() if k != "site"},
                             "threshold": float(ctx.bundle["threshold"])})
            report_rows.append({**row, "threshold": float(ctx.bundle["threshold"])})
            entries.append({"label": f"{name}__{site}", "experiment": name, "site": site, "rows": site_rows,
                            "prob": prob, "threshold": float(ctx.bundle["threshold"]),
                            "roc_auc": row["roc_auc"], "model": "saved v0.4 model", "pure": True})

    # The log is appended once, before any other output is written, so a crash later cannot leave a
    # scoring unrecorded. It is append-only and the run checks that nothing earlier changed.
    append_test_log(ctx.test_log, log_rows, locked=ctx.ev["locked_test_splits"])
    ctx.assert_test_log_only_grew(before)
    log.info("%d test evaluations appended to %s", len(log_rows), show_path(ctx.test_log))

    np.savez_compressed(ctx.model_dir / "test_probabilities.npz", **arrays,
                        **{f"{ctx.source_split}__{model_name}__seed{s}": p for s, p in source.items()})
    write_csv(pd.DataFrame(report_rows), ctx.report_dir / "test_metrics.csv")
    write_csv(pd.DataFrame(validation_rows), ctx.report_dir / "validation_metrics.csv")

    # Comparisons. The saved-model rows are left out of the gap table: that model's threshold comes from
    # another site's validation part, so it answers a different question and gets its own rows above.
    refits = [e for e in entries if not e["pure"]]
    gaps = generalisation_gaps(ctx, source[main_seed], float(ctx.card["threshold"]), refits)
    write_csv(gaps, ctx.report_dir / "generalisation_gaps.csv")
    pairs = same_row_pairs(refits)
    if not pairs:
        log.warning("no two experiments were scored on the same rows, so the specification's "
                    "'two training sites against one' comparison is not in this run")
    paired = [p for p in (two_sites_versus_one(ctx, one, two) for one, two in pairs) if p]
    if paired:
        (ctx.report_dir / "two_sites_versus_one.json").write_text(
            json.dumps(paired if len(paired) > 1 else paired[0], indent=2, default=str), encoding="utf-8")

    zones = load_zones(ctx)
    transfer = zone_transfer(ctx, zones, entries) if zones else pd.DataFrame()
    if len(transfer):
        write_csv(transfer, ctx.report_dir / "zone_transfer.csv")

    parts = {e["label"]: e["rows"] for e in refits}
    shift, regions = shift_diagnostic(ctx, parts)
    write_csv(shift, ctx.report_dir / "shift_diagnostic.csv")
    if len(regions):
        write_csv(regions, ctx.report_dir / "region_shift.csv")

    ctx.assert_test_log_only_grew(before)
    ctx.write_config(seconds=round(time.perf_counter() - started, 1), refit_faithful=faithful,
                     n_test_evaluations=len(log_rows),
                     zone_transfer_tested=bool(len(transfer)),
                     test_log_sha256=file_hash(ctx.test_log))
    log.info("Version 0.7 reports written to %s", show_path(ctx.report_dir))
    log.info("finished in %.1f minutes", (time.perf_counter() - started) / 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="Version 0.7: measure generalisation across sites and time.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        set_seed(int(config["project"]["random_seed"]))
        ctx = Context(config, args.dataset)
        with keep_awake():
            run(ctx)
        return 0
    except (DataError, ConfigError, SplitError, EvaluationError, ModelError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
