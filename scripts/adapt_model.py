"""Version 0.8 production run: does local recalibration help at DRIAMS-C?

    python scripts/adapt_model.py

Executes the methodology locked in docs/v0.8_adaptive_plan.md and protocol amendment 4, whose hash is
recorded in config.yaml. Nothing here may be changed after a result has been seen.

Five models are scored once each on the frozen 30 % protected evaluation part of DRIAMS-C:

    B0  prevalence on the adaptation part            (no-skill reference, no fit)
    B1  the saved project model, unchanged           PRIMARY BASELINE, carried-over Version 0.6 zone edge
    B2  the Version 0.7 `external` refit             secondary baseline, carried-over edge
    A1  recalibration-only                           PRIMARY ARM: Platt (a, b) + zone edge refit on the
                                                     adaptation part; every tree frozen
    A2  refit on A-train + C-adaptation              confirmatory secondary arm, Version 0.4 setting

The decision threshold is NOT among the parameters the protocol lets an arm change (the plan enumerates
exactly "the two Platt parameters and the zone edge"), so every arm is evaluated at the deployed cut-off of
the saved model. Only the probability mapping and the zone edge differ between arms.

Primary endpoint: Brier score, paired difference B1 - arm. Co-primary: the zone pair, NPV >= target AND
coverage >= B1 coverage - slack. Holm across the two confirmatory arms. The verdict is arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # v08_analysis lives beside this script

from src.adaptation import (  # noqa: E402
    AdaptationError,
    ZoneCriterion,
    partition_fingerprint,
    recalibrate_only,
    uncalibrated_probabilities,
)
from src.data_loader import DataError  # noqa: E402
from src.dataset import load_dataset  # noqa: E402
from src.evaluate import (  # noqa: E402
    EvaluationError,
)
from src.predict import ModelError, load_bundle, predict_features  # noqa: E402
from src.splits import SplitError, load_splits  # noqa: E402
from src.train import load_rows  # noqa: E402
from src.tuning import FamilySpec, fit_calibrated, grouped_folds, set_threads  # noqa: E402
from src.uncertainty import ZoneRule, Zones, fit_zones  # noqa: E402
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

log = get_logger("adapt")

STAGE = "v0.8-adaptation"
REFERENCE_STAGE = "v0.8-reference"
METHODOLOGY_HASH = "e0ceb1727e63d7e3f6c9c72ecbe47795a61ac98d6b7f0a8dced3d1456f91cf60"


def versions() -> dict[str, str]:
    import joblib
    import lightgbm
    import sklearn

    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
            "joblib": joblib.__version__}


def section_hash(section: dict[str, Any], exclude: tuple[str, ...] = ()) -> str:
    payload = {k: v for k, v in section.items() if k not in exclude}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class Context:
    """Loads everything and refuses to continue unless the locked state is exactly what was approved."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.ad = config["adaptation"]
        self.ev = config["evaluation"]
        self.tc = config["tuning"]
        self.commit = git_commit()
        self.versions = versions()
        self.seed = int(self.ad["partition"]["seed"])

        # --- the methodology must be the approved one, byte for byte -----------------------------------
        got = section_hash(self.ad, exclude=("established",))
        if got != METHODOLOGY_HASH:
            raise ConfigError(f"The methodology hash is {got}, but the approved protocol is "
                              f"{METHODOLOGY_HASH}. The locked methodology has changed; refusing to run.")
        self.methodology_hash = got
        self.config_hash = section_hash(self.ad)
        log.info("methodology hash verified against the approved value (%s...)", got[:16])

        if self.ad["arms"]["hyperparameter_search_on_c"]:
            raise ConfigError("arms.hyperparameter_search_on_c is true; the protocol forbids a search on C.")
        if self.ad["arms"]["learning_curve_scored_on_holdout"]:
            raise ConfigError("arms.learning_curve_scored_on_holdout is true; the protocol forbids spending "
                              "the protected part on the learning curve.")
        # Stop condition 7. The protocol requires run_config.json to record a hash with no '-dirty' suffix,
        # so a dirty tree has to stop the run BEFORE anything is appended -- not be noticed afterwards in a
        # log line, which is how it was nearly missed once.
        if self.commit and str(self.commit).endswith("-dirty"):
            raise ConfigError(
                f"The working tree is dirty, so this run would record provenance {self.commit!r} and the "
                "results could not be tied to a committed state (stop condition 7). Commit first, then run. "
                "Nothing was fitted or scored.")

        # --- the C cohort and the frozen partition -----------------------------------------------------
        self.cohort = str(self.ad["cohort"]["name"])
        c_dir = project_path(config["dataset"]["output_dir"]) / self.cohort
        self.Xc, self.meta_c, self.summary_c = load_dataset(c_dir, verify_x=True)
        self.yc = self.meta_c["label"].to_numpy().astype(np.int64)
        self.groups_c = self.meta_c["group_id"].to_numpy()

        est = self.ad["established"]
        for key, got_v, want in (("rows fingerprint", self.summary_c["row_fingerprint"],
                                  str(est["cohort_rows_fingerprint"])),
                                 ("feature fingerprint", self.summary_c["feature_fingerprint"],
                                  str(self.ad["cohort"]["require_feature_fingerprint"]))):
            if got_v != want:
                raise ConfigError(f"The C cohort's {key} is {got_v}, but {want} was recorded. The dataset "
                                  "has changed; refusing to run.")

        frozen = np.load(project_path(self.ad["model_dir"]) / self.cohort / "partition_rows.npz")
        self.adapt_rows = np.asarray(frozen["adaptation"], dtype=np.int64)
        self.eval_rows = np.asarray(frozen["evaluation"], dtype=np.int64)
        self.partition = json.loads((project_path(self.ad["report_dir"]) / self.cohort
                                     / "partition.json").read_text(encoding="utf-8"))
        self._check_partition()

        # --- the models ---------------------------------------------------------------------------------
        b1_root = project_path(self.ad["baseline"]["model_dir"]) / config["dataset"]["name"]
        self.b1_path = b1_root / f"best_{self.ad['baseline']['split']}.joblib"
        self.bundle = load_bundle(self.b1_path, n_jobs=1)
        if self.bundle["feature_fingerprint"] != self.summary_c["feature_fingerprint"]:
            raise ModelError("The saved model's feature fingerprint does not match the C cohort's.")
        self.threshold = float(self.bundle["threshold"])
        self.pipeline = self.bundle["pipeline"]
        set_threads(self.pipeline, 1)

        self.rule = ZoneRule.from_config(config["explain"]["uncertainty"])
        edge_path = project_path(self.ad["baseline"]["zones"]) / config["dataset"]["name"] / "uncertainty.json"
        self.carried = Zones.from_dict(json.loads(edge_path.read_text(encoding="utf-8")))
        log.info("carried-over Version 0.6 zone edge: %.4f (fitted on the %s validation part, never refitted "
                 "for a baseline)", self.carried.lower, self.ad["baseline"]["split"])

        self.criterion = ZoneCriterion(float(self.ad["endpoints"]["zone_target_npv"]),
                                       float(self.ad["endpoints"]["zone_coverage_slack"]))

        self.report_dir = project_path(self.ad["report_dir"]) / self.cohort
        self.plot_dir = project_path(self.ad["plot_dir"])
        self.model_dir = project_path(self.ad["model_dir"]) / self.cohort
        for d in (self.report_dir, self.plot_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.test_log = project_path(self.ev["test_log"])
        self.test_log_hash_before = file_hash(self.test_log)

    def _check_partition(self) -> None:
        """The partition must be the frozen one, by fingerprint, and it must still be leakage-free."""
        fp = {"adaptation": partition_fingerprint(self.meta_c, self.adapt_rows),
              "evaluation": partition_fingerprint(self.meta_c, self.eval_rows)}
        for part, want in self.partition["partition_fingerprint"].items():
            if fp[part] != want:
                raise AdaptationError(f"The {part} partition fingerprint is {fp[part]}, but {want} was "
                                      "frozen. The partition has changed; refusing to run.")
        if np.intersect1d(self.adapt_rows, self.eval_rows).size:
            raise AdaptationError("The adaptation and evaluation parts share rows.")
        shared = np.intersect1d(np.unique(self.groups_c[self.adapt_rows]),
                                np.unique(self.groups_c[self.eval_rows]))
        if shared.size:
            raise AdaptationError(f"{shared.size} group(s) span the two parts.")
        if int(self.partition["seed"]) != self.seed:
            raise AdaptationError("The frozen partition's seed does not match the configuration.")
        log.info("frozen partition verified: adaptation %s (%d rows), evaluation %s (%d rows)",
                 fp["adaptation"], self.adapt_rows.size, fp["evaluation"], self.eval_rows.size)

    def features(self, rows: np.ndarray) -> np.ndarray:
        return load_rows(self.Xc, rows)


# --- the five arms ----------------------------------------------------------------------------------------

def zone_for(ctx: Context, prob_adapt: np.ndarray) -> tuple[Zones, dict[str, Any]]:
    """Refit the zone edge on the adaptation part only, by the unchanged Version 0.6 rule."""
    zones, low, high, _ = fit_zones(ctx.yc[ctx.adapt_rows], prob_adapt, ctx.threshold, ctx.rule)
    return zones, {"edge": zones.lower, "fitted_on": "C adaptation part",
                   "n_fitted": int(ctx.adapt_rows.size), "achieved_npv_on_adaptation": low.achieved,
                   "coverage_on_adaptation": low.coverage, "resistant_side_exists": high.exists}


def build_arms(ctx: Context) -> list[dict[str, Any]]:
    """Each arm's probabilities on the protected evaluation part, plus the zone it will be judged with."""
    X_eval = ctx.features(ctx.eval_rows)
    X_adapt = ctx.features(ctx.adapt_rows)
    y_adapt = ctx.yc[ctx.adapt_rows]
    arms: list[dict[str, Any]] = []

    # B0 prevalence: predicts the adaptation part's resistance rate for everyone.
    rate = float((y_adapt == 1).mean())
    arms.append({"key": "prevalence", "role": "B0", "stage": REFERENCE_STAGE,
                 "prob": np.full(ctx.eval_rows.size, rate, dtype=np.float64), "zones": None,
                 "threshold": rate, "model": "prevalence",
                 "setting": f"always predicts {rate:.4f} (the adaptation part's resistance rate)",
                 "train_size": int(ctx.adapt_rows.size), "detail": {"rate": rate}})

    # B1 the saved project model, unchanged, with the carried-over edge. The primary baseline.
    started = time.perf_counter()
    prob_b1 = predict_features(ctx.bundle, X_eval)[0]
    ms = (time.perf_counter() - started) * 1000 / max(ctx.eval_rows.size, 1)
    arms.append({"key": "baseline_saved", "role": "B1", "stage": REFERENCE_STAGE, "prob": prob_b1,
                 "zones": ctx.carried, "threshold": ctx.threshold,
                 "model": f"v0.4_{ctx.bundle['model']}", "seed": int(ctx.bundle["seed"]),
                 "train_size": int(ctx.bundle["split"]["train_size"]),
                 "setting": "saved model, unchanged; carried-over Version 0.6 zone edge",
                 "predict_ms_per_sample": round(ms, 4), "detail": {"model_version": ctx.bundle["model_version"]}})

    # B2 the Version 0.7 external refit (trained on DRIAMS-A only), carried-over edge.
    b2_path = (project_path(ctx.ad["baseline"]["secondary_model_dir"]) / "cache"
               / ctx.config["dataset"]["name"] / "external_seed42.joblib")
    if b2_path.is_file():
        import joblib

        stored = joblib.load(b2_path)
        model = stored["value"]["model"] if isinstance(stored, dict) and "value" in stored else stored
        set_threads(model, 1)
        arms.append({"key": "baseline_v07_external", "role": "B2", "stage": REFERENCE_STAGE,
                     "prob": np.asarray(model.predict_proba(X_eval)[:, 1], dtype=np.float64),
                     "zones": ctx.carried, "threshold": ctx.threshold,
                     "model": "v0.7_external_refit", "seed": 42, "train_size": 3831,
                     "setting": "Version 0.7 external refit (DRIAMS-A only); carried-over zone edge",
                     "detail": {"source": b2_path.name}})
    else:
        log.warning("%s is missing, so the secondary baseline B2 is left out", show_path(b2_path))

    # A1 recalibration-only. THE PRIMARY ARM. Every tree frozen; two parameters and the edge move.
    started = time.perf_counter()
    platt = recalibrate_only(ctx.pipeline, X_adapt, y_adapt)
    raw_adapt = uncalibrated_probabilities(ctx.pipeline, X_adapt)
    prob_a1_adapt = platt.apply(raw_adapt)
    zones_a1, edge_a1 = zone_for(ctx, prob_a1_adapt)
    prob_a1 = platt.apply(uncalibrated_probabilities(ctx.pipeline, X_eval))
    seconds = round(time.perf_counter() - started, 2)
    log.info("A1 recalibration-only: Platt a=%.6f b=%.6f fitted on %d adaptation rows (%d resistant); "
             "refitted zone edge %.4f (was %.4f)", platt.a, platt.b, platt.n_fitted, platt.n_resistant,
             zones_a1.lower if zones_a1.lower is not None else float("nan"), ctx.carried.lower)
    arms.append({"key": "recalibrated", "role": "A1", "stage": STAGE, "prob": prob_a1, "zones": zones_a1,
                 "threshold": ctx.threshold, "model": "recalibrated_v0.4_tuned_lightgbm",
                 "seed": int(ctx.bundle["seed"]), "train_size": int(ctx.adapt_rows.size),
                 "setting": "recalibration-only: Platt (a, b) and the zone edge refitted on the C "
                            "adaptation part; the LightGBM model is frozen",
                 "detail": {"platt": platt.to_dict(), "zone": edge_a1, "seconds": seconds}})

    # A2 refit on A-train + C-adaptation with the Version 0.4 setting unchanged. Confirmatory secondary.
    arms.append(refit_a_plus_c(ctx, X_eval, prob_reference=prob_a1))
    return arms


def refit_a_plus_c(ctx: Context, X_eval: np.ndarray, prob_reference: np.ndarray) -> dict[str, Any]:
    """The confirmatory arm: the same winning setting, refitted on DRIAMS-A plus the C adaptation part."""
    primary = ctx.config["dataset"]["name"]
    p_dir = project_path(ctx.config["dataset"]["output_dir"]) / primary
    Xa, meta_a, summary_a = load_dataset(p_dir, verify_x=False)
    splits = load_splits(p_dir / "splits", meta_a)
    a_rows = np.asarray(splits["external"].train, dtype=np.int64)
    ya = meta_a["label"].to_numpy().astype(np.int64)

    card_dir = project_path(ctx.tc["report_dir"]) / primary
    card = json.loads((card_dir / "best_model_card.json").read_text(encoding="utf-8"))
    setting = dict(card["params"])
    spec = next(FamilySpec.from_config(n, c) for n, c in ctx.tc["families"].items()
                if str(c.get("kind")) == str(card["model_kind"]))

    X_train = np.vstack([load_rows(Xa, a_rows), ctx.features(ctx.adapt_rows)])
    y_train = np.concatenate([ya[a_rows], ctx.yc[ctx.adapt_rows]])
    # Groups must be unique across the two cohorts, or a DRIAMS-A patient could be folded together with an
    # unrelated DRIAMS-C spectrum that happens to share a group number.
    groups = np.concatenate([np.array([f"A:{g}" for g in meta_a['group_id'].to_numpy()[a_rows]], dtype=object),
                             np.array([f"C:{g}" for g in ctx.groups_c[ctx.adapt_rows]], dtype=object)])
    folds = grouped_folds(y_train, groups, int(ctx.tc["cv_folds"]), int(ctx.tc["cv_seed"]))
    log.info("A2 refit: %d training rows (%d from DRIAMS-A, %d from the C adaptation part), %d resistant",
             y_train.size, a_rows.size, ctx.adapt_rows.size, int((y_train == 1).sum()))

    started = time.perf_counter()
    model = fit_calibrated(spec, setting, X_train, y_train, folds, int(card["seed"]),
                           method=str(ctx.tc["calibration"]))
    seconds = round(time.perf_counter() - started, 1)
    set_threads(model, 1)
    prob_adapt = np.asarray(model.predict_proba(ctx.features(ctx.adapt_rows))[:, 1], dtype=np.float64)
    zones_a2, edge_a2 = zone_for(ctx, prob_adapt)
    prob = np.asarray(model.predict_proba(X_eval)[:, 1], dtype=np.float64)
    log.info("A2 fitted in %.0f s; refitted zone edge %.4f", seconds,
             zones_a2.lower if zones_a2.lower is not None else float("nan"))
    return {"key": "refit_a_plus_c", "role": "A2", "stage": STAGE, "prob": prob, "zones": zones_a2,
            "threshold": ctx.threshold, "model": "refit_a_plus_c_tuned_lightgbm", "seed": int(card["seed"]),
            "train_size": int(y_train.size),
            "setting": "Version 0.4 winning setting refitted on DRIAMS-A + C adaptation; own calibrator and "
                       "zone edge; no search on C",
            "detail": {"setting": setting, "a_rows": int(a_rows.size),
                       "c_adaptation_rows": int(ctx.adapt_rows.size), "zone": edge_a2,
                       "fit_seconds": seconds}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Version 0.8 production run: adaptation at DRIAMS-C.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        set_seed(int(config["project"]["random_seed"]))
        ctx = Context(config)
        with keep_awake():
            from v08_analysis import run_analysis

            run_analysis(ctx, build_arms(ctx))
        return 0
    except (DataError, ConfigError, SplitError, EvaluationError, ModelError, AdaptationError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
