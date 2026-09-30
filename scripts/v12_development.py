"""Version 1.2 - development-only cross-validation of calibration and the cut-off (docs/v1.2_calibration_plan.md).

Run from the project root with the virtual environment active:

    python scripts/v12_development.py

**Development only.** It reads no test part: it derives the plan's development pool from the committed split
files, checks it, and runs every arm inside patient-grouped cross-validation folds of that pool. Nothing goes to
the production test log or near the served model; each run appends to the separate development log through the
strict pre-write gate. Every number it produces is exploratory (amendment 9, point 1).

Writes (paths from config.yaml -> v12_development):
- results/metrics/v1.2/<dataset>/*   reports and the generated tables.md (no identifiers)
- models/v1.2/<dataset>/heldout_predictions.npz   (git-ignored)
- results/experiments/development_runs.csv   one row per arm per partition, and per arm for the forward check
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import DataError  # noqa: E402
from src.dataset import load_dataset  # noqa: E402
from src.development import (  # noqa: E402
    DEV_LOG_COLUMNS,
    DevelopmentError,
    DevelopmentLog,
    assert_same_scale,
    cross_fitted_probabilities,
    development_pool,
    outer_folds,
    pool_fingerprint,
    spent_rows,
    stamp,
    supported_cutoff,
    validation_slice,
)
from src.evaluate import EvaluationError, calibration_slope_intercept, interval, summarize_bootstrap  # noqa: E402
from src.splits import SplitError, load_splits  # noqa: E402
from src.tables import md_table  # noqa: E402
from src.train import load_rows  # noqa: E402
from src.tuning import FamilySpec, TuningError, fit_calibrated, grouped_folds, set_threads  # noqa: E402
from src.utils import (  # noqa: E402
    ConfigError,
    get_logger,
    git_commit,
    keep_awake,
    load_config,
    project_path,
    show_path,
)

log = get_logger("v12")
ARMS = ("R0", "B", "C", "D1")
FITTED_ARMS = ("B", "C", "D1")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def section_hash(section: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(section, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class Ledger:
    """Every row a fold passes to a fit, a calibrator or a cut-off, checked against what it may never see."""

    def __init__(self, spent: np.ndarray) -> None:
        self.spent = np.asarray(spent, dtype=np.int64)            # rows of a spent test part: never, anywhere
        self.entries: list[dict[str, Any]] = []

    def use(self, what: str, rows: np.ndarray, held_out: np.ndarray, where: str) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        leaked = np.intersect1d(rows, held_out).size
        if leaked:
            raise DevelopmentError(f"{where}: {what} would see {leaked} held-out row(s).")
        if np.intersect1d(rows, self.spent).size:
            raise DevelopmentError(f"{where}: {what} would see a row of a spent test part.")
        self.entries.append({"where": where, "what": what, "rows": int(rows.size)})


class Run:
    """The configured study: data, pool, settings and provenance, all checked before anything is fitted."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config, self.vc, self.ev = config, config["v12_development"], config["evaluation"]
        vc = self.vc
        self.dataset = str(vc["dataset"])
        self.commit = git_commit() or "unknown"
        dirty = self.commit.endswith("-dirty") or self.commit == "unknown"
        if vc.get("require_clean_tree", True) and dirty:
            raise DevelopmentError(f"The working tree is not clean ({self.commit}); a development run records "
                                   "the exact code it ran, so commit first.")
        self.run_id = f"{self.commit}-{time.strftime('%Y%m%d%H%M%S')}" if dirty else self.commit
        plan = project_path(vc["plan"])
        if not plan.is_file():
            raise ConfigError(f"The plan {show_path(plan)} does not exist; no development run without it.")
        self.plan_sha256 = sha256_file(plan)
        self.config_sha256 = section_hash(vc)

        data_dir = project_path(config["dataset"]["output_dir"]) / self.dataset
        self.X, self.meta, self.summary = load_dataset(data_dir, verify_x=True)
        self.splits = load_splits(data_dir / "splits", self.meta)
        self.y = self.meta["label"].to_numpy().astype(np.int64)
        self.groups = self.meta["group_id"].to_numpy()
        self.dates = pd.to_datetime(self.meta["acquisition_date"]).to_numpy()
        spent = list(vc["spent_splits"])
        self.spent = spent_rows(self.splits, spent)
        self.pool = development_pool(self.meta, self.splits, str(vc["pool_source_split"]), spent)
        self.pool_fp = pool_fingerprint(self.meta, self.pool)
        found = (int(self.pool.size), int(self.y[self.pool].sum()))
        expected = vc.get("expected_pool")
        if expected and found != (int(expected["spectra"]), int(expected["resistant"])):
            raise DevelopmentError(f"The development pool holds {found[0]} spectra, {found[1]} resistant; the "
                                   f"plan's audit found {expected['spectra']} / {expected['resistant']}.")
        self.ledger = Ledger(self.spent)

        self.settings: dict[str, tuple[FamilySpec, dict[str, Any], str]] = {}
        for name, s in vc["settings"].items():
            card_path = project_path(s["card"])
            card = json.loads(card_path.read_text(encoding="utf-8"))
            spec = FamilySpec.from_config(s["family"], config[s["section"]]["families"][s["family"]])
            if card["model_kind"] != spec.kind:
                raise ConfigError(f"Setting {name}: the card is a {card['model_kind']}, the family a {spec.kind}.")
            self.settings[name] = (spec, dict(card["params"]), sha256_file(card_path))
        self.arms = dict(vc["arms"])
        th = vc["threshold"]
        self.rule = {"rule": th["rule"], "min_sensitivity": float(th["min_sensitivity"])}
        self.min_support = int(th["min_resistant_for_support"])
        self.report_dir = project_path(vc["report_dir"]) / self.dataset
        self.model_dir = project_path(vc["model_dir"]) / self.dataset
        for folder in (self.report_dir, self.model_dir):
            folder.mkdir(parents=True, exist_ok=True)
        production = project_path(self.ev["test_log"])
        self.dev_log = DevelopmentLog(project_path(vc["development_log"]), production)
        self.production_log_sha = sha256_file(production) if production.is_file() else None
        self.started = time.monotonic()

    def budget(self) -> None:
        if time.monotonic() - self.started > float(self.vc["budget_seconds"]):
            raise DevelopmentError(f"The compute budget of {self.vc['budget_seconds']} s was exceeded (the plan's "
                                   "section 10); the run is recorded as failed.")

    def fit_arms(self, train: np.ndarray, held_out: np.ndarray, where: str, slice_seed: int) -> dict[str, Any]:
        """R0, B, C and D1 fitted on `train` only; each arm's cut-off chosen on rows inside `train` only."""
        vc, y, groups = self.vc, self.y, self.groups
        model_seed, inner_k, inner_seed = int(vc["model_seed"]), int(vc["inner_folds"]), int(vc["inner_seed"])
        method = str(vc["calibration"])
        X_tr, y_tr, g_tr = load_rows(self.X, train), y[train], groups[train]
        rate = float(y_tr.mean())
        out: dict[str, Any] = {"R0": {"model": None, "rate": rate, "fit_rows": int(train.size), "fit_seconds": 0.0,
                                      "cutoff": {"threshold": rate, "supported": False,
                                                 "selection_n": int(train.size),
                                                 "selection_resistant": int(y_tr.sum())}}}
        # B, the unchanged Version 1.1 procedure: fit on 7/8, cut-off on a patient-grouped 1/8 validation slice
        spec, setting, _ = self.settings[self.arms["B"]]
        fit_pos, val_pos = validation_slice(y_tr, g_tr, int(vc["validation_slice_denominator"]), slice_seed)
        self.ledger.use("B fit and calibration", train[fit_pos], held_out, where)
        started = time.perf_counter()
        inner = grouped_folds(y_tr[fit_pos], g_tr[fit_pos], inner_k, inner_seed)
        model = fit_calibrated(spec, setting, X_tr[fit_pos], y_tr[fit_pos], inner, model_seed, method)
        set_threads(model, 1)
        self.ledger.use("B cut-off", train[val_pos], held_out, where)
        cutoff = supported_cutoff(y_tr[val_pos], model.predict_proba(X_tr[val_pos])[:, 1], self.rule,
                                  self.min_support)
        out["B"] = {"model": model, "cutoff": cutoff, "fit_rows": int(fit_pos.size),
                    "fit_seconds": round(time.perf_counter() - started, 1)}
        # C and D1: fit on the whole outer-training part; cut-off on its cross-fitted predictions
        for arm in ("C", "D1"):
            spec, setting, _ = self.settings[self.arms[arm]]
            inner = grouped_folds(y_tr, g_tr, inner_k, inner_seed)
            self.ledger.use(f"{arm} fit and calibration", train, held_out, where)
            started = time.perf_counter()
            model = fit_calibrated(spec, setting, X_tr, y_tr, inner, model_seed, method)
            crossfit = cross_fitted_probabilities(model, spec, setting, X_tr, y_tr, inner, model_seed)
            set_threads(model, 1)
            assert_same_scale(model, X_tr[: min(200, train.size)])
            self.ledger.use(f"{arm} cut-off", train, held_out, where)
            out[arm] = {"model": model, "cutoff": supported_cutoff(y_tr, crossfit, self.rule, self.min_support),
                        "fit_rows": int(train.size), "fit_seconds": round(time.perf_counter() - started, 1)}
        return out

    def predict(self, fitted: dict[str, Any], rows: np.ndarray) -> dict[str, np.ndarray]:
        X = load_rows(self.X, rows)
        probs = {"R0": np.full(rows.size, fitted["R0"]["rate"], dtype=np.float64)}
        for arm in FITTED_ARMS:
            p = fitted[arm]["model"].predict_proba(X)[:, 1].astype(np.float64)
            if not np.isfinite(p).all():
                raise DevelopmentError(f"Arm {arm} produced a non-finite probability.")
            probs[arm] = p
        return probs


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    """Held-out metrics of one arm; R0's constant prediction gets AUROC 0.5 and no calibration slope."""
    decided = p >= threshold
    positives, negatives = int(y.sum()), int((y == 0).sum())
    varied = np.ptp(p) > 0
    slope, intercept = calibration_slope_intercept(y, p) if varied else (float("nan"), float("nan"))
    return {"n": int(y.size), "n_resistant": positives, "brier": float(brier_score_loss(y, p)),
            "roc_auc": float(roc_auc_score(y, p)) if varied else 0.5,
            "pr_auc": float(average_precision_score(y, p)) if varied else positives / y.size,
            "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
            "calibration_slope": float(slope), "calibration_intercept": float(intercept),
            "delivered_sensitivity": float((decided & (y == 1)).sum() / positives),
            "delivered_specificity": float((~decided & (y == 0)).sum() / negatives)}


def paired_analysis(run: Run, rows: np.ndarray, probs: dict[str, np.ndarray],
                    decisions: dict[str, np.ndarray]) -> dict[str, Any]:
    """Paired patient-group bootstrap over held-out predictions; the reference is B, the comparator."""
    b = run.ev["bootstrap"]
    level = float(b["level"])
    y, g = run.y[rows], run.groups[rows]
    kw = {"resamples": int(b["resamples"]), "level": level, "seed": int(b["seed"]), "reference": "B"}
    scores, samples = summarize_bootstrap(y, g, probs, dict.fromkeys(probs, 0.5),
                                          metrics=("brier", "roc_auc", "pr_auc"), **kw)
    decided, _ = summarize_bootstrap(y, g, {a: d.astype(np.float64) for a, d in decisions.items()},
                                     dict.fromkeys(decisions, 0.5), metrics=("sensitivity", "specificity"), **kw)
    points = scores["intervals"]
    skill = {}
    for arm in FITTED_ARMS:
        low, high = interval(1 - samples[arm]["brier"] / samples["R0"]["brier"], level)
        skill[arm] = {"estimate": 1 - points[arm]["brier"]["estimate"] / points["R0"]["brier"]["estimate"],
                      "low": low, "high": high}
    c_minus_d1 = {}
    for m in ("brier", "roc_auc"):
        low, high = interval(samples["C"][m] - samples["D1"][m], level)
        c_minus_d1[m] = {"estimate": points["C"][m]["estimate"] - points["D1"][m]["estimate"], "low": low, "high": high}
    return {"scores": scores, "decisions": decided, "brier_skill_vs_R0": skill, "C_minus_D1": c_minus_d1}


def verdict(low: float, high: float) -> str:
    """The plan's section 8, for B minus C on Brier: positive favours C."""
    if low > 0:
        return "C better"
    if high < 0:
        return "C worse"
    return "not demonstrated"


def stability_table(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        f = folds[folds["arm"] == arm]
        s = f["delivered_sensitivity"]
        rows.append({"arm": arm, "folds": len(f), "supported_cutoffs": int(f["supported"].sum()),
                     "sensitivity_mean": s.mean(), "sensitivity_sd": s.std(ddof=1), "sensitivity_min": s.min(),
                     "sensitivity_max": s.max(), "folds_at_or_above_0.90": int((s >= 0.90).sum()),
                     "folds_at_or_above_0.85": int((s >= 0.85).sum()),
                     "specificity_mean": f["delivered_specificity"].mean(),
                     "specificity_sd": f["delivered_specificity"].std(ddof=1),
                     "threshold_mean": f["threshold"].mean(), "threshold_sd": f["threshold"].std(ddof=1),
                     "selection_resistant_mean": f["selection_resistant"].mean()})
    return pd.DataFrame(rows).set_index("arm")


def mean_sd(stab: pd.DataFrame, arm: str, what: str) -> str:
    return f"{stab.loc[arm, what + '_mean']:.3f} ({stab.loc[arm, what + '_sd']:.3f})"


def fmt(d: dict[str, float], signed: bool = False) -> str:
    if signed:
        return f"{d['estimate']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]"
    return f"{d['estimate']:.4f} ({d['low']:.4f}–{d['high']:.4f})"


def tables(run: Run, primary: dict[str, Any], pooled: dict[int, dict[str, Any]], stab: pd.DataFrame,
           forward_rows: list[dict[str, Any]], forward: dict[str, Any]) -> str:
    seed = int(run.vc["primary_partition_seed"])
    main = pooled[seed]
    iv, de = main["scores"]["intervals"], main["decisions"]["intervals"]
    labels = {"R0": "R0 no-skill (training resistance rate)", "B": "B unchanged Version 1.1 procedure",
              "C": "C candidate (Version 0.4 setting, cross-fitted cut-off)",
              "D1": "D1 diagnostic (B's setting, C's procedure)"}
    out = [f"# Version 1.2 development tables (`{run.dataset}`) — exploratory",
           "",
           "Generated by `scripts/v12_development.py`. **Development only; every number here is exploratory.** No "
           "test part was read; the pool is the plan's (`docs/v1.2_calibration_plan.md`, amendment 9): "
           f"{run.pool.size:,} spectra, {int(run.y[run.pool].sum())} resistant, fingerprint `{run.pool_fp}`. "
           f"Run `{run.run_id}`, plan SHA-256 `{run.plan_sha256[:16]}…`. Research only; no clinical claim.",
           "",
           "## Primary endpoint (exploratory)",
           "",
           f"Brier score, B minus C, pooled held-out predictions of partition seed {seed} (positive favours C): "
           f"**{fmt(primary, signed=True)}** — **{primary['verdict']}**. An interval containing 0 is \"not "
           "demonstrated\", never \"equivalent\".",
           "",
           "Partitions 43 and 44 (sensitivity analysis): "
           + "; ".join(f"seed {s}: {fmt(r, signed=True)}" for s, r in primary["repeats"].items() if s != str(seed))
           + ".",
           "",
           f"## Pooled held-out predictions, partition seed {seed}",
           "",
           md_table([{"Arm": labels[a], "Brier": fmt(iv[a]["brier"]), "AUROC": fmt(iv[a]["roc_auc"]),
                      "PR-AUC": fmt(iv[a]["pr_auc"]),
                      "Brier skill vs R0": fmt(main["brier_skill_vs_R0"][a]) if a in FITTED_ARMS else "—",
                      "Delivered sensitivity": fmt(de[a]["sensitivity"]),
                      "Delivered specificity": fmt(de[a]["specificity"])} for a in ARMS]),
           "",
           "Each spectrum's decision uses the cut-off its own fold chose. Intervals: 2,000 patient-group bootstrap "
           "resamples, which treat each fold's fitted model as fixed.",
           "",
           f"Diagnostics (paired): D1 against B isolates the procedure, B minus D1 Brier "
           f"{fmt(main['scores']['differences_to_reference']['D1']['brier'], signed=True)}; C against D1 isolates the "
           f"setting, C minus D1 Brier {fmt(main['C_minus_D1']['brier'], signed=True)}, AUROC "
           f"{fmt(main['C_minus_D1']['roc_auc'], signed=True)}.",
           "",
           "## Operating-point stability over all held-out folds",
           "",
           md_table([{"Arm": a, "Folds": int(stab.loc[a, "folds"]),
                      "Cut-offs supported (≥ 50 resistant)": int(stab.loc[a, "supported_cutoffs"]),
                      "Resistant in selection set (mean)": f"{stab.loc[a, 'selection_resistant_mean']:.0f}",
                      "Sensitivity mean (SD)": mean_sd(stab, a, "sensitivity"),
                      "Range": f"{stab.loc[a, 'sensitivity_min']:.3f}–{stab.loc[a, 'sensitivity_max']:.3f}",
                      "Folds ≥ 0.90 / ≥ 0.85": f"{int(stab.loc[a, 'folds_at_or_above_0.90'])} / "
                                               f"{int(stab.loc[a, 'folds_at_or_above_0.85'])}",
                      "Specificity mean (SD)": mean_sd(stab, a, "specificity")}
                     for a in FITTED_ARMS]),
           "",
           f"\"C more stable than B\" (the plan's descriptive rule: smaller SD **and** mean closer to 0.90): "
           f"**{'met' if primary['c_more_stable_descriptive'] else 'not met'}**. A cut-off whose selection set held "
           "fewer than 50 resistant spectra does not support its 0.90 target.",
           "",
           "## Forward in time (fit before 2017, evaluate on 2017; exploratory)",
           "",
           md_table([{"Arm": r["arm"], "Train n / resistant": f"{r['train_n']} / {r['train_resistant']}",
                      "Evaluated n / resistant": f"{r['n']} / {r['n_resistant']}",
                      "Brier": fmt(forward["scores"]["intervals"][r["arm"]]["brier"]),
                      "AUROC": fmt(forward["scores"]["intervals"][r["arm"]]["roc_auc"]),
                      "Cut-off supported": "yes" if r["supported"] else "no",
                      "Delivered sensitivity": fmt(forward["decisions"]["intervals"][r["arm"]]["sensitivity"]),
                      "Delivered specificity": fmt(forward["decisions"]["intervals"][r["arm"]]["specificity"])}
                     for r in forward_rows]),
           "",
           "Forward-in-time Brier, B minus C: "
           f"{fmt(forward['scores']['differences_to_reference']['C']['brier'], signed=True)}. "
           "Patients cannot be linked across years, so this part is date-separated, not patient-separated.",
           ""]
    return "\n".join(out)


def failure_row(run: Run, exc: Exception) -> dict[str, Any]:
    return dict.fromkeys(DEV_LOG_COLUMNS, "") | {
        "logged_at": stamp(), "git_commit": run.commit, "run_id": run.run_id, "plan_sha256": run.plan_sha256,
        "config_sha256": run.config_sha256, "dataset": run.dataset, "experiment": f"v1.2/{run.run_id}/failed",
        "model": "run", "seed": 0, "status": f"failed: {exc}"[:500]}


def study(run: Run) -> dict[str, Any]:
    vc = run.vc
    fold_rows: list[dict[str, Any]] = []
    pooled: dict[int, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {"pool_rows": run.pool}
    y_pool, g_pool = run.y[run.pool], run.groups[run.pool]
    for partition in [int(s) for s in vc["partition_seeds"]]:
        folds = outer_folds(y_pool, g_pool, int(vc["outer_folds"]), partition,
                            int(vc["min_resistant_per_heldout_fold"]))
        probs = {a: np.full(run.pool.size, np.nan) for a in ARMS}
        decisions = {a: np.zeros(run.pool.size, dtype=bool) for a in ARMS}
        fold_id = np.full(run.pool.size, -1)
        for k, (tr_pos, te_pos) in enumerate(folds):
            train, held_out = run.pool[tr_pos], run.pool[te_pos]
            where = f"partition {partition} fold {k}"
            fitted = run.fit_arms(train, held_out, where, partition)
            predicted = run.predict(fitted, held_out)
            for arm in ARMS:
                cut = fitted[arm]["cutoff"]
                probs[arm][te_pos] = predicted[arm]
                decisions[arm][te_pos] = predicted[arm] >= cut["threshold"]
                fold_rows.append({"partition": partition, "fold": k, "arm": arm, **cut,
                                  "fit_rows": fitted[arm]["fit_rows"], "fit_seconds": fitted[arm]["fit_seconds"],
                                  **metrics(run.y[held_out], predicted[arm], cut["threshold"])})
            fold_id[te_pos] = k
            log.info("%s: Brier B %.4f, C %.4f, D1 %.4f; cut-off supported B %s, C %s", where,
                     *(r["brier"] for r in fold_rows[-3:]), fitted["B"]["cutoff"]["supported"],
                     fitted["C"]["cutoff"]["supported"])
            run.budget()
        if (fold_id < 0).any() or any(np.isnan(p).any() for p in probs.values()):
            raise DevelopmentError(f"Partition {partition}: not every pool spectrum was held out exactly once.")
        pooled[partition] = paired_analysis(run, run.pool, probs, decisions)
        arrays[f"p{partition}__fold"] = fold_id
        for arm in ARMS:
            arrays[f"p{partition}__{arm}"], arrays[f"p{partition}__{arm}__decision"] = probs[arm], decisions[arm]

    fw = vc["forward"]
    dates = run.dates[run.pool]
    train = run.pool[dates < np.datetime64(fw["train_before"])]
    later = run.pool[(dates >= np.datetime64(fw["evaluate_from"])) & (dates < np.datetime64(fw["evaluate_before"]))]
    if int(run.y[later].sum()) < int(fw["min_resistant"]):
        raise DevelopmentError(f"The forward part holds {int(run.y[later].sum())} resistant, fewer than "
                               f"{fw['min_resistant']}.")
    if np.intersect1d(np.unique(run.groups[train]), np.unique(run.groups[later])).size:
        raise DevelopmentError("The forward-in-time parts share a patient group.")
    fitted = run.fit_arms(train, later, "forward in time", int(vc["primary_partition_seed"]))
    predicted = run.predict(fitted, later)
    forward_rows = [{"arm": a, "train_n": int(train.size), "train_resistant": int(run.y[train].sum()),
                     **fitted[a]["cutoff"], **metrics(run.y[later], predicted[a], fitted[a]["cutoff"]["threshold"])}
                    for a in ARMS]
    forward = paired_analysis(run, later, predicted,
                              {a: predicted[a] >= fitted[a]["cutoff"]["threshold"] for a in ARMS})
    arrays["forward_rows"] = later
    for arm in ARMS:
        arrays[f"forward__{arm}"] = predicted[arm]
    return {"folds": pd.DataFrame(fold_rows), "pooled": pooled, "arrays": arrays,
            "forward_rows": forward_rows, "forward": forward}


def report(run: Run, result: dict[str, Any]) -> dict[str, Any]:
    vc = run.vc
    seed = int(vc["primary_partition_seed"])
    folds, pooled, arrays = result["folds"], result["pooled"], result["arrays"]
    diff = pooled[seed]["scores"]["differences_to_reference"]["C"]["brier"]      # B minus C
    stab = stability_table(folds)
    closer = abs(stab.loc["C", "sensitivity_mean"] - 0.90) < abs(stab.loc["B", "sensitivity_mean"] - 0.90)
    c_more_stable = bool(stab.loc["C", "sensitivity_sd"] < stab.loc["B", "sensitivity_sd"] and closer)
    provenance = {"run_id": run.run_id, "git_commit": run.commit, "plan": vc["plan"], "plan_sha256": run.plan_sha256,
                  "config_section_sha256": run.config_sha256, "dataset": run.dataset,
                  "dataset_row_fingerprint": run.summary["row_fingerprint"], "x_sha256": run.summary["x_sha256"],
                  "pool_fingerprint": run.pool_fp, "pool_spectra": int(run.pool.size),
                  "pool_resistant": int(run.y[run.pool].sum()),
                  "settings": {k: {"card": vc["settings"][k]["card"], "card_sha256": v[2], "params": v[1]}
                               for k, v in run.settings.items()},
                  "production_log_sha256_at_start": run.production_log_sha, "ledger_entries": len(run.ledger.entries),
                  "seconds": round(time.monotonic() - run.started, 1), "finished": stamp()}
    primary = {"endpoint": f"Brier score, B minus C, pooled held-out predictions, partition seed {seed}",
               "estimate": diff["estimate"], "low": diff["low"], "high": diff["high"],
               "verdict": verdict(diff["low"], diff["high"]), "exploratory": True,
               "repeats": {str(s): {k: r["scores"]["differences_to_reference"]["C"]["brier"][k]
                                    for k in ("estimate", "low", "high")} for s, r in pooled.items()},
               "c_more_stable_descriptive": c_more_stable, "provenance": provenance}
    folds.to_csv(run.report_dir / "fold_results.csv", index=False, lineterminator="\n")
    stab.reset_index().to_csv(run.report_dir / "stability.csv", index=False, lineterminator="\n")
    pd.DataFrame(result["forward_rows"]).to_csv(run.report_dir / "forward_check.csv", index=False, lineterminator="\n")
    (run.report_dir / "pooled_results.json").write_text(
        json.dumps({str(s): r for s, r in pooled.items()} | {"forward": result["forward"]}, indent=2, default=float),
        encoding="utf-8", newline="\n")
    np.savez_compressed(run.model_dir / "heldout_predictions.npz", **arrays)

    base = {"logged_at": stamp(), "git_commit": run.commit, "run_id": run.run_id, "plan_sha256": run.plan_sha256,
            "config_sha256": run.config_sha256, "dataset": run.dataset,
            "dataset_fingerprint": run.summary["row_fingerprint"], "pool_fingerprint": run.pool_fp, "status": "ok"}
    rows = []
    y_pool = run.y[run.pool]
    for s, r in pooled.items():
        f = folds[folds["partition"] == s]
        for arm in ARMS:
            p = arrays[f"p{s}__{arm}"]
            m = metrics(y_pool, p, 0.5)                   # threshold-free parts only; decisions come below
            de = r["decisions"]["intervals"][arm]
            rows.append({**base, "experiment": f"v1.2/{run.run_id}/cv", "model": arm, "seed": int(s),
                         **{k: m[k] for k in ("n", "n_resistant", "brier", "roc_auc", "pr_auc", "log_loss",
                                              "calibration_slope", "calibration_intercept")},
                         "delivered_sensitivity": de["sensitivity"]["estimate"],
                         "delivered_specificity": de["specificity"]["estimate"],
                         "supported_cutoffs": int(f[f["arm"] == arm]["supported"].sum())})
    for r in result["forward_rows"]:
        rows.append({**base, "experiment": f"v1.2/{run.run_id}/forward", "model": r["arm"], "seed": seed,
                     **{k: r[k] for k in ("n", "n_resistant", "brier", "roc_auc", "pr_auc", "log_loss",
                                          "calibration_slope", "calibration_intercept", "delivered_sensitivity",
                                          "delivered_specificity")},
                     "supported_cutoffs": int(r["supported"])})
    primary["provenance"]["development_log_pre_write"] = run.dev_log.append(rows)
    (run.report_dir / "primary_result.json").write_text(json.dumps(primary, indent=2, default=float),
                                                      encoding="utf-8", newline="\n")
    (run.report_dir / "tables.md").write_text(
        tables(run, primary, pooled, stab, result["forward_rows"], result["forward"]), encoding="utf-8", newline="\n")
    production = project_path(run.ev["test_log"])
    if run.production_log_sha and sha256_file(production) != run.production_log_sha:
        raise DevelopmentError("The production test log changed during a development run.")
    return primary


def main() -> int:
    parser = argparse.ArgumentParser(description="Version 1.2: development-only calibration and cut-off study.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    run = None
    try:
        run = Run(load_config(args.config))
        log.info("pool %d spectra (%d resistant, %d patient groups), fingerprint %s; run %s", run.pool.size,
                 int(run.y[run.pool].sum()), np.unique(run.groups[run.pool]).size, run.pool_fp, run.run_id)
        with keep_awake():
            primary = report(run, study(run))
        log.info("primary (exploratory): Brier B minus C %.5f [%.5f, %.5f] -> %s", primary["estimate"],
                 primary["low"], primary["high"], primary["verdict"])
        log.info("reports in %s; development log %s", show_path(run.report_dir), show_path(run.dev_log.path))
        return 0
    except (DataError, ConfigError, SplitError, DevelopmentError, EvaluationError, TuningError) as exc:
        log.error("%s", exc)
        if run is not None:                            # a failed run is recorded, never reported as a result
            try:
                run.dev_log.append([failure_row(run, exc)])
            except (DevelopmentError, EvaluationError, OSError) as inner:
                log.error("the failure could not be recorded in the development log: %s", inner)
        return 1


if __name__ == "__main__":
    sys.exit(main())
