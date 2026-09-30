"""Version 1.3 - does an uncertainty-aware cut-off rule hold the 90 % target in later periods?

Fixed by docs/v1.3_threshold_plan.md (protocol amendment 10) before this file existed.

Run from the project root with the virtual environment active:

    python scripts/v13_threshold.py

**Development only, and exploratory.** It reads no test part and never opens DRIAMS-C. It uses the Version 1.2
development pool, and at each of two rolling origins it fits the fixed model and calibration on the spectra
labelled at least 7 days before the origin, chooses rule E's and rule U's cut-offs on one cross-fitted
prediction per patient group of those same spectra, and measures what each cut-off delivers in the later
period. A ledger records every row that each fit, cut-off and recalibration sees, and the run stops if one is
from the later period, from the label-availability gap, or from a spent test part.

Writes (paths from config.yaml -> v13_threshold):
- results/metrics/v1.3/<dataset>/*   reports and the generated tables.md (no identifiers)
- models/v1.3/<dataset>/later_predictions.npz   (git-ignored)
- results/experiments/development_runs.csv   development-log rows through the strict gate
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
    pool_fingerprint,
    spent_rows,
    stamp,
)
from src.evaluate import (  # noqa: E402
    EvaluationError,
    calibration_slope_intercept,
    group_resample_indices,
    interval,
    summarize_bootstrap,
)
from src.splits import SplitError, load_splits  # noqa: E402
from src.tables import md_table  # noqa: E402
from src.threshold_rules import (  # noqa: E402
    ThresholdRuleError,
    apply_shift,
    assert_available,
    delivered,
    drop_seen_patients,
    empirical_cutoff,
    intercept_shift,
    minimum_feasible_n,
    one_per_group,
    origin_windows,
    tolerance_cutoff,
)
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

log = get_logger("v13")
RULES = ("E", "U")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Ledger:
    """Every row a fit, a cut-off or a recalibration sees, checked against time, the later period and spent rows."""

    def __init__(self, spent: np.ndarray, dates: np.ndarray, gap_days: int) -> None:
        self.spent, self.dates, self.gap_days = np.asarray(spent, dtype=np.int64), dates, gap_days
        self.entries: list[dict[str, Any]] = []

    def use(self, what: str, rows: np.ndarray, later: np.ndarray, origin: str) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        if np.intersect1d(rows, later).size:
            leaked = np.intersect1d(rows, later).size
            raise DevelopmentError(f"{origin}: {what} would see {leaked} later-period row(s).")
        if np.intersect1d(rows, self.spent).size:
            raise DevelopmentError(f"{origin}: {what} would see a row of a spent test part.")
        assert_available(rows, self.dates, origin, self.gap_days, f"{origin}: {what}")
        self.entries.append({"origin": origin, "what": what, "rows": int(rows.size)})


class Study:
    """The configured study; every provenance and design check runs before anything is fitted."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config, self.tc, self.ev = config, config["v13_threshold"], config["evaluation"]
        tc = self.tc
        self.dataset = str(tc["dataset"])
        self.commit = git_commit() or "unknown"
        dirty = self.commit.endswith("-dirty") or self.commit == "unknown"
        if tc.get("require_clean_tree", True) and dirty:
            raise DevelopmentError(f"The working tree is not clean ({self.commit}); commit before a development run.")
        self.run_id = f"{self.commit}-{time.strftime('%Y%m%d%H%M%S')}" if dirty else self.commit
        plan = project_path(tc["plan"])
        if not plan.is_file():
            raise ConfigError(f"The plan {show_path(plan)} does not exist; no Version 1.3 run without it.")
        self.plan_sha256 = sha256_file(plan)
        self.config_sha256 = hashlib.sha256(json.dumps(tc, sort_keys=True, default=str).encode()).hexdigest()
        self.target, self.confidence = float(tc["target_sensitivity"]), float(tc["confidence"])
        self.gap_days = int(tc["gap_days"])

        data_dir = project_path(config["dataset"]["output_dir"]) / self.dataset
        self.X, self.meta, self.summary = load_dataset(data_dir, verify_x=True)
        splits = load_splits(data_dir / "splits", self.meta)
        self.y = self.meta["label"].to_numpy().astype(np.int64)
        self.groups = self.meta["group_id"].to_numpy()
        self.dates = pd.to_datetime(self.meta["acquisition_date"]).to_numpy()
        spent = spent_rows(splits, list(tc["spent_splits"]))
        self.pool = development_pool(self.meta, splits, str(tc["pool_source_split"]), list(tc["spent_splits"]))
        self.pool_fp = pool_fingerprint(self.meta, self.pool)
        expected = tc.get("expected_pool")
        if expected and (int(self.pool.size), int(self.y[self.pool].sum()), self.pool_fp) != (
                int(expected["spectra"]), int(expected["resistant"]), str(expected["fingerprint"])):
            raise DevelopmentError(f"The pool is {self.pool.size} / {int(self.y[self.pool].sum())} ({self.pool_fp}); "
                                   f"the plan's is {expected['spectra']} / {expected['resistant']} "
                                   f"({expected['fingerprint']}).")
        self.ledger = Ledger(spent, self.dates, self.gap_days)
        s = tc["setting"]
        card_path = project_path(s["card"])
        card = json.loads(card_path.read_text(encoding="utf-8"))
        self.spec = FamilySpec.from_config(s["family"], config[s["section"]]["families"][s["family"]])
        self.setting, self.card_sha256 = dict(card["params"]), sha256_file(card_path)
        if card["model_kind"] != self.spec.kind:
            raise ConfigError(f"The card is a {card['model_kind']}, the family a {self.spec.kind}.")
        self.report_dir = project_path(tc["report_dir"]) / self.dataset
        self.model_dir = project_path(tc["model_dir"]) / self.dataset
        for folder in (self.report_dir, self.model_dir):
            folder.mkdir(parents=True, exist_ok=True)
        production = project_path(self.ev["test_log"])
        self.dev_log = DevelopmentLog(project_path(tc["development_log"]), production)
        self.production_log_sha = sha256_file(production) if production.is_file() else None
        self.started = time.monotonic()

    def check(self, name: str, found: dict[str, int], expected: dict[str, int] | None) -> None:
        """The run's counts must equal the plan's audit (section 9); otherwise it stops."""
        if not expected:
            return
        wrong = {k: (found[k], v) for k, v in expected.items() if found.get(k) != v}
        if wrong:
            raise DevelopmentError(f"{name}: counts differ from the plan's audit (found, expected): {wrong}.")

    def origin(self, o: dict[str, Any]) -> dict[str, Any]:
        tc, y, groups, dates = self.tc, self.y, self.groups, self.dates
        name, origin = str(o["name"]), str(o["origin"])
        w = origin_windows(dates, self.pool, origin, str(o["end"]), self.gap_days)
        available, gap = w["available"], w["gap"]
        later, removed = drop_seen_patients(w["later"], np.concatenate([available, gap]), groups)
        selection = one_per_group(available, dates, groups)
        recent = available[dates[available] >= np.datetime64(pd.Timestamp(o["recent_from"]))]
        found = {"available": int(available.size), "available_resistant": int(y[available].sum()),
                 "selection_resistant": int(y[selection].sum()), "gap": int(gap.size), "later": int(later.size),
                 "removed_seen": removed, "later_resistant": int(y[later].sum()),
                 "recent_resistant": int(y[recent].sum()),
                 "recent_resistant_groups": int(np.unique(groups[recent][y[recent] == 1]).size)}
        found["k_star"] = tolerance_cutoff(np.zeros(found["selection_resistant"]), self.target, self.confidence)["k"]
        self.check(name, found, o.get("expected"))

        self.ledger.use("fit and calibration", available, later, origin)
        X_fit, y_fit = load_rows(self.X, available), y[available]
        inner = grouped_folds(y_fit, groups[available], int(tc["inner_folds"]), int(tc["inner_seed"]))
        started = time.perf_counter()
        seed = int(tc["model_seed"])
        model = fit_calibrated(self.spec, self.setting, X_fit, y_fit, inner, seed, str(tc["calibration"]))
        crossfit = cross_fitted_probabilities(model, self.spec, self.setting, X_fit, y_fit, inner, seed)
        set_threads(model, 1)
        assert_same_scale(model, X_fit[: min(200, available.size)])
        fit_seconds = round(time.perf_counter() - started, 1)
        position = {int(r): i for i, r in enumerate(available)}

        self.ledger.use("cut-off selection", selection, later, origin)
        sel_scores = crossfit[[position[int(r)] for r in selection]]
        resistant_scores = sel_scores[y[selection] == 1]
        cutoffs = {"E": empirical_cutoff(resistant_scores, self.target),
                   "U": tolerance_cutoff(resistant_scores, self.target, self.confidence)}

        rc = tc["recalibration"]
        recal_ok = (found["recent_resistant"] >= int(rc["min_resistant"])
                    and found["recent_resistant_groups"] >= int(rc["min_resistant_groups"]))
        shift = None
        if recal_ok:
            self.ledger.use("recalibration intercept", recent, later, origin)
            shift = intercept_shift(crossfit[[position[int(r)] for r in recent]], y[recent])

        p_later = model.predict_proba(load_rows(self.X, later))[:, 1].astype(np.float64)
        if not np.isfinite(p_later).all():
            raise DevelopmentError(f"{name}: a non-finite predicted probability.")
        y_later = y[later]
        selection_sensitivity = {r: delivered(np.ones(resistant_scores.size), resistant_scores,
                                              cutoffs[r]["threshold"])["sensitivity"] for r in RULES}
        return {"name": name, "origin": origin, "end": str(o["end"]), "counts": found, "fit_seconds": fit_seconds,
                "later": later, "y_later": y_later, "g_later": groups[later], "p_later": p_later,
                "resistant_scores": resistant_scores, "cutoffs": cutoffs,
                "selection_sensitivity": selection_sensitivity,
                "delivered": {r: delivered(y_later, p_later, cutoffs[r]["threshold"]) for r in RULES},
                "shift": shift, "p_later_recalibrated": apply_shift(p_later, shift) if shift is not None else None}


def two_level_bootstrap(periods: list[dict[str, Any]], target: float, confidence: float, resamples: int,
                        seed: int) -> dict[str, Any]:
    """Selection and later-period patient groups resampled together; both rules on every replicate (paired).

    Both rules are functions of the resistant selection scores only, so the selection resample redraws the
    resistant patient groups (one score each) with replacement at a fixed count; the later period's patient
    groups are redrawn with replacement as the project's bootstrap does. The models are held fixed.
    """
    rng = np.random.default_rng(seed)
    layout = []
    for p in periods:
        order = np.argsort(p["g_later"], kind="stable")
        _, starts, counts = np.unique(p["g_later"][order], return_index=True, return_counts=True)
        layout.append((order, starts, counts))
    draws: dict[str, list[float]] = {f"{r}_{k}": [] for r in RULES for k in
                                     ("pooled_sens", "pooled_spec", *(f"{p['name']}_sens" for p in periods),
                                      *(f"{p['name']}_spec" for p in periods))}
    skipped = 0
    for _ in range(resamples):
        tp, pos, tn, neg = dict.fromkeys(RULES, 0), 0, dict.fromkeys(RULES, 0), 0
        per: dict[str, float] = {}
        single_class = False
        for p, (order, starts, counts) in zip(periods, layout, strict=True):
            r = p["resistant_scores"][rng.integers(0, p["resistant_scores"].size, p["resistant_scores"].size)]
            t = {"E": empirical_cutoff(r, target)["threshold"],
                 "U": tolerance_cutoff(r, target, confidence)["threshold"]}
            idx = group_resample_indices(order, starts, counts, rng.integers(0, starts.size, starts.size))
            yb, pb = p["y_later"][idx], p["p_later"][idx]
            if yb.min() == yb.max():
                single_class = True
            for rule in RULES:
                flagged = pb >= t[rule] if t[rule] is not None else np.zeros(pb.size, dtype=bool)
                a, b = int((flagged & (yb == 1)).sum()), int((~flagged & (yb == 0)).sum())
                tp[rule] += a
                tn[rule] += b
                per[f"{rule}_{p['name']}_sens"] = a / max(int(yb.sum()), 1)
                per[f"{rule}_{p['name']}_spec"] = b / max(int((yb == 0).sum()), 1)
            pos += int(yb.sum())
            neg += int((yb == 0).sum())
        if single_class:
            skipped += 1
            continue
        for rule in RULES:
            draws[f"{rule}_pooled_sens"].append(tp[rule] / pos)
            draws[f"{rule}_pooled_spec"].append(tn[rule] / neg)
        for key, value in per.items():
            draws[key].append(value)
    return {"draws": {k: np.asarray(v) for k, v in draws.items()}, "skipped": skipped, "resamples": resamples}


def verdict(u_minus_e: tuple[float, float], u_sens: float, e_sens: float, u_spec: float, feasible: bool,
            target: float, useless: float) -> str:
    """The plan's section 8, applied in its order."""
    if not feasible:
        return "infeasible"
    if u_minus_e[0] <= 0:
        return "not demonstrated"
    if u_spec < useless:
        return "higher sensitivity only by flagging nearly everyone: unhelpful"
    if u_sens < target:
        return f"U improves sensitivity but does not reach {target:.2f}"
    if e_sens >= target:
        return f"both reach {target:.2f}; U only adds specificity cost"
    return f"U improves attainment of the {target:.0%} target (exploratory)"


def pooled_point(periods: list[dict[str, Any]], rule: str) -> dict[str, float]:
    y = np.concatenate([p["y_later"] for p in periods])
    if any(p["cutoffs"][rule]["threshold"] is None for p in periods):     # an infeasible rule has no cut-off
        return {**dict.fromkeys(("sensitivity", "specificity", "flag_rate", "precision"), float("nan")),
                "n": int(y.size), "n_resistant": int(y.sum())}
    flagged = np.concatenate([p["p_later"] >= p["cutoffs"][rule]["threshold"] for p in periods])
    return {"sensitivity": float((flagged & (y == 1)).sum() / (y == 1).sum()),
            "specificity": float((~flagged & (y == 0)).sum() / (y == 0).sum()),
            "flag_rate": float(flagged.mean()), "n": int(y.size), "n_resistant": int(y.sum()),
            "precision": float((flagged & (y == 1)).sum() / max(int(flagged.sum()), 1))}


def slash(counts: dict[str, int], a: str, b: str) -> str:
    return f"{counts[a]} / {counts[b]}"


def band(point: float, low: float, high: float) -> str:
    return f"{point:.3f} ({low:.3f}–{high:.3f})"


def signed(d: dict[str, float]) -> str:
    return f"{d['estimate']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]"


def ci(values: np.ndarray, level: float) -> dict[str, float]:
    low, high = interval(values, level)
    return {"low": low, "high": high}


def run(study: Study) -> dict[str, Any]:
    tc = study.tc
    bs = tc["bootstrap"]
    level = float(bs["level"])
    periods = [study.origin(o) for o in tc["origins"]]
    for p in periods:
        log.info("%s: selection resistant %d, E cut-off %.4f (misses %s), U cut-off %s (k* %s); delivered sensitivity "
                 "E %.3f, U %s", p["name"], p["cutoffs"]["E"]["n"], p["cutoffs"]["E"]["threshold"],
                 p["cutoffs"]["E"]["misses_allowed"], p["cutoffs"]["U"]["threshold"], p["cutoffs"]["U"]["k"],
                 p["delivered"]["E"]["sensitivity"], p["delivered"]["U"]["sensitivity"])
    if time.monotonic() - study.started > float(tc["budget_seconds"]):
        raise DevelopmentError("The compute budget was exceeded; the run is recorded as failed.")
    boot = two_level_bootstrap(periods, study.target, study.confidence, int(bs["resamples"]), int(bs["seed"]))
    d = boot["draws"]
    feasible = all(p["cutoffs"]["U"]["feasible"] for p in periods)
    point = {r: pooled_point(periods, r) for r in RULES}
    u_minus_e = d["U_pooled_sens"] - d["E_pooled_sens"]
    diff_ci = ci(u_minus_e, level)
    primary = {"endpoint": "pooled delivered sensitivity in the later periods, U minus E",
               "U": point["U"], "E": point["E"],
               "U_minus_E": {"estimate": point["U"]["sensitivity"] - point["E"]["sensitivity"], **diff_ci},
               "U_sensitivity_interval": ci(d["U_pooled_sens"], level),
               "E_sensitivity_interval": ci(d["E_pooled_sens"], level),
               "U_specificity_interval": ci(d["U_pooled_spec"], level),
               "E_specificity_interval": ci(d["E_pooled_spec"], level),
               "E_minus_U_specificity": {"estimate": point["E"]["specificity"] - point["U"]["specificity"],
                                         **ci(d["E_pooled_spec"] - d["U_pooled_spec"], level)},
               "attainment_probability": {r: float((d[f"{r}_pooled_sens"] >= study.target).mean()) for r in RULES},
               "bootstrap": {"resamples": boot["resamples"], "skipped_single_class": boot["skipped"],
                             "seed": int(bs["seed"]), "level": level},
               "exploratory": True}
    primary["verdict"] = verdict((diff_ci["low"], diff_ci["high"]), point["U"]["sensitivity"],
                                 point["E"]["sensitivity"], point["U"]["specificity"], feasible, study.target,
                                 float(tc["useless_specificity"]))
    return {"periods": periods, "primary": primary, "boot": boot, "level": level}


def period_rows(study: Study, result: dict[str, Any]) -> list[dict[str, Any]]:
    d, level, rows = result["boot"]["draws"], result["level"], []
    for p in result["periods"]:
        for r in RULES:
            c, dl = p["cutoffs"][r], p["delivered"][r]
            sens_ci, spec_ci = ci(d[f"{r}_{p['name']}_sens"], level), ci(d[f"{r}_{p['name']}_spec"], level)
            rows.append({"period": p["name"], "origin": p["origin"], "rule": r, "feasible": c["feasible"],
                         "threshold": c["threshold"], "selection_resistant": c["n"], "k": c["k"],
                         "misses_allowed": c["misses_allowed"], "stated_probability": c["stated_probability"],
                         "selection_sensitivity": p["selection_sensitivity"][r],
                         "later_n": int(p["y_later"].size), "later_resistant": int(p["y_later"].sum()),
                         "delivered_sensitivity": dl["sensitivity"], "sensitivity_low": sens_ci["low"],
                         "sensitivity_high": sens_ci["high"], "delivered_specificity": dl["specificity"],
                         "specificity_low": spec_ci["low"], "specificity_high": spec_ci["high"],
                         "flag_rate": dl["flag_rate"], "precision": dl["precision"],
                         "flags_nearly_everyone": bool(dl["specificity"] < float(study.tc["useless_specificity"])),
                         "attainment_probability": float((d[f"{r}_{p['name']}_sens"] >= study.target).mean())})
    return rows


def recalibration_rows(study: Study, result: dict[str, Any]) -> list[dict[str, Any]]:
    b = study.ev["bootstrap"]
    rows = []
    for p in result["periods"]:
        if p["shift"] is None:
            rows.append({"period": p["name"], "feasible": False})
            continue
        y, g = p["y_later"], p["g_later"]
        probs = {"uncorrected": p["p_later"], "recalibrated": p["p_later_recalibrated"]}
        s, _ = summarize_bootstrap(y, g, probs, dict.fromkeys(probs, 0.5), resamples=int(b["resamples"]),
                                   level=float(b["level"]), seed=int(b["seed"]), reference="uncorrected",
                                   metrics=("brier",))
        diff = s["differences_to_reference"]["recalibrated"]["brier"]          # uncorrected minus recalibrated
        _, icpt_u = calibration_slope_intercept(y, p["p_later"])
        _, icpt_r = calibration_slope_intercept(y, p["p_later_recalibrated"])
        rows.append({"period": p["name"], "feasible": True, "intercept_shift": p["shift"],
                     "brier_uncorrected": s["intervals"]["uncorrected"]["brier"]["estimate"],
                     "brier_recalibrated": s["intervals"]["recalibrated"]["brier"]["estimate"],
                     "brier_uncorrected_minus_recalibrated": diff["estimate"], "low": diff["low"],
                     "high": diff["high"], "calibration_intercept_uncorrected": icpt_u,
                     "calibration_intercept_recalibrated": icpt_r})
    return rows


def tables(study: Study, result: dict[str, Any], periods_df: pd.DataFrame, recal: list[dict[str, Any]]) -> str:
    pr, useless = result["primary"], float(study.tc["useless_specificity"])

    def ival(point: float, i: dict[str, float]) -> str:
        return f"{point:.3f} ({i['low']:.3f}–{i['high']:.3f})"

    counts = [{"Period": p["name"], "Origin": p["origin"],
               "Fit and selection spectra / resistant": slash(p["counts"], "available", "available_resistant"),
               "Resistant patient groups in selection": p["counts"]["selection_resistant"],
               "E: misses allowed": p["cutoffs"]["E"]["misses_allowed"],
               "U: k* (misses allowed)": (f"{p['cutoffs']['U']['k']} ({p['cutoffs']['U']['misses_allowed']})"
                                          if p["cutoffs"]["U"]["feasible"] else "infeasible"),
               "Later spectra / resistant": f"{p['counts']['later']} / {p['counts']['later_resistant']}",
               "Removed (patient seen earlier that year)": p["counts"]["removed_seen"]} for p in result["periods"]]
    per = [{"Period": r.period, "Rule": r.rule,
            "Cut-off": f"{r.threshold:.4f}" if r.feasible else "infeasible (no cut-off)",
            "Selection sensitivity": f"{r.selection_sensitivity:.3f}",
            "Delivered sensitivity": band(r.delivered_sensitivity, r.sensitivity_low, r.sensitivity_high),
            "Delivered specificity": band(r.delivered_specificity, r.specificity_low, r.specificity_high),
            "Flag rate": f"{r.flag_rate:.3f}", "Precision": f"{r.precision:.3f}",
            "P(≥ 0.90) in resamples": f"{r.attainment_probability:.2f}",
            "Flags nearly everyone": "**yes**" if r.flags_nearly_everyone else "no"} for r in periods_df.itertuples()]
    rec = [{"Period": r["period"], "Intercept shift": f"{r['intercept_shift']:+.3f}",
            "Brier uncorrected → recalibrated": f"{r['brier_uncorrected']:.4f} → {r['brier_recalibrated']:.4f}",
            "Brier(uncorrected) minus Brier(recalibrated)": f"{r['brier_uncorrected_minus_recalibrated']:+.4f} "
                                                            f"[{r['low']:+.4f}, {r['high']:+.4f}]",
            "Calibration intercept uncorrected → recalibrated": f"{r['calibration_intercept_uncorrected']:+.2f} → "
                                                                f"{r['calibration_intercept_recalibrated']:+.2f}"}
           for r in recal if r.get("feasible")]
    u, e = pr["U"], pr["E"]
    out = [f"# Version 1.3 development tables (`{study.dataset}`) — exploratory", "",
           "Generated by `scripts/v13_threshold.py`. **Development only; every number here is exploratory.** No test "
           "part was read and DRIAMS-C was not opened. The later periods are 2017 spectra of the development pool, "
           "already evaluated in aggregate by Version 1.2's forward check, which prompted this question: nothing here "
           f"is independent evidence. Run `{study.run_id}`, plan SHA-256 `{study.plan_sha256[:16]}…`. Research only; "
           "no clinical claim.", "",
           "## Feasibility, as audited before the plan (and checked again by the run)", "", md_table(counts), "",
           f"Rule U needs at least {minimum_feasible_n(study.target, study.confidence)} resistant patient groups to "
           "give any cut-off at 0.90 / 0.95. **Its 95 % is a heuristic here** (plan section 7): the selection scores "
           "are cross-fitted rather than from the model applied, patients cannot be linked across 2016/2017, and the "
           "later periods may differ from the earlier ones — which is the question.", "",
           "## Primary endpoint (exploratory)", "",
           f"Pooled delivered sensitivity over both later periods ({u['n']:,} spectra, {u['n_resistant']} resistant): "
           f"U {ival(u['sensitivity'], pr['U_sensitivity_interval'])}, "
           f"E {ival(e['sensitivity'], pr['E_sensitivity_interval'])}. **U minus E: {signed(pr['U_minus_E'])}**.",
           "",
           f"**At what cost:** pooled specificity U {ival(u['specificity'], pr['U_specificity_interval'])}, E "
           f"{ival(e['specificity'], pr['E_specificity_interval'])}; E minus U "
           f"{pr['E_minus_U_specificity']['estimate']:+.3f} [{pr['E_minus_U_specificity']['low']:+.3f}, "
           f"{pr['E_minus_U_specificity']['high']:+.3f}]. Flag rate U {u['flag_rate']:.3f}, E {e['flag_rate']:.3f}; "
           f"precision U {u['precision']:.3f}, E {e['precision']:.3f}.", "",
           f"**Verdict under the plan: {pr['verdict']}.** No verdict here is operational success: there is no "
           "justified "
           f"operational specificity floor, and {useless:.2f} only names a cut-off that flags nearly everyone.", "",
           f"Share of two-level resamples in which pooled delivered sensitivity reached 0.90: U "
           f"{pr['attainment_probability']['U']:.2f}, E {pr['attainment_probability']['E']:.2f}. This describes the "
           "resampling distribution, which holds the fitted models fixed; it is not a guarantee.", "",
           "## Each later period", "", md_table(per), "",
           "Two-level intervals: 2,000 replicates resampling the selection sample's resistant patient groups and the "
           f"later period's patient groups ({pr['bootstrap']['skipped_single_class']} single-class replicates "
           "skipped).",
           "",
           "## Separate exploratory question: recalibrating the intercept on the most recent six months", "",
           md_table(rec) if rec else "Not feasible under the plan's minimum counts.", "",
           "Probability quality only: a monotone correction preserves order, so a cut-off re-derived on the "
           "recalibrated scale would flag the same isolates. The correction was fitted on cross-fitted predictions "
           "and applied to the final model's, so it is approximate.", ""]
    return "\n".join(out)


def report(study: Study, result: dict[str, Any]) -> dict[str, Any]:
    periods_df = pd.DataFrame(period_rows(study, result))
    recal = recalibration_rows(study, result)
    primary = result["primary"]
    primary["provenance"] = {
        "run_id": study.run_id, "git_commit": study.commit, "plan": study.tc["plan"], "plan_sha256": study.plan_sha256,
        "config_section_sha256": study.config_sha256, "dataset": study.dataset,
        "dataset_row_fingerprint": study.summary["row_fingerprint"], "x_sha256": study.summary["x_sha256"],
        "pool_fingerprint": study.pool_fp, "card_sha256": study.card_sha256, "setting": study.setting,
        "counts": {p["name"]: p["counts"] for p in result["periods"]},
        "production_log_sha256_at_start": study.production_log_sha, "ledger_entries": len(study.ledger.entries),
        "seconds": round(time.monotonic() - study.started, 1), "finished": stamp()}
    periods_df.to_csv(study.report_dir / "periods.csv", index=False, lineterminator="\n")
    pd.DataFrame(recal).to_csv(study.report_dir / "recalibration.csv", index=False, lineterminator="\n")
    np.savez_compressed(study.model_dir / "later_predictions.npz",
                        **{f"{p['name']}__rows": p["later"] for p in result["periods"]},
                        **{f"{p['name']}__p": p["p_later"] for p in result["periods"]})

    base = {"logged_at": stamp(), "git_commit": study.commit, "run_id": study.run_id,
            "plan_sha256": study.plan_sha256, "config_sha256": study.config_sha256, "dataset": study.dataset,
            "dataset_fingerprint": study.summary["row_fingerprint"], "pool_fingerprint": study.pool_fp,
            "status": "ok"}
    rows = []
    for p in result["periods"]:
        y, pl = p["y_later"], p["p_later"]
        slope, intercept = calibration_slope_intercept(y, pl)
        model_metrics = {"n": int(y.size), "n_resistant": int(y.sum()), "brier": float(brier_score_loss(y, pl)),
                         "roc_auc": float(roc_auc_score(y, pl)), "pr_auc": float(average_precision_score(y, pl)),
                         "log_loss": float(log_loss(y, np.clip(pl, 1e-15, 1 - 1e-15), labels=[0, 1])),
                         "calibration_slope": slope, "calibration_intercept": intercept}
        for r in RULES:
            rows.append({**base, "experiment": f"v1.3/{study.run_id}/{p['name']}", "model": f"rule-{r}", "seed": 42,
                         **model_metrics, "delivered_sensitivity": p["delivered"][r]["sensitivity"],
                         "delivered_specificity": p["delivered"][r]["specificity"],
                         "supported_cutoffs": int(p["cutoffs"][r]["feasible"])})
    for r in RULES:
        rows.append({**base, "experiment": f"v1.3/{study.run_id}/pooled", "model": f"rule-{r}", "seed": 42,
                     **dict.fromkeys(("brier", "roc_auc", "pr_auc", "log_loss", "calibration_slope",
                                      "calibration_intercept"), float("nan")),
                     "n": primary[r]["n"], "n_resistant": primary[r]["n_resistant"],
                     "delivered_sensitivity": primary[r]["sensitivity"],
                     "delivered_specificity": primary[r]["specificity"],
                     "supported_cutoffs": int(all(p["cutoffs"][r]["feasible"] for p in result["periods"]))})
    for r in recal:
        if r.get("feasible"):
            rows.append({**base, "experiment": f"v1.3/{study.run_id}/{r['period']}", "model": "recalibrated-intercept",
                         "seed": 42, **dict.fromkeys(("n", "n_resistant", "roc_auc", "pr_auc", "log_loss",
                                                      "calibration_slope", "delivered_sensitivity",
                                                      "delivered_specificity", "supported_cutoffs"), float("nan")),
                         "brier": r["brier_recalibrated"],
                         "calibration_intercept": r["calibration_intercept_recalibrated"]})
    primary["provenance"]["development_log_pre_write"] = study.dev_log.append(rows)
    (study.report_dir / "primary_result.json").write_text(json.dumps(primary, indent=2, default=float),
                                                        encoding="utf-8", newline="\n")
    (study.report_dir / "tables.md").write_text(tables(study, result, periods_df, recal), encoding="utf-8",
                                               newline="\n")
    production = project_path(study.ev["test_log"])
    if study.production_log_sha and sha256_file(production) != study.production_log_sha:
        raise DevelopmentError("The production test log changed during a development run.")
    return primary


def failure_row(study: Study, exc: Exception) -> dict[str, Any]:
    return dict.fromkeys(DEV_LOG_COLUMNS, "") | {
        "logged_at": stamp(), "git_commit": study.commit, "run_id": study.run_id, "plan_sha256": study.plan_sha256,
        "config_sha256": study.config_sha256, "dataset": study.dataset, "experiment": f"v1.3/{study.run_id}/failed",
        "model": "run", "seed": 0, "status": f"failed: {exc}"[:500]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Version 1.3: an uncertainty-aware cut-off rule over time.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    study = None
    try:
        study = Study(load_config(args.config))
        log.info("pool %d spectra (%d resistant), fingerprint %s; run %s", study.pool.size,
                 int(study.y[study.pool].sum()), study.pool_fp, study.run_id)
        with keep_awake():
            primary = report(study, run(study))
        log.info("primary (exploratory): pooled sensitivity U %.3f, E %.3f; U minus E %+.3f [%+.3f, %+.3f] -> %s",
                 primary["U"]["sensitivity"], primary["E"]["sensitivity"], primary["U_minus_E"]["estimate"],
                 primary["U_minus_E"]["low"], primary["U_minus_E"]["high"], primary["verdict"])
        return 0
    except (DataError, ConfigError, SplitError, DevelopmentError, EvaluationError, TuningError,
            ThresholdRuleError) as exc:
        log.error("%s", exc)
        if study is not None:
            try:
                study.dev_log.append([failure_row(study, exc)])
            except (DevelopmentError, EvaluationError, OSError) as inner:
                log.error("the failure could not be recorded in the development log: %s", inner)
        return 1


if __name__ == "__main__":
    sys.exit(main())
