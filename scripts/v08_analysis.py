"""Version 0.8 analysis: score each arm once, apply the pre-registered rules, append, report.

Imported by scripts/adapt_model.py. Every rule applied here is fixed in docs/v0.8_adaptive_plan.md and
protocol amendment 4. Nothing in this module chooses anything: it measures, compares by the pre-specified
method, and classifies by the pre-specified arithmetic.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.adaptation import CONFIRMATORY_ARMS, AdaptationError, bootstrap_p_value, brier, classify, holm
from src.evaluate import append_test_log, classification_metrics, summarize_bootstrap
from src.uncertainty import bootstrap_zone_metrics, zone_metrics
from src.utils import file_hash, get_logger, show_path

log = get_logger("adapt")

STAGE = "v0.8-adaptation"
PRIMARY_BASELINE = "baseline_saved"          # B1
PRIMARY_ARM = "recalibrated"                 # A1


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def record(ctx: Any, status: str, **extra: Any) -> dict[str, Any]:
    ad = ctx.ad
    return {"stage": STAGE, "status": status, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": time.strftime("%Z%z"), "git_commit": ctx.commit,
            "methodology_hash": ctx.methodology_hash, "config_hash": ctx.config_hash,
            "cohort": ctx.cohort, "cohort_rows_fingerprint": ctx.summary_c["row_fingerprint"],
            "cohort_x_sha256": ctx.summary_c["x_sha256"],
            "feature_fingerprint": ctx.summary_c["feature_fingerprint"],
            "archive_sha256": str(ad["cohort"]["archive_checksum"]),
            "partition_fingerprint": ctx.partition["partition_fingerprint"],
            "partition_seed": ctx.seed,
            "adaptation_rows": int(ctx.adapt_rows.size), "evaluation_rows": int(ctx.eval_rows.size),
            "decision_threshold": ctx.threshold,
            "threshold_reading": ("the plan enumerates the changeable parameters as the Platt pair and the "
                                  "zone edge, so the decision threshold is unchanged and every arm is "
                                  "evaluated at the saved model's deployed cut-off"),
            "carried_zone_edge": ctx.carried.lower,
            "group_limitation": ctx.partition["group_limitation"],
            "bootstrap": {"resamples": int(ctx.ev["bootstrap"]["resamples"]),
                          "seed": int(ctx.ev["bootstrap"]["seed"]),
                          "level": float(ctx.ev["bootstrap"]["level"]), "paired": True,
                          "resample_unit": "group (each C spectrum is its own group)"},
            "versions": ctx.versions, **extra, "config": ctx.config}


def run_analysis(ctx: Any, arms: list[dict[str, Any]]) -> None:
    started = time.perf_counter()
    (ctx.report_dir / "run_status.json").write_text(
        json.dumps(record(ctx, "running"), indent=2, default=str), encoding="utf-8")
    before = pd.read_csv(ctx.test_log) if ctx.test_log.is_file() and ctx.test_log.stat().st_size else pd.DataFrame()

    y = ctx.yc[ctx.eval_rows]
    groups = ctx.groups_c[ctx.eval_rows]
    by_key = {a["key"]: a for a in arms}
    if PRIMARY_BASELINE not in by_key or PRIMARY_ARM not in by_key:
        raise AdaptationError(f"The primary comparison needs both {PRIMARY_BASELINE} and {PRIMARY_ARM}.")

    # --- every arm, scored once on the protected part -------------------------------------------------
    report_rows, log_rows, zone_rows = [], [], []
    base = {"logged_at": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": ctx.commit,
            "dataset": ctx.cohort, "dataset_fingerprint": ctx.summary_c["row_fingerprint"],
            "x_sha256": ctx.summary_c["x_sha256"]}
    for a in arms:
        m = classification_metrics(y, a["prob"], a["threshold"])
        bs = brier(y, a["prob"])
        if not np.isfinite(bs) or abs(bs - float(m["brier"])) > 1e-12:
            raise AdaptationError(f"{a['key']}: our Brier {bs} disagrees with the shared metric "
                                  f"{m['brier']}; refusing to report an inconsistent number.")
        for k, v in m.items():
            if isinstance(v, float) and not np.isfinite(v) and k not in ("calibration_slope",
                                                                        "calibration_intercept"):
                raise AdaptationError(f"{a['key']}: metric {k} is not finite ({v}).")
        experiment = f"c_holdout__{a['role']}_{a['key']}"
        row = {"experiment": experiment, "split": "c_holdout", "model": a["model"],
               "seed": int(a.get("seed", ctx.seed)), "train_size": int(a["train_size"]),
               "threshold": float(a["threshold"]), **m}
        log_rows.append({**base, "stage": a["stage"], **row})
        report_rows.append({**row, "role": a["role"], "arm": a["key"], "setting": a["setting"],
                            "brier_checked": bs})
        log.info("%-4s %-22s AUROC %.4f  Brier %.4f  sens %.2f  spec %.2f", a["role"], a["key"],
                 m["roc_auc"], m["brier"], m["sensitivity"], m["specificity"])

        if a["zones"] is not None:
            zm = zone_metrics(y, a["prob"], a["zones"])
            ci = bootstrap_zone_metrics(y, groups, a["prob"], a["zones"],
                                        resamples=int(ctx.ev["bootstrap"]["resamples"]),
                                        level=float(ctx.ev["bootstrap"]["level"]),
                                        seed=int(ctx.ev["bootstrap"]["seed"]))
            zone_rows.append({"role": a["role"], "arm": a["key"], "model": a["model"],
                              "edge": a["zones"].lower, "edge_source":
                                  "carried over from Version 0.6" if a["role"].startswith("B")
                                  else "refitted on the C adaptation part",
                              "n": int(y.size), "n_resistant": int((y == 1).sum()),
                              "n_zone_susceptible": zm.get("n_zone_susceptible"),
                              "coverage": zm.get("share_susceptible"),
                              "coverage_low": ci["share_susceptible"]["low"],
                              "coverage_high": ci["share_susceptible"]["high"],
                              "npv": zm.get("npv_susceptible"),
                              "npv_low": ci["npv_susceptible"]["low"],
                              "npv_high": ci["npv_susceptible"]["high"]})
            log.info("     zone edge %.4f -> covers %d/%d (%.1f%%) at NPV %.4f [%.4f, %.4f]",
                     a["zones"].lower, zm.get("n_zone_susceptible", 0), y.size,
                     100 * (zm.get("share_susceptible") or 0.0), zm.get("npv_susceptible", float("nan")),
                     ci["npv_susceptible"]["low"], ci["npv_susceptible"]["high"])

    # --- append once, before anything else is written -------------------------------------------------
    keys = pd.DataFrame(log_rows)[["experiment", "model", "seed"]]
    if keys.duplicated().any():
        raise AdaptationError("Duplicate (experiment, model, seed) among the rows to append; refusing.")
    if len(before) and set(pd.DataFrame(log_rows)["experiment"]) & set(before["experiment"]):
        raise AdaptationError("An experiment key already exists in the log; refusing to score it twice.")
    append_test_log(ctx.test_log, log_rows, locked=ctx.ev["locked_test_splits"])
    after = pd.read_csv(ctx.test_log)
    if len(before):
        pd.testing.assert_frame_equal(after.iloc[:len(before)].reset_index(drop=True),
                                      before.reset_index(drop=True))
    log.info("%d test evaluations appended to %s", len(log_rows), show_path(ctx.test_log))

    # --- the paired bootstrap, reference = B1 ---------------------------------------------------------
    probs = {a["key"]: a["prob"] for a in arms}
    thresholds = {a["key"]: float(a["threshold"]) for a in arms}
    summary, samples = summarize_bootstrap(
        y, groups, probs, thresholds, resamples=int(ctx.ev["bootstrap"]["resamples"]),
        level=float(ctx.ev["bootstrap"]["level"]), seed=int(ctx.ev["bootstrap"]["seed"]),
        reference=PRIMARY_BASELINE)

    # differences_to_reference is reference - model, so for Brier a POSITIVE value favours the arm.
    p_values = {}
    for arm in CONFIRMATORY_ARMS:
        if arm in samples:
            p_values[arm] = bootstrap_p_value(samples[PRIMARY_BASELINE]["brier"] - samples[arm]["brier"])
    holm_out = holm(p_values, alpha=float(ctx.ad["statistics"]["holm_alpha"]))

    # --- the verdicts, arithmetic ---------------------------------------------------------------------
    b1_zone = next(z for z in zone_rows if z["arm"] == PRIMARY_BASELINE)
    baseline_coverage = float(b1_zone["coverage"])
    baseline_zone_ok = ctx.criterion.passes(float(b1_zone["npv"]), baseline_coverage, baseline_coverage)
    verdicts = []
    for arm in CONFIRMATORY_ARMS:
        if arm not in summary["differences_to_reference"]:
            continue
        d = summary["differences_to_reference"][arm]["brier"]
        z = next((x for x in zone_rows if x["arm"] == arm), None)
        zone_eval = (ctx.criterion.explain(float(z["npv"]), float(z["coverage"]), baseline_coverage)
                     if z else {"passes": False, "npv": float("nan")})
        verdict, reasons = classify(d["low"], d["high"], bool(zone_eval["passes"]), baseline_zone_ok,
                                    float(zone_eval.get("npv", float("nan"))),
                                    float(ctx.ad["endpoints"]["zone_target_npv"]))
        entry = {"arm": arm, "role": "A1" if arm == PRIMARY_ARM else "A2",
                 "kind": "primary" if arm == PRIMARY_ARM else "confirmatory secondary",
                 "brier_baseline": summary["intervals"][PRIMARY_BASELINE]["brier"]["estimate"],
                 "brier_arm": summary["intervals"][arm]["brier"]["estimate"],
                 "delta": d["estimate"], "low": d["low"], "high": d["high"],
                 "p_value": p_values.get(arm), "holm": holm_out.get(arm, {}),
                 "zone": zone_eval, "baseline_zone_passes": baseline_zone_ok,
                 "verdict": verdict, "reasons": reasons}
        verdicts.append(entry)
        log.info("%s (%s): Brier %.4f vs baseline %.4f, paired delta %+.4f [%+.4f, %+.4f] -> %s",
                 arm, entry["kind"], entry["brier_arm"], entry["brier_baseline"], d["estimate"],
                 d["low"], d["high"], verdict.upper())
        for r in reasons:
            log.info("      %s", r)

    # --- outputs --------------------------------------------------------------------------------------
    write_csv(pd.DataFrame(report_rows), ctx.report_dir / "evaluation_metrics.csv")
    write_csv(pd.DataFrame(zone_rows), ctx.report_dir / "zone_results.csv")
    (ctx.report_dir / "primary_result.json").write_text(
        json.dumps({"primary_baseline": PRIMARY_BASELINE, "primary_arm": PRIMARY_ARM,
                    "endpoint": "brier", "direction": "lower_is_better",
                    "baseline_coverage_for_the_floor": baseline_coverage,
                    "coverage_slack": ctx.criterion.coverage_slack,
                    "target_npv": ctx.criterion.target_npv,
                    "verdicts": verdicts, "holm": holm_out}, indent=2, default=str), encoding="utf-8")
    (ctx.report_dir / "bootstrap.json").write_text(json.dumps(summary, indent=2, default=str),
                                                   encoding="utf-8")
    np.savez_compressed(ctx.model_dir / "evaluation_probabilities.npz",
                        rows=ctx.eval_rows, **{a["key"]: a["prob"] for a in arms})
    details = {a["key"]: {"role": a["role"], "model": a["model"], "setting": a["setting"],
                          **(a.get("detail") or {})} for a in arms}
    (ctx.report_dir / "arms.json").write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")

    made = figures(ctx, report_rows, zone_rows, verdicts)

    full = record(ctx, "finished", seconds=round(time.perf_counter() - started, 1),
                  n_test_evaluations=len(log_rows),
                  primary=next((v for v in verdicts if v["arm"] == PRIMARY_ARM), None),
                  verdicts=verdicts, holm=holm_out, arms=details,
                  test_log_sha256_before=ctx.test_log_hash_before,
                  test_log_sha256_after=file_hash(ctx.test_log),
                  plots=[p.name for p in made])
    for name in ("run_config.json", "run_status.json"):
        (ctx.report_dir / name).write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")
    log.info("Version 0.8 reports written to %s", show_path(ctx.report_dir))
    log.info("finished in %.1f minutes", (time.perf_counter() - started) / 60)


def figures(ctx: Any, report_rows: list[dict], zone_rows: list[dict], verdicts: list[dict]) -> list[Path]:
    """Two figures, drawn only from what was written. A failure here must not lose a scored run."""
    from src.model_plots import plot_adaptation_brier, plot_adaptation_zone

    stem = f"{ctx.cohort}_adaptation"
    made: list[Path] = []
    for name, draw in (("brier", lambda: plot_adaptation_brier(
                            pd.DataFrame(verdicts),
                            "Version 0.8: paired change in Brier score at DRIAMS-C (protected held-out part)",
                            ctx.plot_dir / f"{stem}_brier.png")),
                       ("zone", lambda: plot_adaptation_zone(
                            pd.DataFrame(zone_rows), float(ctx.criterion.target_npv),
                            float(ctx.criterion.coverage_slack),
                            "Version 0.8: the confidence zone at DRIAMS-C",
                            ctx.plot_dir / f"{stem}_zone.png"))):
        try:
            made.append(draw())
        except (ValueError, KeyError, IndexError) as exc:
            log.warning("the %s figure was not drawn (%s: %s)", name, type(exc).__name__, exc)
    if made:
        log.info("figures: %s", ", ".join(p.name for p in made))
    return made
