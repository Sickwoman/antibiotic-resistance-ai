"""Version 2.0's one-time evaluation of the frozen ciprofloxacin model on MARISMa, as frozen at step 4.

docs/v2.0_marisma_plan.md, amendment F5 (the specification) and the endpoints as amended. This script is written and
tested at step 4; running it on MARISMa is step 5 and needs the owner's separate approval, recorded in the plan as a
scoring authorisation (src/v20_scoring_guard.py). Without that record every production path refuses.

Modes:
- `--check` (the default): verifies everything that can be verified before scoring, and prints the plan of the run:
  the code fingerprint, the candidate cohort, the internal side's reproduction, the log state, the keys and the
  outputs. It loads no model, predicts nothing and reads no interpretation.
- `--production`: the one-time run. Preflight checks, then, in order and once each: predict the 1,594 candidate
  isolates; save the predictions; read the interpretations; save them; analyse; append two rows to the production log
  through the strict gate; write the report. A state file outside the repository records every stage. Any failure
  records its stage and stops, and a later run refuses to start, so nothing is rescored without review.
- `--rebuild-report`: after a failure while writing the report, rebuilds it from the saved predictions and
  interpretations. It never predicts and never appends.

    python scripts/v20_score_marisma.py --check
    python scripts/v20_score_marisma.py --production
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import file_sha256, load_dataset  # noqa: E402
from src.evaluate import (  # noqa: E402
    EvaluationError,
    append_test_log,
    assert_log_ready_for_append,
    classification_metrics,
)
from src.marisma_labels import CONFLICTING, MISSING  # noqa: E402
from src.predict import load_zones  # noqa: E402
from src.utils import get_logger, git_commit, load_config  # noqa: E402
from src.v20_evaluation import (  # noqa: E402
    RESAMPLES,
    SEED,
    UNAVAILABLE,
    evaluate_population,
    family,
    gap,
    labels,
    strip_private,
)
from src.v20_scoring_guard import ROOT as GUARD_ROOT  # noqa: E402
from src.v20_scoring_guard import GuardError, code_fingerprint, verify  # noqa: E402

THRESHOLD = 0.14261540693905073
ZONE_LOWER = 0.10255963637363619
BUNDLE = "models/v0.4/ecoli_ciprofloxacin/best_random.joblib"
ZONE_FILE = "results/metrics/v0.6/ecoli_ciprofloxacin/uncertainty.json"
INTERNAL = "models/v0.4/ecoli_ciprofloxacin/test_probabilities.npz"
INTERNAL_KEY, INTERNAL_ROWS = "random__tuned_lightgbm__seed42", "random__rows"
INTERNAL_LOG_KEY = {"stage": "v0.4-tuned", "experiment": "random", "model": "tuned_lightgbm", "seed": 42}
LOG = "results/experiments/test_evaluations.csv"
LOG_BEFORE = ("6528eb2abcdfb611d65712354b8a05daed03febdb137a0910ed0eb7764f3ddba", 113)
REPORT = "results/metrics/v2.0/marisma_evaluation.json"
STEP3_JSON = "results/metrics/v2.0/step3_run2.json"
ACCEPTED_STEP3 = "8c85055a721bc7bc67cbb808f3e0d61e8ec06c1e"
COLUMN = "Ciprofloxacin"                     # the exact AMR.csv interpretation column; nothing is substituted
CANDIDATE = {"year": "2024", "instrument": "MBT-WIN10"}
EXPERIMENT = "external__MARISMa-2024__saved_model"
ROWS = (  # (stage, dataset, population): the two rows F5.10 fixes
    ("v2.0-external", "marisma_ecoli_ciprofloxacin", "primary"),
    ("v2.0-external-sensitivity", "marisma_ecoli_ciprofloxacin__intermediate-exclude", "i_excluded"),
)
KEYS = tuple((dataset, EXPERIMENT, "v0.4_tuned_lightgbm", 42) for _, dataset, _ in ROWS)
STAGES = ("started", "predicted", "labels_read", "analysed", "log_appended", "complete")

log = get_logger("v20_score")


class RunError(RuntimeError):
    """The run stops here; the state file records the stage."""


def now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def paths() -> dict[str, Path]:
    data_root = Path(load_config()["paths"]["driams_root"])
    work = data_root / "MARISMa_v2.0.0_work"
    return {"run2": work / "run2_2026-10-04", "scoring": work / "scoring_v2.0",
            "amr": data_root / "MARISMa_v2.0.0_sealed" / "AMR.csv"}


# --- inputs fixed before scoring ------------------------------------------------------------------------------------

def candidate_cohort(run2: Path) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """F5.2: the prepared isolates from the 2024 year folder acquired on MBT-WIN10, their frozen feature rows (X.npy
    holds the selected spectra in selection order) and their source classes. Fixed without any label."""
    selection = pd.read_csv(run2 / "selection.csv", dtype=str, keep_default_na=False)
    selected = selection[selection["status"] == "selected"].reset_index(drop=True)
    X = np.load(run2 / "X.npy", mmap_mode="r")
    if X.shape != (len(selected), 6000):
        raise RunError("the feature matrix does not match the selected spectra")
    mask = ((selected["primary"] == "True") & (selected["year"] == CANDIDATE["year"])
            & (selected["instrument"] == CANDIDATE["instrument"])).to_numpy()
    ids = selected.loc[mask, "isolate"].tolist()
    sources = pd.read_csv(run2 / "source_classes.csv", dtype=str, keep_default_na=False)
    source = dict(zip(sources["isolate"], sources["source"], strict=True))
    return ids, np.asarray(X[np.flatnonzero(mask)], dtype=np.float32), {i: source[i] for i in ids}


def step3_consistency(run2: Path) -> dict[str, Any]:
    """The accepted step 3 artifacts: every work file matches hashes.json, and the selection reproduces the counts
    of the accepted step3_run2.json (amendment F2)."""
    hashes = json.loads((run2 / "hashes.json").read_text(encoding="utf-8"))
    for name in ("replicate_folders.csv", "source_classes.csv", "selection.csv", "attempts.csv", "X.npy"):
        if file_sha256(run2 / name) != hashes[name]:
            raise RunError(f"{name} differs from the hash step 3 recorded")
    accepted = json.loads((ROOT / STEP3_JSON).read_text(encoding="utf-8"))
    selection = pd.read_csv(run2 / "selection.csv", dtype=str, keep_default_na=False)
    kept = selection[selection["status"] == "selected"]
    if len(kept) != accepted["features"]["spectra"]:
        raise RunError("the selection does not reproduce the accepted number of prepared spectra")
    cells = kept.groupby(["year", "instrument"]).size()
    for key, cell in accepted["by_year_and_instrument"].items():
        year, instrument = key.split(" ", 1)
        if int(cells.get((year, instrument), 0)) != cell["kept"]:
            raise RunError(f"the selection does not reproduce the accepted count for {key}")
    return {"work_files_match_hashes": True, "counts_match_accepted_json": True,
            "accepted_commit": ACCEPTED_STEP3}


def internal_side() -> dict[str, Any]:
    """Endpoint 2's internal side: M-cip's stored random test probabilities, used only once they reproduce the
    logged AUROC (verification item 8)."""
    from sklearn.metrics import roc_auc_score
    _, meta, _ = load_dataset(ROOT / "data/processed/ecoli_ciprofloxacin", verify_x=False)
    stored = np.load(ROOT / INTERNAL)
    rows = stored[INTERNAL_ROWS].astype(np.int64)
    p = stored[INTERNAL_KEY].astype(np.float64)
    y = meta["label"].to_numpy()[rows].astype(np.int64)
    groups = meta["group_id"].to_numpy()[rows]
    test_log = pd.read_csv(ROOT / LOG)
    row = test_log[(test_log["stage"] == INTERNAL_LOG_KEY["stage"]) & (test_log["experiment"] ==
                   INTERNAL_LOG_KEY["experiment"]) & (test_log["model"] == INTERNAL_LOG_KEY["model"])
                   & (test_log["seed"] == INTERNAL_LOG_KEY["seed"])]
    if len(row) != 1:
        raise RunError("the internal side's logged row is not unique")
    logged, reproduced = float(row["roc_auc"].iloc[0]), float(roc_auc_score(y, p))
    if abs(logged - reproduced) > 1e-9 or y.size != int(row["n"].iloc[0]):
        raise RunError("the stored internal probabilities do not reproduce the logged AUROC")
    return {"y": y, "p": p, "groups": groups, "logged_auroc": logged, "reproduced_auroc": reproduced,
            "n": int(y.size), "n_resistant": int(y.sum())}


def log_state(log_path: Path, expected: tuple[str, int], keys=KEYS) -> pd.DataFrame:
    """The production log is exactly as recorded, and none of the planned keys is in it."""
    if file_sha256(log_path) != expected[0]:
        raise RunError("the production log is not the recorded file")
    frame = pd.read_csv(log_path)
    if len(frame) != expected[1]:
        raise RunError("the production log does not hold the recorded number of rows")
    existing = set(frame[["dataset", "experiment", "model", "seed"]].itertuples(index=False, name=None))
    if existing & set(keys):
        raise RunError("a planned experiment key is already in the production log")
    return frame


def protected() -> dict[str, Any]:
    """The protected state of steps 1-3 (bundles, zone, logs, DRIAMS datasets), via the preparation script."""
    spec = importlib.util.spec_from_file_location("v20_prepare", ROOT / "scripts/v20_prepare_marisma.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.protected_state()


def frozen_zone():
    zones = load_zones(ROOT / ZONE_FILE)
    if zones is None or zones.lower != ZONE_LOWER or zones.threshold != THRESHOLD:
        raise RunError("the frozen zone is not the recorded one")
    return zones


# --- the analysis (pure: saved predictions and interpretations in) -------------------------------------------------

def analyse(ids: list[str], prob: np.ndarray, tokens: dict[str, str], read_counts: dict[str, int],
            sources: dict[str, str], zones, internal: dict[str, Any], *, resamples: int = RESAMPLES,
            seed: int = SEED) -> dict[str, Any]:
    """Every available registered endpoint, from the one set of predictions (F5.2-F5.9)."""
    score = dict(zip(ids, np.asarray(prob, dtype=np.float64), strict=True))
    matched = {i: t for i, t in tokens.items() if i in score and t not in (MISSING, CONFLICTING)}
    primary_labels, primary_counts = labels(matched)
    excluded_i_labels, excluded_i_counts = labels(matched, exclude_intermediate=True)

    def arrays(chosen: dict[str, int]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        order = sorted(chosen)
        return (np.array([chosen[i] for i in order], dtype=np.int64),
                np.array([score[i] for i in order], dtype=np.float64), order)

    y, p, order = arrays(primary_labels)
    main = evaluate_population(y, p, THRESHOLD, zones, test=True, resamples=resamples, seed=seed)
    fam = family(main["primary"], None)
    gap_result = gap(internal["y"], internal["p"], internal["groups"], THRESHOLD,
                     main["metrics"]["roc_auc"]["estimate"] if isinstance(main["metrics"]["roc_auc"]["estimate"],
                                                                          float) else math.nan,
                     main["_auroc_draws"], resamples=resamples, seed=seed)
    yi, pi, _ = arrays(excluded_i_labels)
    excluded_i = evaluate_population(yi, pi, THRESHOLD, zones, test=False, resamples=resamples, seed=seed)
    groups = {}
    for group in ("clinical", "ambiguous"):
        members = {i: v for i, v in primary_labels.items() if sources.get(i) == group}
        yg, pg, _ = arrays(members)
        groups[group] = evaluate_population(yg, pg, THRESHOLD, zones, test=False, resamples=resamples, seed=seed)
        groups[group]["note"] = "descriptive; no p-value (A3.6)"
    populations = {
        "candidate_scoring_cohort": len(ids), "matched_to_an_amr_record": read_counts["matched"],
        "with_a_non_missing_interpretation": read_counts["matched"] - read_counts["missing"],
        "conflicting_interpretations": read_counts["conflicting"],
        "primary_analysis": primary_counts, "i_excluded_analysis": excluded_i_counts,
        "primary_by_source": {g: int(sum(1 for i in primary_labels if sources.get(i) == g))
                              for g in ("clinical", "ambiguous")},
    }
    rows = {"primary": (y, p), "i_excluded": (yi, pi)}
    return {
        "scope": "the frozen ciprofloxacin model on eligible MARISMa E. coli isolates of 2024, acquired on "
                 "MBT-WIN10; all sources retained; screening status could not be determined",
        "populations": populations,
        "primary": main, "holm_family": fam, "gap": gap_result,
        "sensitivity_analyses": {
            "i_excluded": excluded_i,
            "all_sources": "identical to the primary cohort (no isolate was excluded by source, amendment E3); "
                           "not additional evidence",
            "by_sample_source_group": groups,
            "by_period": UNAVAILABLE + ": no E. coli AMR.csv record outside 2024 (amendment D6)"},
        "availability": availability(),
        "internal_side": {k: internal[k] for k in ("logged_auroc", "reproduced_auroc", "n", "n_resistant")},
        "_log_metrics": {name: classification_metrics(a, b, THRESHOLD) if a.size else None
                         for name, (a, b) in rows.items()},
        "_scored_order": order,
    }


def availability() -> dict[str, str]:
    """Which registered endpoints this evaluation can report (amendments D6, F3)."""
    return {
        "primary (AUROC, Brunner–Munzel, Holm)": "ciprofloxacin: available; ceftriaxone: unavailable (dropped at "
                                                 "step 2, Holm input p = 1)",
        "gap (internal minus MARISMa AUROC)": "ciprofloxacin: available; ceftriaxone: unavailable",
        "threshold metrics": "ciprofloxacin: available; ceftriaxone: unavailable",
        "probabilities (Brier, calibration)": "ciprofloxacin: available; ceftriaxone: unavailable",
        "zone (M-cip only)": "available",
        "linkage robustness": "ciprofloxacin: available; ceftriaxone: unavailable",
        "I excluded": "ciprofloxacin: available, descriptive; ceftriaxone: unavailable",
        "all sample sources": "identical to the primary analysis: no isolate was excluded by source",
        "by period (2018-2019, 2020-2021, 2022-2024)": "unavailable: no E. coli AMR.csv record outside 2024",
        "by sample-source group": "ciprofloxacin: descriptive, where each group has both classes",
    }


def log_rows(result: dict[str, Any], commit: str) -> list[dict[str, Any]]:
    """F5.10's two rows; metrics as src/evaluate.classification_metrics defines them for every earlier row."""
    out = []
    for stage, dataset, population in ROWS:
        metrics = result["_log_metrics"][population]
        if metrics is None:
            raise RunError(f"the {population} population is empty: no row can be written")
        out.append({"logged_at": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": commit, "stage": stage,
                    "dataset": dataset, "dataset_fingerprint": f"step3-{ACCEPTED_STEP3[:7]}",
                    "x_sha256": "recorded-locally (run2_2026-10-04/hashes.json)", "experiment": EXPERIMENT,
                    "split": "external", "model": "v0.4_tuned_lightgbm", "seed": 42, "train_size": 2977,
                    **{k: metrics[k] for k in ("n", "n_resistant", "threshold", "roc_auc", "pr_auc", "brier",
                                               "sensitivity", "specificity", "precision", "f1", "accuracy",
                                               "balanced_accuracy", "tp", "fp", "tn", "fn")}})
    return out


# --- the run --------------------------------------------------------------------------------------------------------

class Run:
    """Stages in order, each once, with the state saved outside the repository. Inputs are injected, so the
    sequence, the failure handling and the log gate are tested with synthetic predictions and a temporary log."""

    def __init__(self, work: Path, log_path: Path, report_path: Path, *, ids: list[str], X: np.ndarray,
                 sources: dict[str, str], zones, internal: dict[str, Any],
                 predict: Callable[[np.ndarray], np.ndarray],
                 read_labels: Callable[[list[str]], tuple[dict[str, str], dict[str, int]]],
                 log_expected: tuple[str, int], commit: str, resamples: int = RESAMPLES,
                 provenance: dict[str, Any] | None = None):
        self.work, self.log_path, self.report_path = Path(work), Path(log_path), Path(report_path)
        self.ids, self.X, self.sources, self.zones, self.internal = ids, X, sources, zones, internal
        self.predict, self.read_labels = predict, read_labels
        self.log_expected, self.commit, self.resamples = log_expected, commit, resamples
        self.provenance = provenance or {}
        self.state_path = self.work / "state.json"

    def _state(self, stage: str, **extra: Any) -> None:
        state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.is_file() else \
            {"history": []}
        state["history"].append({"stage": stage, "utc": now()})
        state.update(stage=stage, **extra)
        self.state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    def execute(self) -> dict[str, Any]:
        if self.state_path.exists():
            raise RunError("a state file exists: this evaluation already started. Nothing is rerun; review the "
                           "state and use --rebuild-report only after a failure while writing the report")
        self.work.mkdir(parents=True, exist_ok=True)
        before = pd.read_csv(self.log_path)
        self._state("started", provenance=self.provenance)
        stage = "predict"
        try:
            prob = np.asarray(self.predict(self.X), dtype=np.float64)          # the one prediction
            if prob.shape != (len(self.ids),) or not np.isfinite(prob).all():
                raise RunError("the model returned an unexpected prediction array")
            np.savez(self.work / "predictions.npz", isolates=np.array(self.ids), probabilities=prob)
            self._state("predicted", predictions_sha256=file_sha256(self.work / "predictions.npz"))
            stage = "labels"
            tokens, counts = self.read_labels(self.ids)                       # the first and only label read
            pd.DataFrame({"isolate": list(tokens), "interpretation": list(tokens.values())}).to_csv(
                self.work / "interpretations.csv", index=False, lineterminator="\n")
            (self.work / "read_counts.json").write_text(json.dumps(counts) + "\n", encoding="utf-8")
            self._state("labels_read", interpretations_sha256=file_sha256(self.work / "interpretations.csv"))
            stage = "analyse"
            result = analyse(self.ids, prob, tokens, counts, self.sources, self.zones, self.internal,
                             resamples=self.resamples)
            self._state("analysed")
            stage = "append"
            self._append(result, before)
            self._state("log_appended")
            stage = "report"
            self._report(result)
            self._state("complete")
            return result
        except Exception as exc:
            self._state("failed", failed_stage=stage, error=f"{type(exc).__name__}: {str(exc)[:300]}")
            raise

    def _append(self, result: dict[str, Any], before: pd.DataFrame) -> None:
        rows = log_rows(result, self.commit)
        pre = assert_log_ready_for_append(self.log_path, rows, expected_sha256=self.log_expected[0],
                                          expected_rows=self.log_expected[1], committed=before)
        (self.work / "pre_write_checks.json").write_text(json.dumps(pre, indent=2) + "\n", encoding="utf-8")
        old = self.log_path.read_bytes()
        append_test_log(self.log_path, rows)
        if not self.log_path.read_bytes().startswith(old):
            raise EvaluationError("the log no longer begins with the bytes it had before the append")
        if len(pd.read_csv(self.log_path)) != pre["expected_rows_after"]:
            raise EvaluationError("the log does not hold the expected number of rows after the append")

    def _report(self, result: dict[str, Any]) -> None:
        out = {"what": "Version 2.0: the one-time evaluation of the frozen ciprofloxacin model on MARISMa "
                       "(docs/v2.0_marisma_plan.md, amendment F5)", "written_utc": now(), **self.provenance,
               **strip_private(result)}
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(out, indent=2, default=_json) + "\n", encoding="utf-8")


def _json(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value).__name__)


def rebuild_report(run: Run) -> dict[str, Any]:
    """After a failure while writing the report: the report again, from the saved predictions and interpretations.
    It never predicts and never appends."""
    state = json.loads(run.state_path.read_text(encoding="utf-8"))
    if not (state["stage"] == "log_appended" or (state["stage"] == "failed" and state.get("failed_stage") ==
                                                 "report")):
        raise RunError("the report can be rebuilt only after the log was appended and the report failed")
    saved = np.load(run.work / "predictions.npz")
    if file_sha256(run.work / "predictions.npz") != state["predictions_sha256"] or \
            list(saved["isolates"]) != run.ids:
        raise RunError("the saved predictions are not the ones the run recorded")
    frame = pd.read_csv(run.work / "interpretations.csv", dtype=str, keep_default_na=False)
    tokens = dict(zip(frame["isolate"], frame["interpretation"], strict=True))
    counts = json.loads((run.work / "read_counts.json").read_text(encoding="utf-8"))
    result = analyse(run.ids, saved["probabilities"], tokens, counts, run.sources, run.zones, run.internal,
                     resamples=run.resamples)
    run._report(result)
    run._state("complete", report_rebuilt=True)
    return result


# --- production -----------------------------------------------------------------------------------------------------

def preflight(where: dict[str, Path]) -> dict[str, Any]:
    """Everything checked before a prediction, except the authorisation (verified separately)."""
    checks = {"code_fingerprint": code_fingerprint(GUARD_ROOT), "git_commit": git_commit()}
    state = protected()
    checks["protected_state"] = {"bundle_sha256": state["bundles"]["ciprofloxacin"], "zone_sha256": state["zone"],
                                 "logs": state["logs"], "datasets": state["datasets"]}
    zones = frozen_zone()
    checks["zone"] = {"lower": zones.lower, "threshold": zones.threshold}
    checks["step3"] = step3_consistency(where["run2"])
    ids, X, sources = candidate_cohort(where["run2"])
    checks["candidate_cohort"] = {"isolates": len(ids), "features": list(X.shape)}
    internal = internal_side()
    checks["internal_side"] = {k: internal[k] for k in ("logged_auroc", "reproduced_auroc", "n", "n_resistant")}
    log_state(ROOT / LOG, LOG_BEFORE)
    checks["log"] = {"sha256": LOG_BEFORE[0], "rows": LOG_BEFORE[1], "planned_keys_absent": True}
    for path in (ROOT / REPORT, where["scoring"] / "state.json"):
        if path.exists():
            raise RunError(f"{path.name} already exists: the evaluation has run or started")
    checks["outputs_absent"] = True
    checks["planned"] = {"keys": [list(k) for k in KEYS], "rows_to_append": len(KEYS), "report": REPORT,
                         "work_folder": str(where["scoring"])}
    return {"checks": checks, "ids": ids, "X": X, "sources": sources, "zones": zones, "internal": internal}


def production_predict(X: np.ndarray) -> np.ndarray:
    """The frozen M-cip bundle, verified against its checksum, on the candidate cohort's features: once."""
    from src.predict import load_bundle, predict_features
    bundle = load_bundle(ROOT / BUNDLE)
    if float(bundle["threshold"]) != THRESHOLD or bundle["model_version"] != "v0.4.0-tuned_lightgbm-random-seed42":
        raise RunError("the bundle is not the frozen M-cip")
    return predict_features(bundle, X)[0]


def production_run() -> dict[str, Any]:
    authorisation = verify(GUARD_ROOT)                       # raises GuardError without the owner's record
    where = paths()
    ready = preflight(where)
    from src.marisma_labels import read_interpretations

    def read_labels(ids: list[str]) -> tuple[dict[str, str], dict[str, int]]:
        return read_interpretations(where["amr"], ids, COLUMN)

    run = Run(where["scoring"], ROOT / LOG, ROOT / REPORT, ids=ready["ids"], X=ready["X"], sources=ready["sources"],
              zones=ready["zones"], internal=ready["internal"], predict=production_predict, read_labels=read_labels,
              log_expected=LOG_BEFORE, commit=git_commit() or "unknown",
              provenance={"authorised_commit": authorisation.authorised_commit,
                          "code_fingerprint": authorisation.code_fingerprint, "preflight": ready["checks"]})
    return run.execute()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify and print the plan; no model, no labels")
    mode.add_argument("--production", action="store_true", help="the one-time evaluation (needs the authorisation)")
    mode.add_argument("--rebuild-report", action="store_true", help="rebuild the report from the saved state")
    args = parser.parse_args(argv)
    if args.production:
        try:
            production_run()
        except GuardError as exc:
            log.error("refused: %s", exc)
            return 2
        return 0
    if args.rebuild_report:
        verify(GUARD_ROOT)
        where = paths()
        ids, X, sources = candidate_cohort(where["run2"])
        run = Run(where["scoring"], ROOT / LOG, ROOT / REPORT, ids=ids, X=X, sources=sources, zones=frozen_zone(),
                  internal=internal_side(), predict=_never, read_labels=_never_labels, log_expected=LOG_BEFORE,
                  commit=git_commit() or "unknown")
        rebuild_report(run)
        return 0
    checks = preflight(paths())["checks"]
    try:
        verify(GUARD_ROOT)
        checks["authorisation"] = "recorded and verified"
    except GuardError as exc:
        checks["authorisation"] = f"not recorded or not valid: {exc}"
    print(json.dumps(checks, indent=2, default=_json))
    return 0


def _never(_X: np.ndarray) -> np.ndarray:
    raise RunError("this path never predicts")


def _never_labels(_ids: list[str]):
    raise RunError("this path never reads interpretations")


if __name__ == "__main__":
    raise SystemExit(main())
