"""Version 1.2: development-only cross-validation of calibration and the cut-off.

Everything here is fixed by docs/v1.2_calibration_plan.md (protocol amendment 9), recorded before this file
existed. It works on a **development pool** that never touched an inspected test part, and it never reads a
test part: the pool is derived from the committed split files, checked, and every fit, calibrator and cut-off
is chosen inside cross-validation folds of that pool.

The pieces are the project's own: the Version 0.4 calibrated fit (`fit_calibrated`, sigmoid, `ensemble=False`),
its inner folds (`grouped_folds`), the protocol's cut-off rule (`choose_threshold`) and the strict pre-write
gate (`assert_log_ready_for_append`). What is new is only what the plan needs: the pool, a patient-grouped
validation slice for the unchanged Version 1.1 procedure, the cross-fitted predictions a cut-off can be chosen
on, the support rule for a cut-off's target, and a development log that can never be the production log.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import _SigmoidCalibration
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from src.dataset import sample_keys
from src.evaluate import assert_log_ready_for_append, choose_threshold
from src.splits import Split
from src.tuning import FamilySpec, base_pipeline, pipeline_params

DEV_LOG_COLUMNS = ["logged_at", "git_commit", "run_id", "plan_sha256", "config_sha256", "dataset",
                   "dataset_fingerprint", "pool_fingerprint", "experiment", "model", "seed", "n", "n_resistant",
                   "brier", "roc_auc", "pr_auc", "log_loss", "calibration_slope", "calibration_intercept",
                   "delivered_sensitivity", "delivered_specificity", "supported_cutoffs", "status"]


class DevelopmentError(RuntimeError):
    """A Version 1.2 design or integrity check failed; nothing is reported as a result."""


# --- the development pool --------------------------------------------------------------------------------------

def spent_rows(splits: dict[str, Split], names: list[str]) -> np.ndarray:
    """Every row of the named splits' test parts: the parts whose results have been inspected."""
    missing = [n for n in names if n not in splits]
    if missing:
        raise DevelopmentError(f"The dataset has no {missing} split, so the spent rows cannot be known.")
    return np.unique(np.concatenate([np.asarray(splits[n].test, dtype=np.int64) for n in names]))


def development_pool(meta: pd.DataFrame, splits: dict[str, Split], source: str, spent: list[str]) -> np.ndarray:
    """`source`'s training and validation rows that never touched a spent test part, by row or by patient group.

    Derived rather than drawn, so it is the same pool every time and anyone can recompute it from the committed
    split files. A row whose patient group also has a spent spectrum is removed with it: that patient's other
    result has already been seen.
    """
    s = splits[source]
    candidates = np.unique(np.concatenate([np.asarray(s.train, dtype=np.int64),
                                           np.asarray(s.validation, dtype=np.int64)]))
    used = spent_rows(splits, spent)
    groups = meta["group_id"].to_numpy()
    pool = np.setdiff1d(candidates, used)
    pool = pool[~np.isin(groups[pool], np.unique(groups[used]))]
    if np.intersect1d(pool, used).size or np.isin(groups[pool], groups[used]).any():
        raise DevelopmentError("The development pool shares a row or a patient group with a spent test part.")
    return pool


def pool_fingerprint(meta: pd.DataFrame, rows: np.ndarray) -> str:
    """Which spectra are in the pool (by site, year folder and code, not row number), as 16 hex characters."""
    keys = sorted(sample_keys(meta)[np.asarray(rows, dtype=np.int64)].tolist())
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()[:16]


# --- folds ------------------------------------------------------------------------------------------------------

def outer_folds(y: np.ndarray, groups: np.ndarray, k: int, seed: int,
                min_resistant: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Patient-grouped, stratified outer folds (positions within the pool). Too few resistant in a held-out fold
    is a design failure (the plan's section 9), not something to work around."""
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [(np.asarray(tr), np.asarray(te)) for tr, te in splitter.split(np.zeros(len(y)), y, groups)]
    for i, (tr, te) in enumerate(folds):
        if np.intersect1d(np.unique(groups[tr]), np.unique(groups[te])).size:
            raise DevelopmentError(f"Outer fold {i} splits a patient group between training and held-out rows.")
        if int(y[te].sum()) < min_resistant:
            raise DevelopmentError(f"Outer fold {i} holds {int(y[te].sum())} resistant spectra, fewer than the "
                                   f"{min_resistant} the plan requires; the design cannot support this run.")
    return folds


def validation_slice(y: np.ndarray, groups: np.ndarray, denominator: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Arm B's Version 1.1 procedure inside a fold: a patient-grouped, stratified 1/`denominator` validation
    slice of the outer-training rows (positions within them), the rest for fitting."""
    splitter = StratifiedGroupKFold(n_splits=denominator, shuffle=True, random_state=seed)
    fit, val = next(splitter.split(np.zeros(len(y)), y, groups))
    if np.intersect1d(np.unique(groups[fit]), np.unique(groups[val])).size:
        raise DevelopmentError("The validation slice splits a patient group.")
    return np.asarray(fit), np.asarray(val)


# --- the cut-off and its support -------------------------------------------------------------------------------

def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The 95 % Wilson interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def supported_cutoff(y_sel: np.ndarray, p_sel: np.ndarray, rule: dict[str, Any], min_resistant: int) -> dict[str, Any]:
    """The protocol's cut-off rule on a selection set, with what the plan's section 7 requires beside it.

    The cut-off is always computed and applied; `supported` says whether the selection set held enough
    resistant spectra for its sensitivity target to mean anything.
    """
    y_sel = np.asarray(y_sel).astype(np.int64)
    threshold = float(choose_threshold(y_sel, p_sel, rule))
    positives = int(y_sel.sum())
    caught = int(((np.asarray(p_sel) >= threshold) & (y_sel == 1)).sum())
    low, high = wilson(caught, positives)
    return {"threshold": threshold, "selection_n": int(y_sel.size), "selection_resistant": positives,
            "selection_sensitivity": caught / positives if positives else float("nan"),
            "selection_sensitivity_low": low, "selection_sensitivity_high": high,
            "supported": positives >= min_resistant}


def cross_fitted_probabilities(model: Any, spec: FamilySpec, setting: dict[str, Any], X: np.ndarray,
                               y: np.ndarray, folds: list, seed: int) -> np.ndarray:
    """The out-of-fold predictions `fit_calibrated` fitted its calibrator on, mapped through that calibrator.

    `CalibratedClassifierCV(ensemble=False)` computes these internally and discards them. They are recomputed
    here the same way — the same pipeline, the same folds, the same response method — and then **proved** to be
    the same: a sigmoid fitted on them must reproduce the model's own calibrator. That is what makes "the
    cut-off was chosen on the predictions the calibrator was fitted on" a checked fact rather than a hope.

    The response is the pipeline's `decision_function` (LightGBM's raw margin) when it has one, as scikit-learn
    prefers it, not `predict_proba`: mapping a probability through this calibrator gives a different scale.
    """
    pipeline = base_pipeline(spec, seed).set_params(**pipeline_params(spec, setting, seed))
    method = response_method(pipeline)
    raw = cross_val_predict(clone(pipeline), X, y, cv=folds, method=method)
    raw = raw[:, 1] if method == "predict_proba" else raw
    fitted = model.calibrated_classifiers_
    if len(fitted) != 1 or len(fitted[0].calibrators) != 1:
        raise DevelopmentError("Expected one calibrated classifier with one calibrator (ensemble=False, binary).")
    calibrator = fitted[0].calibrators[0]
    check = _SigmoidCalibration().fit(raw, y)
    same = all(np.isclose(getattr(check, k), getattr(calibrator, k), rtol=0, atol=1e-8) for k in ("a_", "b_"))
    if not same:
        raise DevelopmentError(f"The recomputed out-of-fold predictions do not reproduce the model's calibrator "
                               f"(a {check.a_:.10f} vs {calibrator.a_:.10f}, b {check.b_:.10f} vs "
                               f"{calibrator.b_:.10f}); a cut-off chosen on them would not be the model's.")
    return np.asarray(calibrator.predict(raw), dtype=np.float64)


def response_method(pipeline: Any) -> str:
    """The response scikit-learn's calibration uses: `decision_function` if the pipeline has it."""
    return "decision_function" if hasattr(pipeline, "decision_function") else "predict_proba"


def assert_same_scale(model: Any, X: np.ndarray) -> None:
    """The model's calibrated output equals its calibrator applied to its fitted pipeline's response.

    Run on training rows. It proves that a cut-off chosen on cross-fitted predictions mapped through the
    calibrator sits on the same probability scale the final model outputs.
    """
    fitted = model.calibrated_classifiers_[0]
    pipeline, calibrator = fitted.estimator, fitted.calibrators[0]
    method = response_method(pipeline)
    raw = getattr(pipeline, method)(X)
    raw = raw[:, 1] if method == "predict_proba" else raw
    if not np.allclose(model.predict_proba(X)[:, 1], calibrator.predict(raw), rtol=0, atol=1e-12):
        raise DevelopmentError("The model's calibrated output is not its calibrator applied to its response; a "
                               "cut-off chosen on cross-fitted predictions would be on another scale.")


# --- the development log -----------------------------------------------------------------------------------------

@dataclass
class DevelopmentLog:
    """Append-only development log, written through the strict pre-write gate, never the production log."""

    path: Path
    production_log: Path

    def __post_init__(self) -> None:
        self.path, self.production_log = Path(self.path), Path(self.production_log)
        if self.path.resolve() == self.production_log.resolve():
            raise DevelopmentError(f"The development log is configured as the production test log "
                                   f"({self.production_log}); development results may never be written there.")

    def append(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """The seven pre-write checks, the append, then a byte-prefix check. Returns the pre-write state."""
        if not self.path.is_file():                  # first use: an empty log with the columns, nothing else
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=DEV_LOG_COLUMNS).to_csv(self.path, index=False, lineterminator="\n")
        before = self.path.read_bytes()
        current = pd.read_csv(self.path)
        if current.columns.tolist() != DEV_LOG_COLUMNS:
            raise DevelopmentError(f"{self.path} has unexpected columns; refusing to append.")
        frame = pd.DataFrame(rows)
        missing = [c for c in DEV_LOG_COLUMNS if c not in frame.columns]
        if missing:
            raise DevelopmentError(f"Development-log rows lack column(s) {missing}.")
        state = assert_log_ready_for_append(self.path, frame[DEV_LOG_COLUMNS].to_dict("records"),
                                            expected_sha256=hashlib.sha256(before).hexdigest(),
                                            expected_rows=int(len(current)),
                                            committed=current if len(current) else None)
        frame[DEV_LOG_COLUMNS].to_csv(self.path, mode="a", header=False, index=False, lineterminator="\n")
        if not self.path.read_bytes().startswith(before):
            raise DevelopmentError("The development log no longer begins with its earlier bytes.")
        if len(pd.read_csv(self.path)) != state["expected_rows_after"]:
            raise DevelopmentError("The development log does not hold the expected number of rows.")
        return state


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
