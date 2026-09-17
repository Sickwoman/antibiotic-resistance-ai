"""Evaluation as fixed in docs/evaluation_protocol.md: thresholds, metrics, calibration, bootstrap
intervals and the append-only log of test-set evaluations.

Label 1 = resistant (the positive class), label 0 = susceptible. "Sensitivity" and "recall" are the
same number (share of resistant isolates that are flagged); both names are reported because both are
commonly asked for. PR-AUC is estimated as average precision.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

THRESHOLD_FREE = ("roc_auc", "pr_auc", "brier")
AT_THRESHOLD = ("accuracy", "balanced_accuracy", "sensitivity", "recall", "specificity", "precision", "f1")
COUNTS = ("tp", "fp", "tn", "fn")
INTERVAL_METRICS = ("roc_auc", "pr_auc", "brier", "sensitivity", "specificity", "balanced_accuracy")
TEST_LOG_COLUMNS = ["logged_at", "git_commit", "stage", "dataset", "dataset_fingerprint", "x_sha256", "experiment",
                    "split", "model", "seed", "train_size", "n", "n_resistant", "threshold", "roc_auc", "pr_auc",
                    "brier", "sensitivity", "specificity", "precision", "f1", "accuracy", "balanced_accuracy",
                    "tp", "fp", "tn", "fn"]


class EvaluationError(ValueError):
    pass


def _check(y: Iterable[int], prob: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=np.float64)
    if y.ndim != 1 or prob.ndim != 1 or y.size != prob.size:
        raise EvaluationError(f"Labels and probabilities must be 1-D and equally long ({y.shape} vs {prob.shape}).")
    if y.size == 0:
        raise EvaluationError("Nothing to evaluate (0 samples).")
    if not np.isin(y, (0, 1)).all():
        raise EvaluationError("Labels must be 0 (susceptible) or 1 (resistant).")
    if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
        raise EvaluationError("Probabilities must be finite and between 0 and 1.")
    return y.astype(np.int64), prob


def threshold_for_sensitivity(y: Iterable[int], prob: Iterable[float], min_sensitivity: float) -> float:
    """Highest cut-off t such that 'resistant if prob >= t' flags at least `min_sensitivity` of the resistant
    samples. Use validation data only."""
    y, prob = _check(y, prob)
    if not 0 < min_sensitivity <= 1:
        raise EvaluationError(f"min_sensitivity must be in (0, 1], got {min_sensitivity}")
    positives = np.sort(prob[y == 1])[::-1]
    if positives.size == 0:
        raise EvaluationError("No resistant samples: a sensitivity-based threshold cannot be chosen.")
    k = int(np.ceil(min_sensitivity * positives.size - 1e-9))   # resistant samples that must be flagged
    return float(positives[k - 1])


def choose_threshold(y: Iterable[int], prob: Iterable[float], rule: dict[str, Any]) -> float:
    if rule.get("rule") != "min_sensitivity":
        raise EvaluationError(f"Unknown threshold rule {rule.get('rule')!r} (the protocol uses 'min_sensitivity').")
    return threshold_for_sensitivity(y, prob, float(rule["min_sensitivity"]))


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def confusion_counts(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {"tp": int(np.sum((pred == 1) & (y == 1))), "fp": int(np.sum((pred == 1) & (y == 0))),
            "tn": int(np.sum((pred == 0) & (y == 0))), "fn": int(np.sum((pred == 0) & (y == 1)))}


def _fit_logistic(design: np.ndarray, y: np.ndarray, offset: np.ndarray, max_iter: int = 100) -> np.ndarray | None:
    """Newton-Raphson for a small logistic model with an offset; None if it does not converge."""
    beta = np.zeros(design.shape[1])
    for _ in range(max_iter):
        mu = expit(offset + design @ beta)
        w = mu * (1 - mu)
        grad = design.T @ (y - mu)
        hess = (design * w[:, None]).T @ design
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if not np.isfinite(beta).all() or np.abs(beta).max() > 1e6:
            return None
        if np.abs(step).max() < 1e-10:
            return beta
    return None


def calibration_slope_intercept(y: Iterable[int], prob: Iterable[float]) -> tuple[float, float]:
    """Slope of logit(p) in a logistic recalibration model (1 = ideal) and the calibration-in-the-large
    intercept with the slope fixed at 1 (0 = ideal). NaN when undefined (one class, constant p)."""
    y, prob = _check(y, prob)
    if y.min() == y.max():
        return float("nan"), float("nan")
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    ones = np.ones((y.size, 1))
    intercept = _fit_logistic(ones, y, z)
    slope = None if np.ptp(z) < 1e-9 else _fit_logistic(np.column_stack([ones, z]), y, np.zeros(y.size))
    return (float(slope[1]) if slope is not None else float("nan"),
            float(intercept[0]) if intercept is not None else float("nan"))


def classification_metrics(y: Iterable[int], prob: Iterable[float], threshold: float) -> dict[str, Any]:
    """All metrics of the protocol for one set of predictions and a fixed threshold."""
    y, prob = _check(y, prob)
    pred = (prob >= threshold).astype(np.int64)
    c = confusion_counts(y, pred)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    sensitivity = _ratio(c["tp"], n_pos)
    specificity = _ratio(c["tn"], n_neg)
    both = n_pos > 0 and n_neg > 0
    slope, intercept = calibration_slope_intercept(y, prob)
    return {
        "n": int(y.size), "n_resistant": n_pos, "prevalence": n_pos / y.size, "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, prob)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, prob)) if n_pos else float("nan"),
        "brier": float(np.mean((prob - y) ** 2)),
        "accuracy": (c["tp"] + c["tn"]) / y.size,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity, "recall": sensitivity, "specificity": specificity,
        "precision": _ratio(c["tp"], c["tp"] + c["fp"]),
        "f1": _ratio(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"]),
        **c,
        "calibration_slope": slope, "calibration_intercept": intercept,
    }


def reliability_table(y: Iterable[int], prob: Iterable[float], n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted probability with the observed resistance rate in each."""
    y, prob = _check(y, prob)
    order = np.argsort(prob, kind="stable")
    rows = []
    for i, idx in enumerate(np.array_split(order, min(n_bins, y.size))):
        rows.append({"bin": i + 1, "n": int(idx.size), "mean_predicted": float(prob[idx].mean()),
                     "observed_rate": float(y[idx].mean()), "min_predicted": float(prob[idx].min()),
                     "max_predicted": float(prob[idx].max())})
    return pd.DataFrame(rows)


def _metric_values(y: np.ndarray, prob: np.ndarray, threshold: float, metrics: Iterable[str]) -> dict[str, float]:
    pred = prob >= threshold
    tp = np.sum(pred & (y == 1))
    tn = np.sum(~pred & (y == 0))
    n_pos, n_neg = y.sum(), y.size - y.sum()
    out = {}
    for m in metrics:
        if m == "roc_auc":
            out[m] = roc_auc_score(y, prob)
        elif m == "pr_auc":
            out[m] = average_precision_score(y, prob)
        elif m == "brier":
            out[m] = np.mean((prob - y) ** 2)
        elif m == "sensitivity":
            out[m] = tp / n_pos
        elif m == "specificity":
            out[m] = tn / n_neg
        elif m == "balanced_accuracy":
            out[m] = (tp / n_pos + tn / n_neg) / 2
        else:
            raise EvaluationError(f"No bootstrap support for metric {m!r}")
    return {k: float(v) for k, v in out.items()}


def group_resample_indices(order: np.ndarray, starts: np.ndarray, counts: np.ndarray,
                           picked: np.ndarray) -> np.ndarray:
    """Row indices of the picked groups (with repeats). `order` sorts rows by group; `starts`/`counts`
    locate each group in that order."""
    lens = counts[picked]
    offsets = np.repeat(np.cumsum(lens) - lens, lens)
    return order[np.repeat(starts[picked], lens) + np.arange(lens.sum()) - offsets]


def bootstrap(y: Iterable[int], groups: Iterable[Any], probs: dict[str, np.ndarray], thresholds: dict[str, float], *,
              resamples: int = 2000, seed: int = 42, metrics: Iterable[str] = INTERVAL_METRICS
              ) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    """Resample whole patient groups; returns per-model metric samples and the number of skipped
    resamples (those containing only one class). Every model sees the same resamples (paired)."""
    y = np.asarray(y).astype(np.int64)
    groups = np.asarray(groups)
    metrics = list(metrics)
    for name, p in probs.items():
        _check(y, p)
        if name not in thresholds:
            raise EvaluationError(f"No threshold for model {name!r}")
    order = np.argsort(groups, kind="stable")
    _, starts, counts = np.unique(groups[order], return_index=True, return_counts=True)
    rng = np.random.default_rng(seed)
    samples: dict[str, dict[str, list[float]]] = {name: {m: [] for m in metrics} for name in probs}
    skipped = 0
    for _ in range(resamples):
        idx = group_resample_indices(order, starts, counts, rng.integers(0, starts.size, starts.size))
        yb = y[idx]
        if yb.min() == yb.max():
            skipped += 1
            continue
        for name, p in probs.items():
            for m, v in _metric_values(yb, np.asarray(p, dtype=np.float64)[idx], thresholds[name], metrics).items():
                samples[name][m].append(v)
    return {name: {m: np.asarray(v) for m, v in d.items()} for name, d in samples.items()}, skipped


def interval(values: np.ndarray, level: float) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    alpha = (1 - level) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def summarize_bootstrap(y: Iterable[int], groups: Iterable[Any], probs: dict[str, np.ndarray],
                        thresholds: dict[str, float], *, resamples: int, level: float, seed: int,
                        reference: str | None = None, metrics: Iterable[str] = INTERVAL_METRICS
                        ) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Point estimates with percentile intervals, plus paired differences (reference minus model)."""
    metrics = list(metrics)
    y_arr = np.asarray(y).astype(np.int64)
    samples, skipped = bootstrap(y_arr, groups, probs, thresholds, resamples=resamples, seed=seed, metrics=metrics)
    points = {name: _metric_values(y_arr, np.asarray(p, dtype=np.float64), thresholds[name], metrics)
              for name, p in probs.items()}
    out: dict[str, Any] = {"unit": "patient group (a spectrum without patient ID is its own group)",
                           "resamples": resamples, "used": resamples - skipped, "skipped_single_class": skipped,
                           "level": level, "seed": seed, "reference": reference, "intervals": {},
                           "differences_to_reference": {}}
    for name in probs:
        out["intervals"][name] = {m: dict(zip(("low", "high"), interval(samples[name][m], level), strict=True),
                                          estimate=points[name][m]) for m in metrics}
        if reference is not None and name != reference:
            out["differences_to_reference"][name] = {
                m: dict(zip(("low", "high"), interval(samples[reference][m] - samples[name][m], level), strict=True),
                        estimate=points[reference][m] - points[name][m]) for m in metrics}
    return out, samples


def unpaired_difference(a: np.ndarray, b: np.ndarray, level: float, seed: int = 42) -> tuple[float, float]:
    """Interval for mean(a-type) - mean(b-type) from two independent bootstrap sample sets."""
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = min(a.size, b.size)
    diff = rng.choice(a, n, replace=False) - rng.choice(b, n, replace=False)
    return interval(diff, level)


def append_test_log(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append one row per test-set evaluation. The file is never rewritten, only extended."""
    if not rows:
        return
    path = Path(path)
    frame = pd.DataFrame(rows)
    missing = [c for c in TEST_LOG_COLUMNS if c not in frame.columns]
    if missing:
        raise EvaluationError(f"Test-log rows lack column(s) {missing}")
    frame = frame[TEST_LOG_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if header != TEST_LOG_COLUMNS:
            raise EvaluationError(f"{path} has different columns; refusing to append.")
        frame.to_csv(path, mode="a", header=False, index=False, lineterminator="\n")
    else:
        frame.to_csv(path, index=False, lineterminator="\n")
