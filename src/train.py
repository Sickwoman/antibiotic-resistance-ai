"""Version 0.3 baseline models: construction and training with fixed, untuned settings.

Every learned step (StandardScaler, the classifier) lives in one scikit-learn Pipeline that is fitted on
training rows only. The decision threshold is chosen on the validation part. Test rows are only scored
when the caller passes them explicitly (see scripts/train_baselines.py --evaluate-test).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import choose_threshold, classification_metrics

MODEL_KINDS = ("dummy", "logistic_regression", "random_forest", "lightgbm")


class TrainingError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    scale: bool = False          # StandardScaler before the model (fitted on training rows only)
    stochastic: bool = False     # True: trained once per seed; False: one run is enough

    @classmethod
    def from_config(cls, name: str, cfg: dict[str, Any]) -> ModelSpec:
        kind = cfg.get("kind")
        if kind not in MODEL_KINDS:
            raise TrainingError(f"Model {name!r}: unknown kind {kind!r} (known: {', '.join(MODEL_KINDS)})")
        return cls(name, kind, dict(cfg.get("params") or {}), bool(cfg.get("scale", False)),
                   bool(cfg.get("stochastic", False)))


def model_specs(config: dict[str, Any]) -> list[ModelSpec]:
    return [ModelSpec.from_config(name, cfg) for name, cfg in config["baselines"]["models"].items()]


def build_pipeline(spec: ModelSpec, seed: int, n_jobs: int = -1) -> Pipeline:
    if spec.kind == "dummy":
        model = DummyClassifier(strategy="prior", **spec.params)          # predicts the training resistance rate
    elif spec.kind == "logistic_regression":
        model = LogisticRegression(random_state=seed, **spec.params)
    elif spec.kind == "random_forest":
        model = RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **spec.params)
    elif spec.kind == "lightgbm":
        from lightgbm import LGBMClassifier  # optional dependency

        model = LGBMClassifier(random_state=seed, n_jobs=n_jobs, verbose=-1, deterministic=True,
                               force_col_wise=True, **spec.params)
    else:
        raise TrainingError(f"Unknown model kind {spec.kind!r}")
    steps = [("scale", StandardScaler())] if spec.scale else []
    return Pipeline([*steps, ("model", model)])


def load_rows(X: np.ndarray, rows: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Copy the selected rows of a (memory-mapped) matrix into RAM, a chunk at a time."""
    rows = np.asarray(rows, dtype=np.int64)
    out = np.empty((rows.size, X.shape[1]), dtype=X.dtype)
    for start in range(0, rows.size, chunk):
        out[start:start + chunk] = X[rows[start:start + chunk]]
    return out


def grouped_subsample(meta: pd.DataFrame, rows: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Random whole patient groups from `rows`, adding groups while the total stays <= `size`."""
    rows = np.asarray(rows, dtype=np.int64)
    if not 0 < size <= rows.size:
        raise TrainingError(f"Subsample size {size} must be between 1 and {rows.size}")
    groups = meta["group_id"].to_numpy()[rows]
    uniq, counts = np.unique(groups, return_counts=True)
    size_of = dict(zip(uniq.tolist(), counts.tolist(), strict=True))
    chosen, total = [], 0
    for g in np.random.default_rng(seed).permutation(uniq).tolist():
        if total + size_of[g] <= size:
            chosen.append(g)
            total += size_of[g]
            if total == size:
                break
    return np.sort(rows[np.isin(groups, chosen)])


@dataclass
class RunResult:
    experiment: str
    split: str
    model: str
    seed: int
    train_size: int
    fit_seconds: float
    threshold: float
    validation: dict[str, Any]
    pipeline: Pipeline
    validation_prob: np.ndarray
    test_prob: np.ndarray | None = None
    test: dict[str, Any] | None = None
    predict_ms_per_sample: float = float("nan")

    def row(self, part: str) -> dict[str, Any]:
        metrics = self.validation if part == "validation" else self.test
        return {"experiment": self.experiment, "split": self.split, "model": self.model, "seed": self.seed,
                "train_size": self.train_size, "fit_seconds": round(self.fit_seconds, 2),
                "predict_ms_per_sample": round(self.predict_ms_per_sample, 4), **(metrics or {})}


def run_experiment(X: np.ndarray, meta: pd.DataFrame, *, experiment: str, split: str, train_rows: np.ndarray,
                   validation_rows: np.ndarray, test_rows: np.ndarray | None, specs: list[ModelSpec],
                   seeds: list[int], threshold_rule: dict[str, Any], log=None) -> list[RunResult]:
    """Train every model on `train_rows`; choose its threshold on `validation_rows`; score `test_rows`
    only if given. Stochastic models are trained once per seed, the others once (first seed)."""
    y = meta["label"].to_numpy().astype(np.int64)
    X_train, y_train = load_rows(X, train_rows), y[train_rows]
    X_val, y_val = load_rows(X, validation_rows), y[validation_rows]
    X_test = load_rows(X, test_rows) if test_rows is not None else None
    if np.unique(y_train).size < 2 or np.unique(y_val).size < 2:
        raise TrainingError(f"{experiment}: training and validation parts need both classes.")
    results = []
    for spec in specs:
        for seed in (seeds if spec.stochastic else seeds[:1]):
            started = time.perf_counter()
            pipeline = build_pipeline(spec, seed).fit(X_train, y_train)
            fit_seconds = time.perf_counter() - started
            val_prob = pipeline.predict_proba(X_val)[:, 1]
            threshold = choose_threshold(y_val, val_prob, threshold_rule)
            result = RunResult(experiment, split, spec.name, seed, int(len(train_rows)), fit_seconds, threshold,
                               classification_metrics(y_val, val_prob, threshold), pipeline, val_prob)
            if X_test is not None:
                started = time.perf_counter()
                result.test_prob = pipeline.predict_proba(X_test)[:, 1]
                result.predict_ms_per_sample = (time.perf_counter() - started) * 1000 / max(len(test_rows), 1)
                result.test = classification_metrics(y[test_rows], result.test_prob, threshold)
            if log is not None:
                log.info("%s: %s (seed %d) trained on %d samples in %.1f s; validation AUROC %.3f",
                         experiment, spec.name, seed, len(train_rows), fit_seconds, result.validation["roc_auc"])
            results.append(result)
    return results


def select_best(results: list[RunResult], metric: str, seed: int) -> RunResult:
    """Model with the highest validation `metric` among the runs with `seed` (ties: first in config order)."""
    candidates = [r for r in results if r.seed == seed and np.isfinite(r.validation[metric])]
    if not candidates:
        raise TrainingError(f"No run with seed {seed} has a finite validation {metric}.")
    return max(candidates, key=lambda r: r.validation[metric])
