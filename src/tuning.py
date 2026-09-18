"""Version 0.4: hyperparameter search, calibration and caching (docs/v0.4_search_plan.md).

Everything here works on the training part of one split only:
- patient-grouped stratified folds,
- a grid / random search scored by cross-validated AUROC,
- the winning setting refitted on the whole training part with sigmoid calibration fitted on the
  out-of-fold predictions of the same folds.
Validation and test data never enter these functions.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, ParameterSampler, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FAMILY_KINDS = ("logistic_regression", "random_forest", "lightgbm", "svm_rbf")
FEATURE_PATTERN = re.compile(r"^(bins_3da|bins_18da|pca_(\d+)|kbest_(\d+))$")
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}


class TuningError(ValueError):
    pass


class CoarsenBins(TransformerMixin, BaseEstimator):     # TransformerMixin first: scikit-learn tag order
    """Sum every `factor` neighbouring bins (3 Da x 6 = 18 Da). Nothing is learned from the data."""

    def __init__(self, factor: int = 6):
        self.factor = factor

    def fit(self, X, y=None):
        n = np.asarray(X).shape[1]
        if self.factor < 1 or n % self.factor:
            raise TuningError(f"Cannot group {n} bins into blocks of {self.factor}.")
        self.n_features_in_ = n
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.shape[1] != self.n_features_in_:
            raise TuningError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        return X.reshape(X.shape[0], -1, self.factor).sum(axis=2)


def feature_steps(label: str, seed: int) -> dict[str, Any]:
    """Pipeline steps ('coarsen', 'reduce') for a feature variant label."""
    match = FEATURE_PATTERN.fullmatch(str(label))      # fullmatch: '$' would also accept a trailing newline
    if not match:
        raise TuningError(f"Unknown feature variant {label!r} (bins_3da, bins_18da, pca_<n>, kbest_<k>)")
    steps: dict[str, Any] = {"coarsen": "passthrough", "reduce": "passthrough"}
    if label == "bins_18da":
        steps["coarsen"] = CoarsenBins(6)
    elif match.group(2):
        steps["reduce"] = PCA(n_components=int(match.group(2)), svd_solver="randomized", random_state=seed)
    elif match.group(3):
        steps["reduce"] = SelectKBest(f_classif, k=int(match.group(3)))
    return steps


def feature_label(params: dict[str, Any]) -> str:
    """Inverse of feature_steps for a parameter dict (families without these steps use all bins)."""
    reduce = params.get("reduce", "passthrough")
    if isinstance(reduce, PCA):
        return f"pca_{reduce.n_components}"
    if isinstance(reduce, SelectKBest):
        return f"kbest_{reduce.k}"
    return "bins_18da" if isinstance(params.get("coarsen"), CoarsenBins) else "bins_3da"


@dataclass(frozen=True)
class FamilySpec:
    name: str
    kind: str
    grid: tuple = ()                               # list of {setting: [values]} dicts
    space: dict[str, Any] = field(default_factory=dict)
    random_draws: int = 0
    fixed: dict[str, Any] = field(default_factory=dict)
    stochastic: bool = False
    n_jobs: int = 1

    @classmethod
    def from_config(cls, name: str, cfg: dict[str, Any]) -> FamilySpec:
        kind = cfg.get("kind")
        if kind not in FAMILY_KINDS:
            raise TuningError(f"Family {name!r}: unknown kind {kind!r} (known: {', '.join(FAMILY_KINDS)})")
        grid, space, draws = tuple(cfg.get("grid") or ()), dict(cfg.get("space") or {}), int(cfg.get("random_draws", 0))
        if bool(grid) == bool(space):
            raise TuningError(f"Family {name!r}: give either 'grid' or 'space' (with 'random_draws').")
        if space and draws < 1:
            raise TuningError(f"Family {name!r}: 'space' needs random_draws >= 1.")
        return cls(name, kind, grid, space, draws, dict(cfg.get("fixed") or {}), bool(cfg.get("stochastic", False)),
                   int(cfg.get("n_jobs", 1)))

    def fingerprint(self) -> str:
        payload = json.dumps({"kind": self.kind, "grid": list(self.grid), "space": self.space,
                              "draws": self.random_draws, "fixed": self.fixed}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def family_specs(config: dict[str, Any]) -> list[FamilySpec]:
    return [FamilySpec.from_config(name, cfg) for name, cfg in config["tuning"]["families"].items()]


def base_pipeline(spec: FamilySpec, seed: int, n_jobs: int = -1) -> Pipeline:
    """Untuned pipeline of a family; `spec.fixed` overrides the defaults set here."""
    def params(**defaults: Any) -> dict[str, Any]:
        return {**defaults, **spec.fixed}

    if spec.kind == "logistic_regression":
        return Pipeline([("coarsen", "passthrough"), ("scale", StandardScaler()), ("reduce", "passthrough"),
                         ("model", LogisticRegression(**params(max_iter=5000, random_state=seed)))])
    if spec.kind == "random_forest":
        return Pipeline([("model", RandomForestClassifier(**params(random_state=seed, n_jobs=n_jobs)))])
    if spec.kind == "lightgbm":
        from lightgbm import LGBMClassifier  # optional dependency

        return Pipeline([("model", LGBMClassifier(**params(random_state=seed, n_jobs=n_jobs, verbose=-1,
                                                           deterministic=True, force_col_wise=True)))])
    if spec.kind == "svm_rbf":
        return Pipeline([("scale", StandardScaler()), ("model", SVC(**params(kernel="rbf", random_state=seed)))])
    raise TuningError(f"Unknown family kind {spec.kind!r}")


# scikit-learn >= 1.8 selects the penalty with l1_ratio (the `penalty` argument is deprecated there).
PENALTIES = {"l1": {"model__l1_ratio": 1.0, "model__solver": "liblinear"},
             "l2": {"model__l1_ratio": 0.0, "model__solver": "lbfgs"}}


def pipeline_params(spec: FamilySpec, setting: dict[str, Any], seed: int) -> dict[str, Any]:
    """Translate a readable setting (features, penalty, C, ...) into scikit-learn pipeline parameters."""
    params: dict[str, Any] = {}
    for key, value in setting.items():
        if key == "features":
            if spec.kind != "logistic_regression":
                raise TuningError("Feature variants are only defined for logistic regression.")
            params.update(feature_steps(value, seed))
        elif key == "penalty":
            if spec.kind != "logistic_regression" or value not in PENALTIES:
                raise TuningError(f"Unsupported penalty {value!r} (l1 or l2, logistic regression only).")
            params.update(PENALTIES[value])
        else:
            params[f"model__{key}"] = value
    return params


def candidates(spec: FamilySpec, seed: int) -> list[dict[str, Any]]:
    """Every setting to score, in a fixed order, as readable dicts (the order decides ties)."""
    if spec.grid:
        return [dict(p) for part in spec.grid for p in ParameterGrid(dict(part))]
    return [dict(p) for p in ParameterSampler(spec.space, n_iter=spec.random_draws, random_state=seed)]


def grouped_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified folds that keep every patient group in one fold; indices are positions in y."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(y)), y, groups))
    for _, test in folds:
        if np.unique(y[test]).size < 2:
            raise TuningError("A cross-validation fold has only one class.")
    return folds


@dataclass
class SearchResult:
    family: str
    table: pd.DataFrame                  # one row per setting, in candidate order
    best_index: int
    best_setting: dict[str, Any]
    seconds: float

    def to_json(self) -> dict[str, Any]:
        return {"family": self.family, "best_index": self.best_index, "best_setting": self.best_setting,
                "seconds": self.seconds}


def run_search(spec: FamilySpec, X: np.ndarray, y: np.ndarray, folds: list, seed: int,
               metric: str = "roc_auc") -> SearchResult:
    """Score every candidate setting by cross-validation on (X, y); X and y are the training part only."""
    settings = candidates(spec, seed)
    grid = [{k: [v] for k, v in pipeline_params(spec, s, seed).items()} for s in settings]
    inner_jobs = 1 if spec.n_jobs > 1 else -1                # avoid threads x processes oversubscription
    search = GridSearchCV(base_pipeline(spec, seed, n_jobs=inner_jobs), grid, scoring=SCORING, refit=False,
                          cv=folds, n_jobs=spec.n_jobs, error_score="raise")
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                      # convergence/constant-feature notes of single fits
        search.fit(X, y)
    seconds = time.perf_counter() - started
    res = search.cv_results_
    rows = []
    for i, setting in enumerate(settings):
        row = {"setting": i, **{k: ("none" if v is None else v) for k, v in setting.items()}}
        if spec.kind == "logistic_regression":
            row["features"] = setting.get("features", "bins_3da")
        for m in SCORING:
            row[f"cv_{m}_mean"] = float(res[f"mean_test_{m}"][i])
            row[f"cv_{m}_sd"] = float(res[f"std_test_{m}"][i])
        row["fit_seconds_mean"] = float(res["mean_fit_time"][i])
        rows.append(row)
    table = pd.DataFrame(rows)
    scores = table[f"cv_{metric}_mean"].to_numpy()
    if not np.isfinite(scores).any():
        raise TuningError(f"{spec.name}: no finite cross-validated {metric}.")
    best = int(np.nanargmax(scores))                         # first maximum -> earlier setting wins ties
    table["rank"] = pd.Series(scores).rank(ascending=False, method="min", na_option="bottom").astype(int).to_numpy()
    return SearchResult(spec.name, table, best, settings[best], seconds)


def fit_calibrated(spec: FamilySpec, setting: dict[str, Any], X: np.ndarray, y: np.ndarray, folds: list,
                   seed: int, method: str = "sigmoid") -> CalibratedClassifierCV:
    """Fit the setting on all of (X, y); calibrate on the out-of-fold predictions of `folds`."""
    pipeline = base_pipeline(spec, seed).set_params(**pipeline_params(spec, setting, seed))
    model = CalibratedClassifierCV(pipeline, method=method, cv=folds, ensemble=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return model.fit(X, y)


def uncalibrated(model: CalibratedClassifierCV) -> Pipeline:
    """The pipeline inside a fitted calibrated model (trained on the whole training part)."""
    fitted = model.calibrated_classifiers_
    if len(fitted) != 1:
        raise TuningError(f"Expected one calibrated classifier (ensemble=False), found {len(fitted)}.")
    return fitted[0].estimator


def converged(model: CalibratedClassifierCV) -> bool | None:
    """False if an iterative model stopped at its iteration limit; None if not applicable."""
    est = uncalibrated(model).steps[-1][1]
    if isinstance(est, LogisticRegression):
        return bool(np.all(est.n_iter_ < est.max_iter))
    return None


def set_threads(obj: Any, n_jobs: int) -> None:
    """Set n_jobs on every estimator inside pipelines and calibrated models (predictions do not change)."""
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            set_threads(step, n_jobs)
    for cc in getattr(obj, "calibrated_classifiers_", []):
        set_threads(cc.estimator, n_jobs)
    if hasattr(obj, "get_params") and "n_jobs" in obj.get_params(deep=False):
        obj.set_params(n_jobs=n_jobs)
        if isinstance(obj, CalibratedClassifierCV) and obj.estimator is not None:
            set_threads(obj.estimator, n_jobs)


def code_fingerprint(paths: list[str | Path]) -> str:
    """Hash of the code files that determine a result (used to invalidate caches)."""
    digest = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(f"{path.name}:".encode())            # separator: file names cannot merge into contents
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:16]


def cache_key(**parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


def cached(path: Path, key: str, compute, log=None):
    """Return the object stored at `path` if it was made with `key`; otherwise compute and store it."""
    path = Path(path)
    if path.is_file():
        try:
            stored = joblib.load(path)
            if isinstance(stored, dict) and stored.get("key") == key:
                if log is not None:
                    log.info("using cached %s", path.name)
                return stored["value"]
        except Exception:                                  # damaged cache file: recompute
            pass
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"key": key, "value": value}, path, compress=3)
    return value
