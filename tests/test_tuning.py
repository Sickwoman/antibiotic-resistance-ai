"""Tests for the Version 0.4 search, calibration and caching helpers (synthetic data only)."""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import warnings

import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

import src.tuning as tuning
from src.tuning import (
    PENALTIES,
    CoarsenBins,
    FamilySpec,
    SearchResult,
    TuningError,
    base_pipeline,
    cache_key,
    cached,
    candidates,
    code_fingerprint,
    converged,
    family_specs,
    feature_label,
    feature_steps,
    fit_calibrated,
    grouped_folds,
    pipeline_params,
    run_search,
    set_threads,
    uncalibrated,
)
from src.utils import load_config


def synthetic(n: int = 240, p: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spectra-like matrix with four informative columns and about two rows per patient group."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.35).astype(int)
    X = rng.normal(size=(n, p))
    X[:, :4] += y[:, None] * 1.2
    groups = rng.integers(0, n // 2, n)
    return X, y, groups


# Tiny families (settings are passed explicitly where it matters). Forests and boosting use one thread:
# on tiny data, all-core threading only adds overhead (and is very slow while another job uses the CPU).
LR = FamilySpec("lr", "logistic_regression", grid=({"C": [1.0]},))
RF = FamilySpec("rf", "random_forest", grid=({"min_samples_leaf": [3]},), fixed={"n_estimators": 20, "n_jobs": 1})
SVM = FamilySpec("svm", "svm_rbf", grid=({"C": [1.0]},))
LGBM = FamilySpec("lgbm", "lightgbm", space={"n_estimators": [20, 40], "num_leaves": [3, 7], "subsample": [0.7, 1.0]},
                  random_draws=3, fixed={"subsample_freq": 1, "min_child_samples": 5, "n_jobs": 1})


@pytest.fixture(scope="module")
def train():
    X, y, groups = synthetic()
    return X, y, groups, grouped_folds(y, groups, 4, 42)


@pytest.fixture(scope="module")
def holdout():
    X, y, _ = synthetic(n=200, seed=1)
    return X, y


@pytest.fixture(scope="module")
def project_config():
    return load_config()


# --- CoarsenBins -------------------------------------------------------------------------------------------------

def test_coarsen_bins_sums_neighbouring_blocks():
    X = np.arange(12, dtype=float).reshape(2, 6)
    np.testing.assert_array_equal(CoarsenBins(3).fit_transform(X), [[3, 12], [21, 30]])
    np.testing.assert_array_equal(CoarsenBins(2).fit(X).transform(X), [[1, 5, 9], [13, 17, 21]])
    np.testing.assert_array_equal(CoarsenBins().fit_transform(X), [[15], [51]])      # default: 6 bins (18 Da)
    np.testing.assert_array_equal(CoarsenBins(1).fit_transform(X), X)
    # nothing is learned: fitting on other rows of the same width gives the same output
    np.testing.assert_array_equal(CoarsenBins(3).fit(np.zeros((1, 6))).transform(X), [[3, 12], [21, 30]])
    spectra = np.random.default_rng(0).random((5, 36))
    coarse = CoarsenBins(6).fit_transform(spectra)
    assert coarse.shape == (5, 6)
    np.testing.assert_allclose(coarse.sum(axis=1), spectra.sum(axis=1))                  # total intensity kept


@pytest.mark.parametrize("factor, width", [(4, 6), (5, 12), (6, 6001), (0, 6), (-2, 6)])
def test_coarsen_bins_rejects_bad_factors(factor, width):
    with pytest.raises(TuningError, match="Cannot group"):
        CoarsenBins(factor).fit(np.ones((2, width)))


def test_coarsen_bins_rejects_another_width_at_transform():
    step = CoarsenBins(3).fit(np.ones((2, 6)))
    with pytest.raises(TuningError, match="Expected 6 features, got 9"):
        step.transform(np.ones((2, 9)))
    assert issubclass(TuningError, ValueError)


def test_coarsen_bins_inside_a_pipeline(train):
    X, y, _, _ = train                                                   # 30 columns
    pipe = Pipeline([("coarsen", CoarsenBins(6)), ("model", LogisticRegression())]).fit(X, y)
    assert pipe.named_steps["model"].coef_.shape == (1, 5)
    assert pipe.predict_proba(X).shape == (len(y), 2)
    assert pipe.set_params(coarsen__factor=5).fit(X, y).named_steps["model"].coef_.shape == (1, 6)
    assert clone(CoarsenBins(3)).get_params() == {"factor": 3}


# --- feature variants --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("label", ["bins_3da", "bins_18da", "pca_7", "pca_200", "kbest_5", "kbest_1000"])
def test_feature_steps_round_trip(label):
    steps = feature_steps(label, seed=3)
    assert set(steps) == {"coarsen", "reduce"}
    assert feature_label(steps) == label


def test_feature_steps_build_the_planned_steps():
    assert feature_steps("bins_3da", 0) == {"coarsen": "passthrough", "reduce": "passthrough"}
    coarse = feature_steps("bins_18da", 0)
    assert isinstance(coarse["coarsen"], CoarsenBins) and coarse["coarsen"].factor == 6
    assert coarse["reduce"] == "passthrough"
    pca = feature_steps("pca_50", 7)
    assert pca["coarsen"] == "passthrough" and isinstance(pca["reduce"], PCA)
    assert (pca["reduce"].n_components, pca["reduce"].random_state, pca["reduce"].svd_solver) == (50, 7, "randomized")
    kbest = feature_steps("kbest_300", 0)
    assert kbest["coarsen"] == "passthrough" and isinstance(kbest["reduce"], SelectKBest)
    assert kbest["reduce"].k == 300 and kbest["reduce"].score_func is f_classif
    assert feature_label({}) == "bins_3da"                               # families without these steps
    assert feature_label({"model__C": 1.0}) == "bins_3da"


@pytest.mark.parametrize("label", ["bins_6da", "pca_", "pca_x", "kbest", "PCA_50", "kbest_-5", "", " bins_3da",
                                   None, 50])
def test_feature_steps_rejects_unknown_labels(label):
    with pytest.raises(TuningError, match="Unknown feature variant"):
        feature_steps(label, 0)


# --- FamilySpec and candidates -----------------------------------------------------------------------------------

SVM_CFG = {"kind": "svm_rbf", "grid": [{"C": [1.0, 10.0], "gamma": [1e-3]}]}
SPACE_CFG = {"kind": "lightgbm", "space": {"num_leaves": [7, 15]}, "random_draws": 2}


@pytest.mark.parametrize("cfg, message", [
    ({"kind": "knn", "grid": [{"n_neighbors": [3]}]}, "unknown kind"),
    ({"grid": [{"C": [1.0]}]}, "unknown kind"),
    ({"kind": "svm_rbf"}, "either 'grid' or 'space'"),
    ({"kind": "svm_rbf", "grid": [], "space": {}}, "either 'grid' or 'space'"),
    ({**SPACE_CFG, "grid": [{"num_leaves": [7]}]}, "either 'grid' or 'space'"),
    ({"kind": "lightgbm", "space": {"num_leaves": [7, 15]}}, "random_draws >= 1"),
    ({**SPACE_CFG, "random_draws": 0}, "random_draws >= 1"),
])
def test_family_spec_validation(cfg, message):
    with pytest.raises(TuningError, match=message):
        FamilySpec.from_config("bad", cfg)


def test_family_spec_from_config_reads_every_field():
    cfg = {"kind": "random_forest", "grid": [{"min_samples_leaf": [1, 3]}], "fixed": {"n_estimators": 50},
           "stochastic": True, "n_jobs": 2}
    assert FamilySpec.from_config("forest", cfg) == FamilySpec(
        "forest", "random_forest", ({"min_samples_leaf": [1, 3]},), {}, 0, {"n_estimators": 50}, True, 2)
    svm = FamilySpec.from_config("svm", SVM_CFG)
    assert (svm.fixed, svm.stochastic, svm.n_jobs, svm.space, svm.random_draws) == ({}, False, 1, {}, 0)
    gbm = FamilySpec.from_config("gbm", SPACE_CFG)
    assert (gbm.grid, gbm.space, gbm.random_draws) == ((), {"num_leaves": [7, 15]}, 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        svm.n_jobs = 4


def test_family_spec_fingerprint_tracks_what_decides_the_results():
    base = FamilySpec.from_config("svm", SVM_CFG)
    fp = base.fingerprint()
    assert len(fp) == 16 and fp == FamilySpec.from_config("svm", copy.deepcopy(SVM_CFG)).fingerprint()
    # the name, thread count and seed-repeat flag do not change which settings are scored
    assert dataclasses.replace(base, name="other", n_jobs=8, stochastic=True).fingerprint() == fp
    new_value, reordered, new_key = (copy.deepcopy(SVM_CFG) for _ in range(3))
    new_value["grid"][0]["C"] = [1.0, 100.0]
    reordered["grid"][0]["C"] = [10.0, 1.0]                              # the order decides ties
    new_key["grid"][0]["class_weight"] = [None]
    changed = [new_value, reordered, new_key, {**SVM_CFG, "fixed": {"cache_size": 500}},
               {**SVM_CFG, "kind": "logistic_regression"}]
    fps = {FamilySpec.from_config("svm", cfg).fingerprint() for cfg in changed}
    assert len(fps) == len(changed) and fp not in fps
    assert (FamilySpec.from_config("g", SPACE_CFG).fingerprint()
            != FamilySpec.from_config("g", {**SPACE_CFG, "random_draws": 3}).fingerprint())


def test_family_specs_from_project_config(project_config):
    seed = project_config["tuning"]["cv_seed"]
    specs = {s.name: s for s in family_specs(project_config)}
    assert list(specs) == ["logistic_regression", "random_forest", "lightgbm", "svm_rbf"]
    assert [s.kind for s in specs.values()] == list(specs)
    counts = {name: len(candidates(spec, seed)) for name, spec in specs.items()}
    assert counts == {"logistic_regression": 100, "random_forest": 18, "lightgbm": 30, "svm_rbf": 36}
    assert specs["lightgbm"].space and not specs["lightgbm"].grid and specs["lightgbm"].random_draws == 30
    assert specs["random_forest"].stochastic and specs["lightgbm"].stochastic
    assert not specs["logistic_regression"].stochastic and not specs["svm_rbf"].stochastic
    # logistic regression: 6 feature variants x 7 C x 2 class weights with L2, then 2 x 4 x 2 with L1
    lr = candidates(specs["logistic_regression"], seed)
    assert [s["penalty"] for s in lr] == ["l2"] * 84 + ["l1"] * 16
    assert {s["features"] for s in lr[84:]} == {"bins_3da", "bins_18da"}


def test_every_project_candidate_maps_onto_its_pipeline(project_config):
    seed = project_config["tuning"]["cv_seed"]
    for spec in family_specs(project_config):
        settings = candidates(spec, seed)
        assert len({json.dumps(s, sort_keys=True) for s in settings}) == len(settings), spec.name   # no repeats
        pipe = base_pipeline(spec, seed)
        valid = set(pipe.get_params(deep=True))
        for setting in settings:
            params = pipeline_params(spec, setting, seed)
            assert set(params) <= valid, (spec.name, setting)
            pipe.set_params(**params)


def test_lightgbm_draws_are_reproducible(project_config):
    seed = project_config["tuning"]["cv_seed"]
    spec = {s.name: s for s in family_specs(project_config)}["lightgbm"]
    first = candidates(spec, seed)
    assert first == candidates(spec, seed)
    assert first != candidates(spec, seed + 1)
    for setting in first:
        assert set(setting) == set(spec.space)
        assert all(setting[k] in spec.space[k] for k in setting)


def test_grid_candidates_follow_parameter_grid_order():
    parts = ({"b": [2, 1], "a": ["x", "y"]}, {"c": [0.5]})
    spec = FamilySpec("svm", "svm_rbf", grid=parts)
    assert candidates(spec, 0) == [*ParameterGrid(parts[0]), {"c": 0.5}]
    assert candidates(spec, 0)[0] == {"a": "x", "b": 2}                  # listed values keep their order
    assert candidates(spec, 0) == candidates(spec, 99)                    # a grid does not depend on the seed


# --- base_pipeline and pipeline_params ---------------------------------------------------------------------------

def test_base_pipeline_defaults_and_fixed_overrides():
    lr = base_pipeline(LR, 3)
    assert [name for name, _ in lr.steps] == ["coarsen", "scale", "reduce", "model"]
    assert (lr.named_steps["model"].max_iter, lr.named_steps["model"].random_state) == (5000, 3)
    assert base_pipeline(dataclasses.replace(LR, fixed={"max_iter": 7}), 3).named_steps["model"].max_iter == 7
    rf = base_pipeline(dataclasses.replace(RF, fixed={"n_estimators": 20}), seed=1, n_jobs=4).named_steps["model"]
    assert (rf.n_estimators, rf.n_jobs, rf.random_state) == (20, 4, 1)
    rf = base_pipeline(dataclasses.replace(RF, fixed={"n_jobs": 2, "random_state": 9}), seed=1, n_jobs=4)
    assert (rf.named_steps["model"].n_jobs, rf.named_steps["model"].random_state) == (2, 9)
    svm = base_pipeline(SVM, 5)
    assert [name for name, _ in svm.steps] == ["scale", "model"]
    assert (svm.named_steps["model"].kernel, svm.named_steps["model"].random_state) == ("rbf", 5)
    gbm_spec = dataclasses.replace(LGBM, fixed={"subsample_freq": 1, "min_child_samples": 5})
    gbm = base_pipeline(gbm_spec, 5, n_jobs=2).named_steps["model"]
    assert (gbm.random_state, gbm.n_jobs, gbm.subsample_freq, gbm.min_child_samples) == (5, 2, 1, 5)
    assert gbm.get_params()["deterministic"] is True
    with pytest.raises(TuningError, match="Unknown family kind"):
        base_pipeline(FamilySpec("knn", "knn", grid=({"k": [1]},)), 0)


def test_penalties_map_to_l1_ratio_and_solver():
    assert pipeline_params(LR, {"penalty": "l1"}, 0) == {"model__l1_ratio": 1.0, "model__solver": "liblinear"}
    assert pipeline_params(LR, {"penalty": "l2"}, 0) == {"model__l1_ratio": 0.0, "model__solver": "lbfgs"}
    assert set(PENALTIES) == {"l1", "l2"}


def test_pipeline_params_translate_a_readable_setting():
    params = pipeline_params(LR, {"features": "kbest_5", "penalty": "l2", "C": 0.1, "class_weight": None}, 0)
    assert set(params) == {"coarsen", "reduce", "model__l1_ratio", "model__solver", "model__C", "model__class_weight"}
    assert params["coarsen"] == "passthrough" and feature_label(params) == "kbest_5"
    assert params["model__C"] == 0.1 and params["model__class_weight"] is None
    assert pipeline_params(LR, {"features": "pca_3"}, 11)["reduce"].random_state == 11
    assert pipeline_params(RF, {"max_features": "sqrt", "min_samples_leaf": 3}, 0) == {
        "model__max_features": "sqrt", "model__min_samples_leaf": 3}
    assert pipeline_params(SVM, {"C": 10.0, "gamma": 1e-4}, 0) == {"model__C": 10.0, "model__gamma": 1e-4}
    assert pipeline_params(LR, {}, 0) == {}


@pytest.mark.parametrize("spec, setting", [
    (RF, {"features": "bins_18da"}),
    (SVM, {"features": "bins_3da"}),
    (LGBM, {"features": "kbest_5"}),
    (SVM, {"penalty": "l2"}),
    (LGBM, {"penalty": "l1"}),
    (LR, {"penalty": "elasticnet"}),
    (LR, {"penalty": None}),
    (LR, {"features": "pca_big"}),
])
def test_pipeline_params_errors(spec, setting):
    with pytest.raises(TuningError):
        pipeline_params(spec, setting, 0)


def test_l1_setting_gives_sparse_coefficients_without_deprecation_warnings():
    X, y, _ = synthetic(n=200, p=60, seed=2)                              # 4 informative, 56 noise columns
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = {pen: base_pipeline(LR, 0).set_params(**pipeline_params(LR, {"penalty": pen, "C": 0.05}, 0))
                  .fit(X, y).named_steps["model"] for pen in ("l1", "l2")}
    messages = [f"{w.category.__name__}: {w.message}" for w in caught]
    assert not [w for w in caught if issubclass(w.category, FutureWarning | DeprecationWarning)], messages
    assert not [m for m in messages if "penalty" in m], messages
    l1, l2 = fitted["l1"], fitted["l2"]
    assert (l1.solver, l1.l1_ratio, l2.solver, l2.l1_ratio) == ("liblinear", 1.0, "lbfgs", 0.0)
    assert np.sum(l1.coef_ == 0) >= 40                                    # embedded feature selection
    assert np.all(l1.coef_[0, :4] != 0)                                   # the informative columns survive
    assert not np.any(l2.coef_ == 0)


# --- grouped folds -----------------------------------------------------------------------------------------------

def test_grouped_folds_keep_patients_together_and_cover_every_row():
    _, y, groups = synthetic(n=300, seed=3)
    for ids in (groups, np.array([f"patient{g}" for g in groups])):
        folds = grouped_folds(y, ids, 5, 42)
        assert len(folds) == 5
        tests = np.concatenate([test for _, test in folds])
        np.testing.assert_array_equal(np.sort(tests), np.arange(len(y)))  # test folds partition the rows
        fold_of_group = {}
        for i, (train_idx, test_idx) in enumerate(folds):
            assert np.intersect1d(train_idx, test_idx).size == 0
            assert train_idx.size + test_idx.size == len(y)
            assert np.intersect1d(ids[train_idx], ids[test_idx]).size == 0
            assert set(np.unique(y[test_idx])) == {0, 1}
            for g in np.unique(ids[test_idx]):
                assert fold_of_group.setdefault(g, i) == i                # each patient in one test fold only
        assert set(fold_of_group) == set(np.unique(ids))


def test_grouped_folds_are_reproducible():
    _, y, groups = synthetic(n=300, seed=3)
    a, b, c = (grouped_folds(y, groups, 5, seed) for seed in (42, 42, 7))
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b, strict=True):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)
    assert any(not np.array_equal(te_a, te_c) for (_, te_a), (_, te_c) in zip(a, c, strict=True))


def test_grouped_folds_reject_a_fold_without_both_classes():
    _, _, groups = synthetic(n=120, seed=4)
    y = np.zeros(120, dtype=int)
    y[:2] = 1                                                             # two positives cannot reach five folds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)                      # "least populated class has 2 members"
        with pytest.raises(TuningError, match="only one class"):
            grouped_folds(y, groups, 5, 42)


# --- run_search --------------------------------------------------------------------------------------------------

# row 0: L1 with C = 1e-4 zeroes every coefficient (constant scores, AUROC 0.5);
# row 1: the 4 best ANOVA columns (the informative ones); row 2: all 30 columns.
LR_SEARCH = FamilySpec("lr", "logistic_regression",
                       grid=({"penalty": ["l1"], "C": [1e-4]}, {"features": ["kbest_4", "bins_3da"], "C": [1.0]}))


@pytest.fixture(scope="module")
def lr_search(train):
    X, y, _, folds = train
    return run_search(LR_SEARCH, X, y, folds, seed=0)


def test_run_search_picks_the_best_setting(lr_search):
    result = lr_search
    settings = candidates(LR_SEARCH, 0)
    assert isinstance(result, SearchResult) and result.family == "lr" and result.seconds > 0
    assert result.best_index == 1
    assert result.best_setting == settings[1] == {"C": 1.0, "features": "kbest_4"}
    t = result.table
    assert list(t["setting"]) == [0, 1, 2]                                # one row per candidate, same order
    assert list(t["C"]) == [s["C"] for s in settings]
    assert list(t["features"]) == ["bins_3da", "kbest_4", "bins_3da"]
    assert t.loc[0, "penalty"] == "l1"
    assert t.loc[0, "cv_roc_auc_mean"] == pytest.approx(0.5)
    assert t.loc[1, "cv_roc_auc_mean"] > max(0.9, t.loc[2, "cv_roc_auc_mean"])
    assert list(t["rank"]) == [3, 1, 2]
    for column in ("cv_roc_auc_mean", "cv_roc_auc_sd", "cv_pr_auc_mean", "cv_pr_auc_sd", "fit_seconds_mean"):
        assert t[column].between(0, 1 if "cv_" in column else np.inf).all(), column
    assert json.loads(json.dumps(result.to_json()))["best_index"] == 1


def test_run_search_scores_do_not_depend_on_other_candidates(train, lr_search):
    # row 2 (default lbfgs / L2) comes after an L1 liblinear candidate; no parameter may leak between them
    X, y, _, folds = train
    alone = run_search(LR, X, y, folds, seed=0).table.iloc[0]
    row = lr_search.table.iloc[2]
    for column in ("cv_roc_auc_mean", "cv_roc_auc_sd", "cv_pr_auc_mean", "cv_pr_auc_sd"):
        assert alone[column] == pytest.approx(row[column], abs=1e-12), column


@pytest.mark.parametrize("leaves", [[5000, 1000], [1000, 5000]])
def test_run_search_ties_go_to_the_setting_listed_first(train, leaves):
    # leaves larger than the data: every tree is a single leaf, so both settings score exactly AUROC 0.5
    X, y, _, folds = train
    spec = FamilySpec("rf", "random_forest", grid=({"min_samples_leaf": leaves},),
                      fixed={"n_estimators": 5, "n_jobs": 1})
    result = run_search(spec, X, y, folds, seed=0)
    assert list(result.table["cv_roc_auc_mean"]) == [0.5, 0.5]
    assert result.best_index == 0 and result.best_setting == {"min_samples_leaf": leaves[0]}
    assert list(result.table["rank"]) == [1, 1]
    assert "features" not in result.table                                # only logistic regression has variants


def test_run_search_duplicate_settings_tie(train):
    X, y, _, folds = train
    spec = FamilySpec("lr", "logistic_regression",
                      grid=({"penalty": ["l1"], "C": [1e-4]}, {"features": ["kbest_4"], "C": [1.0, 1.0]}))
    result = run_search(spec, X, y, folds, seed=0)
    t = result.table
    assert t.loc[1, "cv_roc_auc_mean"] == t.loc[2, "cv_roc_auc_mean"]
    assert result.best_index == 1 and list(t["rank"]) == [3, 1, 1]


def preset_search(roc_auc: list[float], pr_auc: list[float]):
    """Stand-in for GridSearchCV that returns preset mean scores, one per candidate (no fitting)."""

    class PresetSearch:
        def __init__(self, estimator, param_grid, **kwargs):
            self.n_candidates = len(param_grid)

        def fit(self, X, y):
            assert self.n_candidates == len(roc_auc)
            zeros = np.zeros(self.n_candidates)
            self.cv_results_ = {"mean_test_roc_auc": np.array(roc_auc), "std_test_roc_auc": zeros,
                                "mean_test_pr_auc": np.array(pr_auc), "std_test_pr_auc": zeros,
                                "mean_fit_time": zeros}
            return self

    return PresetSearch


FOUR_SVMS = FamilySpec("svm", "svm_rbf", grid=({"C": [1.0, 2.0, 3.0, 4.0]},))


@pytest.mark.parametrize("roc_auc, metric, best, ranks", [
    ([0.7, 0.9, 0.9, 0.8], "roc_auc", 1, [4, 1, 1, 3]),
    ([np.nan, 0.6, np.nan, 0.6], "roc_auc", 1, [3, 1, 3, 1]),             # failed settings rank last
    ([0.9, 0.8, 0.7, 0.6], "pr_auc", 3, [4, 3, 2, 1]),                    # pr_auc is the reverse list here
])
def test_run_search_selection_and_ranks(monkeypatch, roc_auc, metric, best, ranks):
    monkeypatch.setattr(tuning, "GridSearchCV", preset_search(roc_auc, roc_auc[::-1]))
    X, y, _ = synthetic(n=20)
    result = run_search(FOUR_SVMS, X, y, folds=[], seed=0, metric=metric)
    assert result.best_index == best and result.best_setting == {"C": float(best + 1)}
    assert list(result.table["rank"]) == ranks


def test_run_search_without_any_finite_score(monkeypatch):
    nan = [np.nan] * 4
    monkeypatch.setattr(tuning, "GridSearchCV", preset_search(nan, nan))
    X, y, _ = synthetic(n=20)
    with pytest.raises(TuningError, match="no finite"):
        run_search(FOUR_SVMS, X, y, folds=[], seed=0)


class RowSpy(BaseEstimator, TransformerMixin):
    """Pass-through step recording which rows (identified by their first column) reach fit and transform."""

    seen: dict[str, list[np.ndarray]] = {"fit": [], "transform": []}     # class level: survives clone()

    def fit(self, X, y=None):
        RowSpy.seen["fit"].append(np.asarray(X)[:, 0].copy())
        return self

    def transform(self, X):
        RowSpy.seen["transform"].append(np.asarray(X)[:, 0].copy())
        return X


@pytest.fixture
def row_spy(monkeypatch):
    seen: dict[str, list[np.ndarray]] = {"fit": [], "transform": []}
    monkeypatch.setattr(RowSpy, "seen", seen)
    original = tuning.base_pipeline

    def spied(spec, seed, n_jobs=-1):
        return Pipeline([("spy", RowSpy()), *original(spec, seed, n_jobs).steps])

    monkeypatch.setattr(tuning, "base_pipeline", spied)
    return seen


def test_search_and_calibration_only_see_the_rows_passed_in(row_spy):
    X, y, groups = synthetic(n=300, seed=5)
    assert len(set(X[:, 0])) == 300
    inside = np.arange(300) < 200                                         # training part; the rest is held out
    X_tr, y_tr = X[inside], y[inside]
    folds = grouped_folds(y_tr, groups[inside], 4, 42)
    spec = FamilySpec("lr", "logistic_regression", grid=({"C": [0.1, 1.0]},))
    result = run_search(spec, X_tr, y_tr, folds, seed=0)
    model = fit_calibrated(spec, result.best_setting, X_tr, y_tr, folds, seed=0)
    assert len(result.table) == 2

    all_train, held_out = frozenset(X_tr[:, 0]), frozenset(X[~inside, 0])
    seen = frozenset(np.concatenate(row_spy["fit"] + row_spy["transform"]))
    assert seen == all_train and not seen & held_out
    fits = [frozenset(ids) for ids in row_spy["fit"]]
    fold_train = {frozenset(X_tr[tr, 0]) for tr, _ in folds}
    # search: 2 settings x 4 folds; calibration: 4 out-of-fold fits + one fit on the whole training part
    assert len(fits) == 2 * 4 + 4 + 1
    assert fits.count(all_train) == 1 and fits[-1] == all_train
    assert set(fits[:-1]) == fold_train
    assert uncalibrated(model).named_steps["scale"].n_samples_seen_ == 200


# --- fit_calibrated, uncalibrated, converged ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def calibrated_lr(train):
    X, y, _, folds = train
    return fit_calibrated(LR, {"C": 0.1}, X, y, folds, seed=0)


def test_fit_calibrated_refits_on_all_rows_and_keeps_the_ranking(train, holdout, calibrated_lr):
    X, y, _, _ = train
    model = calibrated_lr
    assert isinstance(model, CalibratedClassifierCV)
    assert (model.method, model.ensemble, len(model.calibrated_classifiers_)) == ("sigmoid", False, 1)
    inner = uncalibrated(model)
    assert isinstance(inner, Pipeline)
    np.testing.assert_allclose(inner.named_steps["scale"].mean_, X.mean(axis=0))
    direct = base_pipeline(LR, 0).set_params(model__C=0.1).fit(X, y)
    np.testing.assert_allclose(inner.named_steps["model"].coef_, direct.named_steps["model"].coef_)
    Xv, yv = holdout
    cal, raw = model.predict_proba(Xv), inner.predict_proba(Xv)
    assert cal.shape == (len(yv), 2) and np.all((cal >= 0) & (cal <= 1))
    np.testing.assert_allclose(cal.sum(axis=1), 1.0)
    assert np.abs(cal[:, 1] - raw[:, 1]).max() > 0.01                    # calibration moves probabilities ...
    assert roc_auc_score(yv, cal[:, 1]) == pytest.approx(roc_auc_score(yv, raw[:, 1]), abs=1e-12)  # ... not order
    assert converged(model) is True


def test_calibration_depends_on_the_folds_but_the_refit_does_not(train, holdout, calibrated_lr):
    X, y, groups, _ = train
    other = fit_calibrated(LR, {"C": 0.1}, X, y, grouped_folds(y, groups, 4, 7), seed=0)
    np.testing.assert_array_equal(uncalibrated(other).named_steps["model"].coef_,
                                  uncalibrated(calibrated_lr).named_steps["model"].coef_)
    Xv, _ = holdout
    assert not np.allclose(other.predict_proba(Xv), calibrated_lr.predict_proba(Xv))


@pytest.mark.parametrize("spec, setting", [
    (RF, {"min_samples_leaf": 3}),
    (SVM, {"C": 1.0, "gamma": 0.01}),
    (LGBM, {"n_estimators": 30, "num_leaves": 7, "subsample": 0.7}),
], ids=["random_forest", "svm_rbf", "lightgbm"])
def test_fit_calibrated_other_families(train, holdout, spec, setting):
    X, y, _, folds = train
    model = fit_calibrated(spec, setting, X, y, folds, seed=0)
    inner = uncalibrated(model)
    Xv, yv = holdout
    cal = model.predict_proba(Xv)[:, 1]
    if spec.kind == "svm_rbf":
        assert not hasattr(inner, "predict_proba")                        # probabilities come from calibration
        raw = inner.decision_function(Xv)
    else:
        raw = inner.predict_proba(Xv)[:, 1]
    assert np.all((cal >= 0) & (cal <= 1))
    assert roc_auc_score(yv, cal) > 0.8
    assert roc_auc_score(yv, cal) == pytest.approx(roc_auc_score(yv, raw), abs=1e-12)
    assert converged(model) is None


@pytest.mark.parametrize("spec, setting", [
    (dataclasses.replace(LR, fixed={"max_iter": 1}), {"C": 1.0}),         # fixed overrides the default 5000
    (LR, {"C": 1.0, "max_iter": 1}),
    (LR, {"penalty": "l1", "C": 1.0, "max_iter": 1}),
], ids=["fixed_lbfgs", "setting_lbfgs", "setting_liblinear"])
def test_converged_reports_models_stopped_at_the_iteration_limit(train, spec, setting):
    X, y, _, folds = train
    model = fit_calibrated(spec, setting, X, y, folds, seed=0)
    assert uncalibrated(model).named_steps["model"].max_iter == 1
    assert converged(model) is False


def test_converged_for_l1_logistic_regression(train):
    X, y, _, folds = train
    assert converged(fit_calibrated(LR, {"penalty": "l1", "C": 1.0}, X, y, folds, seed=0)) is True


# --- set_threads -------------------------------------------------------------------------------------------------

def test_set_threads_reaches_every_forest_and_keeps_predictions(train, holdout):
    X, y, _, folds = train
    two_threads = dataclasses.replace(RF, fixed={"n_estimators": 20, "n_jobs": 2})
    model = fit_calibrated(two_threads, {"min_samples_leaf": 3}, X, y, folds, seed=0)
    Xv, _ = holdout

    def forests():
        inner = [cc.estimator.named_steps["model"] for cc in model.calibrated_classifiers_]
        return [*inner, model.estimator.named_steps["model"]]

    assert [f.n_jobs for f in forests()] == [2, 2] and model.n_jobs is None
    before = model.predict_proba(Xv)
    set_threads(model, 1)
    assert [f.n_jobs for f in forests()] == [1, 1] and model.n_jobs == 1
    single = model.predict_proba(Xv)
    # the trees are the same; only the summation order of the threaded average can differ (float rounding)
    np.testing.assert_allclose(single, before, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(model.predict_proba(Xv), single)        # one thread: bit-identical
    set_threads(model, 3)
    assert [f.n_jobs for f in forests()] == [3, 3] and model.n_jobs == 3
    np.testing.assert_allclose(model.predict_proba(Xv), single, rtol=0, atol=1e-12)


def test_set_threads_on_pipelines_and_plain_estimators():
    lr = base_pipeline(LR, 0)                                             # passthrough steps, scaler, model
    set_threads(lr, 3)
    assert lr.named_steps["model"].n_jobs == 3
    svm = base_pipeline(SVM, 0)
    set_threads(svm, 3)                                                   # nothing to set, must not fail
    assert "n_jobs" not in svm.named_steps["model"].get_params()
    gbm = base_pipeline(LGBM, 0)
    set_threads(gbm, 2)
    assert gbm.named_steps["model"].n_jobs == 2
    forest = RandomForestClassifier(n_jobs=4)
    set_threads(forest, 1)
    assert forest.n_jobs == 1
    nested = Pipeline([("inner", base_pipeline(RF, 0))])
    set_threads(nested, 5)
    assert nested.named_steps["inner"].named_steps["model"].n_jobs == 5


# --- caching -----------------------------------------------------------------------------------------------------

def test_code_fingerprint_ignores_line_endings_only(tmp_path):
    text = "def f():\n    return 1\n"
    folders = {name: tmp_path / name for name in ("lf", "crlf", "edited")}
    for folder in folders.values():
        folder.mkdir()
    (folders["lf"] / "m.py").write_bytes(text.encode())
    (folders["crlf"] / "m.py").write_bytes(text.replace("\n", "\r\n").encode())
    (folders["edited"] / "m.py").write_bytes(text.replace("1", "2").encode())
    fp = code_fingerprint([folders["lf"] / "m.py"])
    assert len(fp) == 16
    assert code_fingerprint([folders["crlf"] / "m.py"]) == fp
    assert code_fingerprint([folders["edited"] / "m.py"]) != fp
    (folders["lf"] / "n.py").write_bytes(b"x = 1\n")
    both = code_fingerprint([folders["lf"] / "m.py", folders["lf"] / "n.py"])
    assert both != fp and both == code_fingerprint([folders["lf"] / "n.py", folders["lf"] / "m.py"])
    (folders["lf"] / "renamed.py").write_bytes(text.encode())
    assert code_fingerprint([folders["lf"] / "renamed.py"]) != fp         # file names are part of the hash


def test_cache_key_is_deterministic():
    key = cache_key(split="random", seed=42, family="abc", setting={"C": 1.0, "class_weight": None})
    assert len(key) == 20 and set(key) <= set("0123456789abcdef")
    assert key == cache_key(setting={"class_weight": None, "C": 1.0}, family="abc", seed=42, split="random")
    assert key != cache_key(split="random", seed=43, family="abc", setting={"C": 1.0, "class_weight": None})
    assert key != cache_key(split="random", seed=42, family="abc", setting={"C": 1.0, "class_weight": "balanced"})
    assert key != cache_key(split="random", seed=42, family="abc", setting={"C": 1.0, "class_weight": None}, x=1)
    assert cache_key(value=np.float32(0.5)) == cache_key(value=np.float32(0.5))   # non-JSON values are accepted


def test_cached_computes_once_and_reuses(tmp_path, caplog):
    path = tmp_path / "cache" / "random" / "search.joblib"                # folders are created
    calls = []

    def compute():
        calls.append(1)
        return {"run": len(calls), "array": np.arange(3)}

    log = logging.getLogger("test_tuning.cache")
    with caplog.at_level(logging.INFO, logger="test_tuning.cache"):
        first = cached(path, "k1", compute, log)
        assert calls == [1] and path.is_file() and first["run"] == 1 and "using cached" not in caplog.text
        again = cached(str(path), "k1", compute, log)
        assert calls == [1] and again["run"] == 1
        np.testing.assert_array_equal(again["array"], np.arange(3))
        assert "using cached search.joblib" in caplog.text
    assert cached(path, "k2", compute)["run"] == 2                        # another key: recompute, overwrite
    assert cached(path, "k2", compute)["run"] == 2 and len(calls) == 2
    assert cached(path, "k1", compute)["run"] == 3 and len(calls) == 3    # the k1 entry was replaced
    assert joblib.load(path)["key"] == "k1"


def truncated_cache(path):
    joblib.dump({"key": "k", "value": np.arange(10_000)}, path, compress=3)
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])


@pytest.mark.parametrize("damage", [
    lambda path: path.write_bytes(b""),
    lambda path: path.write_bytes(b"not a joblib file"),
    truncated_cache,
    lambda path: joblib.dump(["not", "a", "cache", "entry"], path),
    lambda path: joblib.dump({"key": "k"}, path),                          # right key, value missing
], ids=["empty", "garbage", "truncated", "foreign", "no_value"])
def test_cached_recomputes_damaged_files(tmp_path, damage):
    path = tmp_path / "fit.joblib"
    damage(path)
    calls = []
    assert cached(path, "k", lambda: calls.append(1) or "fresh") == "fresh" and calls == [1]
    assert joblib.load(path) == {"key": "k", "value": "fresh"}
    assert cached(path, "k", lambda: calls.append(1) or "again") == "fresh" and calls == [1]


def test_cached_keeps_the_old_file_when_compute_fails(tmp_path):
    path = tmp_path / "fit.joblib"
    cached(path, "old", lambda: 1)

    def broken():
        raise RuntimeError("fit failed")

    with pytest.raises(RuntimeError, match="fit failed"):
        cached(path, "new", broken)
    assert joblib.load(path) == {"key": "old", "value": 1}
