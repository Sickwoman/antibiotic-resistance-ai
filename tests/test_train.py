"""Tests for the Version 0.3 model construction and training loop (synthetic data only)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.evaluate import threshold_for_sensitivity
from src.train import (
    ModelSpec,
    TrainingError,
    build_pipeline,
    grouped_subsample,
    load_rows,
    model_specs,
    run_experiment,
    select_best,
)
from src.utils import load_config


def synthetic(n: int = 300, p: int = 40, seed: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    X = rng.normal(size=(n, p)).astype(np.float32)
    X[:, :5] += y[:, None] * 1.5                       # five informative features
    meta = pd.DataFrame({"sample_index": np.arange(n), "label": y, "group_id": rng.integers(0, n // 2, n)})
    return X, meta


SPECS = [ModelSpec("prevalence", "dummy"),
         ModelSpec("logistic_regression", "logistic_regression", {"C": 1.0, "max_iter": 2000}, scale=True),
         ModelSpec("random_forest", "random_forest", {"n_estimators": 20}, stochastic=True),
         ModelSpec("lightgbm", "lightgbm", {"n_estimators": 20, "min_child_samples": 5})]


def test_model_specs_from_project_config():
    config = load_config()
    specs = {s.name: s for s in model_specs(config)}
    assert list(specs) == ["prevalence", "logistic_regression", "random_forest", "lightgbm"]
    assert specs["logistic_regression"].scale and not specs["random_forest"].scale
    assert specs["random_forest"].stochastic and not specs["lightgbm"].stochastic
    bad = copy.deepcopy(config)
    bad["baselines"]["models"]["svm"] = {"kind": "svm"}
    with pytest.raises(TrainingError, match="unknown kind"):
        model_specs(bad)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_every_baseline_fits_and_gives_probabilities(spec):
    X, meta = synthetic()
    y = meta["label"].to_numpy()
    pipe = build_pipeline(spec, seed=42).fit(X[:200], y[:200])
    prob = pipe.predict_proba(X[200:])[:, 1]
    assert prob.shape == (100,) and np.all((prob >= 0) & (prob <= 1))
    if spec.kind == "dummy":
        assert np.allclose(prob, y[:200].mean())         # the no-skill reference predicts the training rate


def test_scaler_is_fitted_on_training_rows_only():
    X, meta = synthetic()
    pipe = build_pipeline(SPECS[1], seed=42).fit(X[:150], meta["label"].to_numpy()[:150])
    assert np.allclose(pipe.named_steps["scale"].mean_, X[:150].mean(axis=0), atol=1e-5)
    assert not np.allclose(pipe.named_steps["scale"].mean_, X.mean(axis=0), atol=1e-3)


def test_load_rows_matches_fancy_indexing(tmp_path):
    X, _ = synthetic(n=50)
    path = tmp_path / "X.npy"
    np.save(path, X)
    mm = np.load(path, mmap_mode="r")
    rows = np.array([49, 3, 3, 17])
    assert np.array_equal(load_rows(mm, rows, chunk=3), X[rows])


def test_grouped_subsample_keeps_groups_whole():
    X, meta = synthetic(n=400)
    rows = np.arange(10, 390)
    sub = grouped_subsample(meta, rows, 150, seed=42)
    assert sub.size <= 150 and set(sub) <= set(rows)
    groups = meta["group_id"].to_numpy()
    for g in np.unique(groups[sub]):                   # every chosen group is complete within `rows`
        assert set(rows[groups[rows] == g]) <= set(sub)
    assert np.array_equal(sub, grouped_subsample(meta, rows, 150, seed=42))
    assert not np.array_equal(sub, grouped_subsample(meta, rows, 150, seed=7))
    with pytest.raises(TrainingError, match="between 1"):
        grouped_subsample(meta, rows, 1000, seed=42)


def test_run_experiment_uses_validation_for_thresholds_and_leaves_test_alone():
    X, meta = synthetic()
    tr, va = np.arange(0, 180), np.arange(180, 240)
    results = run_experiment(X, meta, experiment="e", split="s", train_rows=tr, validation_rows=va, test_rows=None,
                             specs=SPECS, seeds=[42, 43, 44], threshold_rule={"rule": "min_sensitivity",
                                                                               "min_sensitivity": 0.9})
    assert [(r.model, r.seed) for r in results] == [
        ("prevalence", 42), ("logistic_regression", 42), ("random_forest", 42), ("random_forest", 43),
        ("random_forest", 44), ("lightgbm", 42)]
    y = meta["label"].to_numpy()
    for r in results:
        assert r.test is None and r.test_prob is None and r.train_size == 180
        assert r.threshold == threshold_for_sensitivity(y[va], r.validation_prob, 0.9)
        assert r.validation["sensitivity"] >= 0.9
    rf = [r for r in results if r.model == "random_forest"]
    assert not np.array_equal(rf[0].validation_prob, rf[1].validation_prob)   # seeds matter for the forest
    best = select_best(results, "roc_auc", 42)
    assert best.model != "prevalence" and best.seed == 42


def test_run_experiment_scores_test_rows_with_the_validation_threshold():
    X, meta = synthetic()
    tr, va, te = np.arange(0, 180), np.arange(180, 240), np.arange(240, 300)
    results = run_experiment(X, meta, experiment="e", split="s", train_rows=tr, validation_rows=va, test_rows=te,
                             specs=SPECS[:2], seeds=[42], threshold_rule={"rule": "min_sensitivity",
                                                                          "min_sensitivity": 0.9})
    for r in results:
        assert r.test["n"] == 60 and r.test["threshold"] == r.threshold
        assert r.test_prob.shape == (60,) and r.predict_ms_per_sample >= 0
        assert r.row("test")["roc_auc"] == r.test["roc_auc"]


def test_run_experiment_needs_both_classes():
    X, meta = synthetic()
    meta.loc[:179, "label"] = 0
    with pytest.raises(TrainingError, match="both classes"):
        run_experiment(X, meta, experiment="e", split="s", train_rows=np.arange(180),
                       validation_rows=np.arange(180, 240), test_rows=None, specs=SPECS[:1], seeds=[42],
                       threshold_rule={"rule": "min_sensitivity", "min_sensitivity": 0.9})
