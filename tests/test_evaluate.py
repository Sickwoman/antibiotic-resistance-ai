"""Tests for the Version 0.3 evaluation functions (synthetic data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluate import (
    TEST_LOG_COLUMNS,
    EvaluationError,
    append_test_log,
    bootstrap,
    calibration_slope_intercept,
    choose_threshold,
    classification_metrics,
    group_resample_indices,
    reliability_table,
    summarize_bootstrap,
    threshold_for_sensitivity,
    unpaired_difference,
)


def scores(n: int = 400, seed: int = 0, separation: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(int)
    prob = 1 / (1 + np.exp(-(rng.normal(size=n) + separation * (y - 0.5) * 2)))
    return y, prob


def test_threshold_is_the_highest_cutoff_reaching_the_target():
    y = np.array([1, 1, 1, 1, 0, 0])
    prob = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1])
    t = threshold_for_sensitivity(y, prob, 0.75)          # 3 of 4 resistant must be flagged
    assert t == 0.7
    assert ((prob >= t) & (y == 1)).sum() == 3
    assert ((prob >= np.nextafter(t, 1)) & (y == 1)).sum() < 3    # any higher cut-off misses the target
    assert threshold_for_sensitivity(y, prob, 1.0) == 0.2


def test_threshold_handles_float_rounding_and_ties():
    y = np.ones(10, dtype=int)
    prob = np.linspace(0.1, 1.0, 10)
    assert threshold_for_sensitivity(y, prob, 0.9) == pytest.approx(0.2)   # 9 of 10, not all 10
    y2 = np.array([1, 1, 1, 0])
    tied = np.array([0.5, 0.5, 0.2, 0.9])
    assert threshold_for_sensitivity(y2, tied, 0.5) == 0.5


def test_threshold_validation():
    with pytest.raises(EvaluationError, match="No resistant"):
        threshold_for_sensitivity([0, 0], [0.1, 0.2], 0.9)
    with pytest.raises(EvaluationError, match="min_sensitivity"):
        threshold_for_sensitivity([1, 0], [0.1, 0.2], 0)
    with pytest.raises(EvaluationError, match="between 0 and 1"):
        threshold_for_sensitivity([1, 0], [0.1, 1.2], 0.9)
    with pytest.raises(EvaluationError, match="Unknown threshold rule"):
        choose_threshold([1, 0], [0.1, 0.2], {"rule": "youden"})
    assert choose_threshold([1, 0], [0.3, 0.2], {"rule": "min_sensitivity", "min_sensitivity": 0.9}) == 0.3


def test_metrics_match_scikit_learn():
    y, prob = scores()
    t = 0.45
    m = classification_metrics(y, prob, t)
    pred = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (tp, fp, tn, fn)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, pred))
    assert m["precision"] == pytest.approx(precision_score(y, pred))
    assert m["sensitivity"] == m["recall"] == pytest.approx(recall_score(y, pred))
    assert m["specificity"] == pytest.approx(tn / (tn + fp))
    assert m["f1"] == pytest.approx(f1_score(y, pred))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, prob))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, prob))
    assert m["brier"] == pytest.approx(brier_score_loss(y, prob))
    assert m["n"] == 400 and m["n_resistant"] == y.sum() and m["threshold"] == t


def test_metrics_edge_cases():
    m = classification_metrics([0, 1, 0], [0.1, 0.2, 0.3], threshold=0.9)   # nothing predicted resistant
    assert np.isnan(m["precision"]) and m["sensitivity"] == 0 and m["specificity"] == 1
    single = classification_metrics([0, 0], [0.1, 0.2], threshold=0.5)
    assert np.isnan(single["roc_auc"]) and np.isnan(single["pr_auc"]) and np.isnan(single["sensitivity"])
    with pytest.raises(EvaluationError, match="equally long"):
        classification_metrics([0, 1], [0.1], 0.5)
    with pytest.raises(EvaluationError, match="0 \\(susceptible\\) or 1"):
        classification_metrics([0, 2], [0.1, 0.2], 0.5)


def test_calibration_slope_and_intercept():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, 20000)
    p = 1 / (1 + np.exp(-z))
    y = (rng.random(z.size) < p).astype(int)
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1, abs=0.06) and intercept == pytest.approx(0, abs=0.06)
    over = 1 / (1 + np.exp(-2 * z))                        # overconfident: logits twice too large
    assert calibration_slope_intercept(y, over)[0] == pytest.approx(0.5, abs=0.04)
    constant = np.full(y.size, y.mean())
    slope_c, intercept_c = calibration_slope_intercept(y, constant)
    assert np.isnan(slope_c) and intercept_c == pytest.approx(0, abs=1e-6)


def test_reliability_table_uses_equal_count_bins():
    y, prob = scores(n=103)
    table = reliability_table(y, prob, n_bins=10)
    assert len(table) == 10 and table["n"].sum() == 103 and table["n"].max() - table["n"].min() <= 1
    assert table["mean_predicted"].is_monotonic_increasing
    assert (table["observed_rate"] * table["n"]).sum() == pytest.approx(y.sum())


def test_group_resample_indices_take_whole_groups():
    groups = np.array([5, 7, 5, 9, 7, 5])
    order = np.argsort(groups, kind="stable")
    _, starts, counts = np.unique(groups[order], return_index=True, return_counts=True)
    idx = group_resample_indices(order, starts, counts, np.array([0, 0, 2]))   # group 5 twice, group 9 once
    assert sorted(idx.tolist()) == [0, 0, 2, 2, 3, 5, 5]


def test_bootstrap_is_reproducible_and_paired():
    y, prob = scores(n=300, seed=2)
    groups = np.repeat(np.arange(150), 2)
    worse = np.clip(prob + np.random.default_rng(3).normal(0, 0.3, prob.size), 0, 1)
    probs, thresholds = {"a": prob, "b": worse}, {"a": 0.5, "b": 0.5}
    s1, skipped = bootstrap(y, groups, probs, thresholds, resamples=200, seed=7)
    s2, _ = bootstrap(y, groups, probs, thresholds, resamples=200, seed=7)
    assert skipped == 0 and s1["a"]["roc_auc"].size == 200
    assert np.array_equal(s1["a"]["roc_auc"], s2["a"]["roc_auc"])
    out, _ = summarize_bootstrap(y, groups, probs, thresholds, resamples=300, level=0.95, seed=7, reference="a")
    a = out["intervals"]["a"]["roc_auc"]
    assert a["low"] < a["estimate"] < a["high"]
    diff = out["differences_to_reference"]["b"]["roc_auc"]
    assert diff["estimate"] == pytest.approx(roc_auc_score(y, prob) - roc_auc_score(y, worse))
    assert diff["low"] > 0                                  # the noisier model is clearly worse
    assert "a" not in out["differences_to_reference"]


def test_bootstrap_skips_single_class_resamples():
    y = np.array([1, 0])
    groups = np.array([0, 1])
    samples, skipped = bootstrap(y, groups, {"m": np.array([0.8, 0.2])}, {"m": 0.5}, resamples=200, seed=0)
    assert skipped > 0 and samples["m"]["roc_auc"].size == 200 - skipped
    with pytest.raises(EvaluationError, match="No threshold"):
        bootstrap(y, groups, {"m": np.array([0.8, 0.2])}, {}, resamples=5)


def test_unpaired_difference():
    rng = np.random.default_rng(0)
    low, high = unpaired_difference(rng.normal(0.8, 0.01, 2000), rng.normal(0.7, 0.01, 1500), 0.95)
    assert 0.05 < low < 0.1 < high < 0.15
    assert all(np.isnan(unpaired_difference(np.array([]), np.array([1.0]), 0.95)))


def test_test_log_is_append_only(tmp_path):
    row = {c: 1 for c in TEST_LOG_COLUMNS}
    path = tmp_path / "experiments" / "log.csv"
    append_test_log(path, [row])
    append_test_log(path, [row, row])
    assert len(pd.read_csv(path)) == 3
    assert path.read_text(encoding="utf-8").count("logged_at") == 1
    with pytest.raises(EvaluationError, match="lack column"):
        append_test_log(path, [{"logged_at": 1}])
    other = tmp_path / "other.csv"
    other.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="different columns"):
        append_test_log(other, [row])


# --- review fixes: input validation and the unpaired difference --------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.5])
def test_classification_metrics_rejects_impossible_thresholds(bad):
    """A threshold outside [0, 1] would silently label every sample the same way."""
    y, prob = scores(120)
    with pytest.raises(EvaluationError, match="threshold"):
        classification_metrics(y, prob, bad)


def test_classification_metrics_accepts_the_edges():
    y, prob = scores(120)
    assert classification_metrics(y, prob, 0.0)["sensitivity"] == 1.0      # everything flagged
    assert classification_metrics(y, prob, 1.0)["tp"] + classification_metrics(y, prob, 1.0)["fp"] == 0


def test_bootstrap_checks_the_group_vector():
    y, prob = scores(60)
    groups = np.arange(60)
    with pytest.raises(EvaluationError, match="one entry per label"):
        bootstrap(y, groups[:-1], {"m": prob}, {"m": 0.5}, resamples=5)
    with pytest.raises(EvaluationError, match="one entry per label"):
        bootstrap(y, groups.reshape(30, 2), {"m": prob}, {"m": 0.5}, resamples=5)
    missing = np.array([*groups[:-1].astype(object), None], dtype=object)
    with pytest.raises(EvaluationError, match="missing values"):
        bootstrap(y, missing, {"m": prob}, {"m": 0.5}, resamples=5)


def test_unpaired_difference_recovers_a_known_difference():
    """a and b come from different test sets: the interval is the convolution of the two distributions."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.75, 0.02, 2000)
    b = rng.normal(0.70, 0.02, 2000)
    low, high = unpaired_difference(a, b, 0.95, seed=1)
    assert low < 0.05 < high                                   # contains the true difference
    width = high - low
    expected = 2 * 1.96 * np.sqrt(0.02 ** 2 + 0.02 ** 2)       # sd of the difference of two independents
    assert expected * 0.85 < width < expected * 1.15
    # a paired reading would be far too narrow, which is exactly the mistake this function avoids
    assert width > 2 * 1.96 * 0.02


def test_unpaired_difference_handles_unequal_sizes_and_is_reproducible():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.8, 0.03, 2000), rng.normal(0.6, 0.03, 500)
    first = unpaired_difference(a, b, 0.95, seed=7)
    assert first == unpaired_difference(a, b, 0.95, seed=7)     # same seed, same interval
    assert first != unpaired_difference(a, b, 0.95, seed=8)
    assert first[0] < 0.2 < first[1]
    flipped = unpaired_difference(b, a, 0.95, seed=7)
    assert flipped[0] < -0.2 < flipped[1]                       # the difference simply changes sign


def test_unpaired_difference_without_samples_is_not_a_number():
    empty = np.array([])
    assert all(np.isnan(unpaired_difference(empty, np.array([0.5]), 0.95)))
    assert all(np.isnan(unpaired_difference(np.array([0.5]), empty, 0.95)))


def test_the_test_log_refuses_a_locked_split(tmp_path):
    """Last line of defence: a locked test part cannot even be written to the log."""
    row = {c: 0 for c in TEST_LOG_COLUMNS}
    row.update(split="temporal", model="m", experiment="temporal")
    path = tmp_path / "log.csv"
    with pytest.raises(EvaluationError, match="locked by the evaluation protocol"):
        append_test_log(path, [row], locked=["temporal", "external"])
    assert not path.exists()                                   # nothing written
    append_test_log(path, [{**row, "split": "random", "experiment": "random"}], locked=["temporal", "external"])
    assert pd.read_csv(path)["split"].tolist() == ["random"]
