"""Tests for the Version 0.6 explanations (synthetic spectra, small models, no data files needed).

The properties that matter are checked directly: TreeSHAP contributions must add up to the margin the
model really predicts, bins must map to the right m/z, a planted marker must be found by permutation
importance, and a model that cannot be explained exactly must be refused rather than approximated.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.explain import (
    ExplainError,
    bin_span,
    class_contrast,
    column_frame,
    contiguous_blocks,
    contributions,
    explain_one,
    global_importance,
    inner_pipeline,
    merge_regions,
    model_columns,
    permutation_importance,
    rank_agreement,
)
from src.preprocessing import PreprocessingConfig
from src.tuning import CoarsenBins

BINS = 60                      # 60 bins of 3 Da: m/z 2000 to 2180, small enough to fit a model in a moment
MARKER = slice(20, 26)         # the planted signal: m/z 2060-2078


def pcfg(n_bins: int = BINS) -> PreprocessingConfig:
    """Preprocessing settings whose bin grid is `n_bins` wide, so bin j is m/z [2000+3j, 2003+3j)."""
    return PreprocessingConfig(intensity_transform="none", smoothing_method="none", half_window_size=10,
                               polynomial_order=3, baseline_method="none", baseline_iterations=(20, 100),
                               normalization="none", mz_min=2000.0, mz_max=2000.0 + 3.0 * n_bins,
                               bin_width=3.0)


def synthetic(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Spectra whose only real difference between the classes is one region of neighbouring bins."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(np.int64)
    X = rng.normal(1.0, 0.25, size=(n, BINS))
    X[:, MARKER] += y[:, None] * 1.2
    return X.astype(np.float32), y


@pytest.fixture(scope="module")
def data():
    return synthetic()


@pytest.fixture(scope="module")
def tree(data):
    from lightgbm import LGBMClassifier

    X, y = data
    model = LGBMClassifier(n_estimators=40, num_leaves=7, learning_rate=0.1, random_state=42, verbose=-1,
                           deterministic=True, force_col_wise=True)
    return Pipeline([("model", model)]).fit(X, y)


# --- which pipeline is being explained ---------------------------------------------------------------------------

def test_inner_pipeline_refuses_something_that_is_not_a_model():
    with pytest.raises(ExplainError):
        inner_pipeline("not a model")


def test_bin_span_is_one_without_coarsening_and_the_factor_with_it(tree):
    assert bin_span(tree) == 1
    coarse = Pipeline([("coarsen", CoarsenBins(6)), ("model", LogisticRegression())])
    assert bin_span(coarse) == 6


# --- exact TreeSHAP ----------------------------------------------------------------------------------------------

def test_contributions_sum_to_the_predicted_margin(tree, data):
    X, _ = data
    contrib, base = contributions(tree, X)
    margin = tree.steps[-1][1].booster_.predict(X, raw_score=True)
    assert contrib.shape == X.shape and base.shape == (len(X),)
    assert np.abs(contrib.sum(axis=1) + base - margin).max() < 1e-9


def test_contributions_accept_a_single_spectrum(tree, data):
    X, _ = data
    contrib, base = contributions(tree, X[0])
    assert contrib.shape == (1, BINS) and base.shape == (1,)


def test_contributions_are_largest_where_the_signal_was_planted(tree, data):
    X, _ = data
    contrib, _ = contributions(tree, X)
    strength = np.abs(contrib).mean(axis=0)
    assert strength[MARKER].max() == strength.max()


def test_a_model_without_trees_is_refused_instead_of_approximated(data):
    X, y = data
    linear = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500))]).fit(X, y)
    with pytest.raises(ExplainError, match="permutation_importance"):
        contributions(linear, X)


def test_a_tolerance_that_cannot_be_met_is_reported(tree, data):
    X, _ = data
    with pytest.raises(ExplainError, match="do not sum"):
        contributions(tree, X, tolerance=0.0)


# --- bins to m/z -------------------------------------------------------------------------------------------------

def test_columns_carry_the_m_z_interval_they_cover():
    frame = column_frame(BINS, 1, pcfg())
    assert frame.loc[0, "mz_start"] == pytest.approx(2000.0) and frame.loc[0, "mz_end"] == pytest.approx(2003.0)
    assert frame.loc[10, "mz_start"] == pytest.approx(2030.0)
    coarse = column_frame(BINS // 6, 6, pcfg())
    assert coarse.loc[0, "mz_end"] == pytest.approx(2018.0)       # one coarse column spans six 3 Da bins
    assert coarse.loc[1, "mz_start"] == pytest.approx(2018.0)


def test_a_column_count_that_does_not_fit_the_bin_grid_is_refused():
    with pytest.raises(ExplainError, match="disagree"):
        column_frame(BINS + 5, 1, pcfg())


# --- global importance and regions -------------------------------------------------------------------------------

def test_global_importance_is_sorted_and_finds_the_marker(tree, data):
    X, _ = data
    frame = global_importance(contributions(tree, X)[0], 1, pcfg())
    assert frame["mean_abs"].is_monotonic_decreasing
    assert len(frame) == BINS and frame["rows_used"].iloc[0] == len(X)
    top = frame.iloc[0]
    assert MARKER.start * 3 + 2000 <= top["mz_start"] < MARKER.stop * 3 + 2000
    assert frame["value_correlation"].isna().all()                 # no columns given, so no direction


def test_the_global_direction_comes_from_the_value_correlation_not_the_mean(tree, data):
    X, _ = data
    contrib, _ = contributions(tree, X)
    frame = global_importance(contrib, 1, pcfg(), model_columns(tree, X))
    top = frame.iloc[0]
    # Averaged over a whole part the signed contributions cancel out; the correlation still points the way.
    assert abs(top["mean_signed"]) < 0.1 * top["mean_abs"]
    assert top["value_correlation"] > 0.3                          # higher intensity here -> resistant


def test_global_importance_refuses_an_empty_matrix_or_mismatched_columns(tree, data):
    X, _ = data
    with pytest.raises(ExplainError):
        global_importance(np.zeros((0, BINS)), 1, pcfg())
    with pytest.raises(ExplainError, match="do not match"):
        global_importance(contributions(tree, X)[0], 1, pcfg(), X[:5])


def test_neighbouring_bins_merge_into_one_region(tree, data):
    X, _ = data
    importance = global_importance(contributions(tree, X)[0], 1, pcfg(), model_columns(tree, X))
    regions = merge_regions(importance, top_bins=8, merge_gap=2, keep=None)
    assert len(regions) >= 1
    first = regions.iloc[0]
    assert first["mz_start"] < first["mz_end"] and first["n_columns"] >= 2
    assert first["towards"] == "resistant"
    assert first["mz_start"] >= 2000.0 + 3 * (MARKER.start - 2)     # the region sits on the planted marker
    assert regions["rank"].tolist() == sorted(regions["rank"].tolist())


def test_a_gap_larger_than_merge_gap_keeps_two_regions():
    frame = column_frame(BINS, 1, pcfg())
    frame["mean_abs"] = 0.0
    frame["mean_signed"] = 0.0
    frame.loc[[0, 1, 30, 31], "mean_abs"] = 1.0                    # two pairs, far apart
    frame.loc[[0, 1, 30, 31], "mean_signed"] = 1.0
    assert len(merge_regions(frame, top_bins=4, merge_gap=2, keep=None, towards_from="signed")) == 2
    assert len(merge_regions(frame, top_bins=4, merge_gap=40, keep=None, towards_from="signed")) == 1


def test_merge_regions_keeps_only_what_was_asked_for_and_checks_its_arguments():
    frame = column_frame(BINS, 1, pcfg())
    frame["mean_abs"] = np.linspace(1, 0, BINS)
    frame["mean_signed"] = -frame["mean_abs"]
    assert len(merge_regions(frame, top_bins=20, merge_gap=0, keep=3, towards_from="signed")) == 3
    kept = merge_regions(frame, top_bins=20, merge_gap=0, keep=3, towards_from="signed")
    assert kept["towards"].eq("susceptible").all()
    with pytest.raises(ExplainError):
        merge_regions(frame, top_bins=0)
    with pytest.raises(ExplainError, match="towards_from"):
        merge_regions(frame, top_bins=4, towards_from="guess")
    with pytest.raises(ExplainError, match="No value correlations"):
        merge_regions(frame, top_bins=4)                           # no correlations available at all


# --- permutation importance --------------------------------------------------------------------------------------

def test_blocks_cover_every_column_exactly_once():
    blocks = contiguous_blocks(BINS, 7)
    assert blocks[0] == (0, 7) and blocks[-1][1] == BINS
    assert sum(end - start for start, end in blocks) == BINS
    with pytest.raises(ExplainError):
        contiguous_blocks(BINS, 0)


def test_permuting_the_marker_block_costs_the_most_auroc(tree, data):
    X, y = data
    frame = permutation_importance(tree, X, y, contiguous_blocks(BINS, 6), repeats=3, seed=1, pcfg=pcfg())
    best = frame.iloc[0]
    assert best["bin_start"] <= MARKER.start < best["bin_end"] + 1
    assert best["auroc_drop_mean"] > 0
    assert 0 < best["baseline_roc_auc"] <= 1
    assert best["mz_start"] == pytest.approx(2000.0 + 3 * best["bin_start"])


def test_permutation_importance_leaves_the_matrix_it_was_given_unchanged(tree, data):
    X, y = data
    before = X.copy()
    permutation_importance(tree, X, y, contiguous_blocks(BINS, 20), repeats=2, seed=1)
    assert np.array_equal(X, before)


def test_permutation_importance_checks_its_inputs(tree, data):
    X, y = data
    with pytest.raises(ExplainError):
        permutation_importance(tree, X, y[:10], contiguous_blocks(BINS, 6))
    with pytest.raises(ExplainError):
        permutation_importance(tree, X, np.zeros(len(X), dtype=np.int64), contiguous_blocks(BINS, 6))
    with pytest.raises(ExplainError):
        permutation_importance(tree, X, y, [(0, BINS + 5)])
    with pytest.raises(ExplainError):
        permutation_importance(tree, X, y, contiguous_blocks(BINS, 6), repeats=0)


def test_permutation_importance_runs_on_a_model_that_has_no_trees(data):
    X, y = data
    linear = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500))]).fit(X, y)
    frame = permutation_importance(linear, X, y, contiguous_blocks(BINS, 6), repeats=2, seed=1)
    assert frame.iloc[0]["auroc_drop_mean"] > 0                    # it is model-agnostic, as advertised


# --- the model-free reference ------------------------------------------------------------------------------------

def test_class_contrast_finds_the_planted_difference_without_a_model(tree, data):
    X, y = data
    importance = global_importance(contributions(tree, X)[0], 1, pcfg(), model_columns(tree, X))
    regions = merge_regions(importance, top_bins=8, merge_gap=2, keep=3)
    contrast = class_contrast(X, y, regions)
    assert len(contrast) == len(regions)
    assert contrast.iloc[0]["standardised_difference"] > 0.5       # higher in resistant spectra, as planted
    assert contrast.iloc[0]["mean_resistant"] > contrast.iloc[0]["mean_susceptible"]


def test_class_contrast_needs_both_classes(tree, data):
    X, y = data
    importance = global_importance(contributions(tree, X)[0], 1, pcfg(), model_columns(tree, X))
    regions = merge_regions(importance, top_bins=4, keep=1)
    with pytest.raises(ExplainError):
        class_contrast(X, np.zeros(len(X), dtype=np.int64), regions)


# --- agreement ---------------------------------------------------------------------------------------------------

def test_rank_agreement_is_perfect_for_identical_vectors_and_low_for_reversed():
    values = np.linspace(0, 1, 50)
    same = rank_agreement(values, values, top_k=10)
    assert same["spearman"] == pytest.approx(1.0) and same["overlap"] == 10
    assert same["overlap_fraction"] == pytest.approx(1.0)
    reversed_ = rank_agreement(values, values[::-1], top_k=10)
    assert reversed_["spearman"] == pytest.approx(-1.0) and reversed_["overlap"] == 0


def test_rank_agreement_caps_top_k_at_the_number_of_columns_and_checks_shapes():
    assert rank_agreement(np.arange(3.0), np.arange(3.0), top_k=20)["top_k"] == 3
    with pytest.raises(ExplainError):
        rank_agreement(np.arange(5.0), np.arange(4.0))


# --- one spectrum ------------------------------------------------------------------------------------------------

def test_explain_one_names_regions_in_m_z_and_signs_them(tree, data):
    X, _ = data
    regions = explain_one(tree, X[0], pcfg(), top_bins=10, merge_gap=2, keep=3)
    assert 1 <= len(regions) <= 3
    assert set(regions.columns) >= {"rank", "mz_start", "mz_end", "total_signed", "towards"}
    assert (regions["mz_start"] >= 2000.0).all() and (regions["mz_end"] <= 2000.0 + 3 * BINS).all()
    assert regions["towards"].isin(["resistant", "susceptible"]).all()


def test_explain_one_refuses_more_than_one_spectrum(tree, data):
    X, _ = data
    with pytest.raises(ExplainError, match="one spectrum"):
        explain_one(tree, X[:3], pcfg())
