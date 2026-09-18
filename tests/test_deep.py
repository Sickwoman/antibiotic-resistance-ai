"""Tests for the Version 0.5 networks (synthetic data only, a few epochs each)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.deep import (
    MLP,
    DeepError,
    SpectrumCNN,
    TorchClassifier,
    parse_sizes,
    set_threads,
    single_thread,
)

SMALL = {"max_epochs": 4, "batch_size": 16, "patience": 2}


def synthetic(n: int = 120, p: int = 24, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Small matrix whose first columns carry the signal, like a few informative m/z bins."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    X = rng.normal(size=(n, p))
    X[:, :4] += y[:, None] * 1.5
    return X.astype(np.float32), y


@pytest.fixture(scope="module")
def data():
    return synthetic()


# --- layer sizes -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [("256,64", (256, 64)), (" 8 , 4 ", (8, 4)), ("16", (16,)),
                                             ([128], (128,)), ((32, 16), (32, 16)), (["8", 4], (8, 4))])
def test_parse_sizes_reads_strings_and_lists(value, expected):
    assert parse_sizes(value) == expected


@pytest.mark.parametrize("value", ["", " , ", "0", "-3", "4,0", [], ["a"], [1.5], None])
def test_parse_sizes_rejects_anything_else(value):
    with pytest.raises(DeepError):
        parse_sizes(value)


# --- the two architectures -------------------------------------------------------------------------------------

def test_mlp_maps_a_batch_to_one_logit_per_row():
    net = MLP(n_features=24, hidden=(8, 4), dropout=0.1)
    assert net(torch.zeros(7, 24)).shape == (7,)


def test_cnn_maps_a_batch_to_one_logit_per_row_whatever_the_length():
    net = SpectrumCNN(channels=(4, 8), kernel_size=5, dropout=0.1)
    assert net(torch.zeros(7, 24)).shape == (7,)
    assert net(torch.zeros(7, 64)).shape == (7,)      # global pooling: the head does not depend on the length


def test_parameter_count_matches_the_layer_sizes(data):
    X, y = data
    model = TorchClassifier(kind="mlp", hidden="4", dropout=0.0, **SMALL).fit(X, y)
    assert model.n_parameters() == 24 * 4 + 4 + 4 * 1 + 1                 # weights and biases of two layers
    cnn = TorchClassifier(kind="cnn", channels="4", kernel_size=3, **SMALL).fit(X, y)
    assert cnn.n_parameters() == (1 * 4 * 3 + 4) + (4 + 4) + (4 + 1)      # conv, batch norm, linear head


# --- fitting and predicting ------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["mlp", "cnn"])
def test_fit_predict_round_trip(data, kind):
    X, y = data
    model = TorchClassifier(kind=kind, hidden="8", channels="4", kernel_size=3, **SMALL).fit(X, y)
    check_is_fitted(model, "model_")
    assert list(model.classes_) == [0, 1] and model.n_features_in_ == X.shape[1]
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert set(np.unique(model.predict(X))) <= {0, 1}
    assert model.decision_function(X).shape == (len(X),)
    # the 0.5 cut-off of predict is the 0 cut-off of the logit
    np.testing.assert_array_equal(model.predict(X), (model.decision_function(X) >= 0).astype(np.int64))


def test_the_same_seed_gives_the_same_model(data):
    X, y = data
    first = TorchClassifier(kind="mlp", hidden="8", random_state=1, **SMALL).fit(X, y)
    again = TorchClassifier(kind="mlp", hidden="8", random_state=1, **SMALL).fit(X, y)
    other = TorchClassifier(kind="mlp", hidden="8", random_state=2, **SMALL).fit(X, y)
    np.testing.assert_array_equal(first.predict_proba(X), again.predict_proba(X))
    assert first.history == again.history
    assert not np.array_equal(first.predict_proba(X), other.predict_proba(X))


def test_a_cloned_estimator_keeps_its_settings():
    model = TorchClassifier(kind="cnn", channels="4,8", kernel_size=7, dropout=0.4, positive_class_weight=True)
    assert clone(model).get_params() == model.get_params()


# --- early stopping, checkpointing, history --------------------------------------------------------------------

def test_early_stopping_restores_the_best_epoch(data):
    X, y = data
    model = TorchClassifier(kind="mlp", hidden="8", max_epochs=40, patience=2, batch_size=16,
                            random_state=0).fit(X, y)
    history = model.history
    assert history["stopped_early"] is True                       # noise data: it cannot improve for ever
    assert len(history["train_loss"]) == len(history["validation_loss"]) == history["epochs"]
    assert len(history["validation_roc_auc"]) == history["epochs"]
    assert 0 <= history["best_epoch"] < history["epochs"]
    best = history["validation_roc_auc"][history["best_epoch"]]
    assert best == max(history["validation_roc_auc"]) == pytest.approx(model.inner_validation_roc_auc_)
    # the kept epoch is at most `patience` epochs before the last one
    assert history["epochs"] - 1 - history["best_epoch"] == model.patience


def test_the_epoch_limit_is_respected(data):
    X, y = data
    model = TorchClassifier(kind="mlp", hidden="8", max_epochs=3, patience=99, batch_size=16).fit(X, y)
    assert model.history["epochs"] == 3 and model.history["stopped_early"] is False


def test_history_needs_a_fitted_model():
    with pytest.raises(NotFittedError):
        _ = TorchClassifier().history


def test_a_single_row_at_the_end_of_an_epoch_is_dropped():
    """Batch normalisation cannot standardise one row, so a trailing batch of one must not reach it."""
    X, y = synthetic(n=20, p=24, seed=3)
    inner_train = 20 - int(np.ceil(20 * 0.15))                     # one batch of 16, then a single row
    assert inner_train % 16 == 1
    model = TorchClassifier(kind="cnn", channels="4", kernel_size=3, **SMALL).fit(X, y)
    assert model.history["epochs"] >= 1


# --- errors ----------------------------------------------------------------------------------------------------

def test_unknown_kind_is_refused(data):
    X, y = data
    with pytest.raises(DeepError, match="Unknown network kind"):
        TorchClassifier(kind="transformer", **SMALL).fit(X, y)


def test_one_class_is_refused(data):
    X, _ = data
    with pytest.raises(DeepError, match="Both classes"):
        TorchClassifier(**SMALL).fit(X, np.zeros(len(X), dtype=int))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.5])
def test_the_inner_split_must_be_a_fraction(data, fraction):
    X, y = data
    with pytest.raises(DeepError, match="inner_validation_fraction"):
        TorchClassifier(inner_validation_fraction=fraction, **SMALL).fit(X, y)


def test_shapes_must_agree(data):
    X, y = data
    with pytest.raises(DeepError, match="one label per row"):
        TorchClassifier(**SMALL).fit(X, y[:-1])
    with pytest.raises(DeepError, match="one label per row"):
        TorchClassifier(**SMALL).fit(X.ravel(), y)


def test_a_kernel_must_have_a_width(data):
    X, y = data
    with pytest.raises(DeepError, match="kernel_size"):
        TorchClassifier(kind="cnn", kernel_size=0, **SMALL).fit(X, y)


def test_predicting_needs_the_training_width(data):
    X, y = data
    model = TorchClassifier(kind="mlp", hidden="8", **SMALL).fit(X, y)
    with pytest.raises(DeepError, match="features per row"):
        model.decision_function(X[:, :-1])


# --- threads ---------------------------------------------------------------------------------------------------

def test_single_thread_restores_the_previous_setting():
    before = torch.get_num_threads()
    try:
        set_threads(2)
        with single_thread():
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == 2
        with pytest.raises(ValueError), single_thread():          # also restored when something fails
            raise ValueError("boom")
        assert torch.get_num_threads() == 2
    finally:
        set_threads(before)


def test_predictions_do_not_depend_on_the_thread_count(data):
    """The test run must reproduce the development run's validation numbers exactly."""
    X, y = data
    before = torch.get_num_threads()
    try:
        set_threads(1)
        model = TorchClassifier(kind="mlp", hidden="8", **SMALL).fit(X, y)
        one = model.predict_proba(X)
        set_threads(4)
        np.testing.assert_array_equal(model.predict_proba(X), one)
    finally:
        set_threads(before)
