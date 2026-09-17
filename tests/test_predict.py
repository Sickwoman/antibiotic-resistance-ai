"""Tests for saving/loading model bundles and predicting from spectra (synthetic data only)."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from src.data_loader import SpectrumFormatError
from src.predict import (
    BUNDLE_FORMAT,
    DISCLAIMER,
    ModelError,
    load_bundle,
    predict_features,
    predict_spectrum_file,
    save_bundle,
)
from src.preprocessing import PreprocessingConfig
from src.train import ModelSpec, build_pipeline
from tests.test_preprocessing import synthetic_spectrum, write_spectrum


@pytest.fixture()
def bundle():
    rng = np.random.default_rng(0)
    X = rng.random((60, 6000)).astype(np.float32) * 1e-4
    y = (rng.random(60) < 0.4).astype(int)
    X[y == 1, :50] += 1e-3
    pcfg = PreprocessingConfig()
    pipe = build_pipeline(ModelSpec("lr", "logistic_regression", {"max_iter": 500}, scale=True), seed=42).fit(X, y)
    return {"format": BUNDLE_FORMAT, "pipeline": pipe, "model_version": "test-lr", "species": "Escherichia coli",
            "antibiotic": "Ciprofloxacin", "threshold": 0.5, "n_features": 6000,
            "preprocessing": pcfg.to_dict(), "feature_fingerprint": pcfg.fingerprint()}


def test_preprocessing_config_round_trip():
    pcfg = PreprocessingConfig(bin_width=6.0, baseline_iterations=(20,))
    again = PreprocessingConfig.from_dict(json.loads(json.dumps(pcfg.to_dict())))
    assert again == pcfg and again.fingerprint() == pcfg.fingerprint()


def test_save_and_load_bundle(tmp_path, bundle):
    path = save_bundle(bundle, tmp_path / "models" / "m.joblib")
    loaded = load_bundle(path)
    assert loaded["model_version"] == "test-lr"
    X = np.full((2, 6000), 1e-4, dtype=np.float32)
    assert np.allclose(predict_features(loaded, X)[0], predict_features(bundle, X)[0])
    card = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert "pipeline" not in card and card["threshold"] == 0.5
    with pytest.raises(ModelError, match="Not a model bundle"):
        save_bundle({"pipeline": None}, tmp_path / "x.joblib")


def test_load_bundle_errors(tmp_path):
    with pytest.raises(ModelError, match="Model file not found"):
        load_bundle(tmp_path / "missing.joblib")
    joblib.dump({"something": 1}, tmp_path / "other.joblib")
    with pytest.raises(ModelError, match="not a model bundle"):
        load_bundle(tmp_path / "other.joblib")
    (tmp_path / "broken.joblib").write_bytes(b"not a joblib file")
    with pytest.raises(ModelError, match="Could not read"):
        load_bundle(tmp_path / "broken.joblib")


def test_predict_features_checks_its_input(bundle):
    prob, label = predict_features(bundle, np.zeros(6000))
    assert prob.shape == (1,) and label[0] == int(prob[0] >= 0.5)
    with pytest.raises(ModelError, match="expects 6000 per spectrum, got 5999"):
        predict_features(bundle, np.zeros(5999))
    bad = np.zeros(6000)
    bad[3] = np.nan
    with pytest.raises(ModelError, match="missing or infinite"):
        predict_features(bundle, bad)


def test_predict_spectrum_file(tmp_path, bundle):
    mz, intensity = synthetic_spectrum(n=1500, seed=3)
    path = write_spectrum(tmp_path / "s.txt", mz, intensity)
    result = predict_spectrum_file(bundle, path).to_dict()
    assert result["prediction"] in ("Resistant", "Susceptible")
    assert (result["prediction"] == "Resistant") == (result["resistance_probability"] >= 0.5)
    assert result["species"] == "Escherichia coli" and result["model_version"] == "test-lr"
    assert result["total_ms"] >= result["preprocessing_ms"] >= 0 and result["inference_ms"] >= 0
    assert result["disclaimer"] == DISCLAIMER and "not a clinically validated" in DISCLAIMER
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    with pytest.raises(SpectrumFormatError):
        predict_spectrum_file(bundle, tmp_path / "empty.txt")
    bundle["feature_fingerprint"] = "0" * 16
    with pytest.raises(ModelError, match="fingerprint"):
        predict_spectrum_file(bundle, path)
