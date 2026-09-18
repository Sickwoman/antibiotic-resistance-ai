"""Tests for saving/loading model bundles and predicting from spectra (synthetic data only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.data_loader import SpectrumFormatError
from src.predict import (
    BUNDLE_FORMAT,
    DISCLAIMER,
    ModelError,
    checksum_path,
    load_bundle,
    predict_features,
    predict_spectrum_file,
    save_bundle,
    verify_digest,
)
from src.preprocessing import PreprocessingConfig
from src.train import ModelSpec, build_pipeline
from tests.test_preprocessing import synthetic_spectrum, write_spectrum

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import predict_spectrum  # noqa: E402


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


def test_loaded_models_use_one_thread_unless_asked(tmp_path, bundle):
    rng = np.random.default_rng(1)
    X = rng.random((40, 6000)).astype(np.float32)
    y = np.tile([0, 1], 20)
    forest = build_pipeline(ModelSpec("rf", "random_forest", {"n_estimators": 10}, stochastic=True), seed=42)
    path = save_bundle({**bundle, "pipeline": forest.fit(X, y)}, tmp_path / "rf.joblib")
    single, saved = load_bundle(path), load_bundle(path, n_jobs=None)
    assert single["pipeline"].steps[-1][1].n_jobs == 1 and saved["pipeline"].steps[-1][1].n_jobs == -1
    assert np.array_equal(predict_features(single, X)[0], predict_features(saved, X)[0])
    lr = load_bundle(save_bundle(bundle, tmp_path / "lr.joblib"))   # LogisticRegression also has n_jobs;
    assert lr["pipeline"].steps[-1][1].n_jobs == 1                   # setting it does not change results


def test_predict_spectrum_command(tmp_path, bundle, capsys):
    mz, intensity = synthetic_spectrum(n=1500, seed=4)
    spectrum = write_spectrum(tmp_path / "s.txt", mz, intensity)
    model = save_bundle(bundle, tmp_path / "m.joblib")
    assert predict_spectrum.main([str(spectrum), "--model", str(model)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["prediction"] in ("Resistant", "Susceptible") and out["disclaimer"] == DISCLAIMER
    assert predict_spectrum.main([str(spectrum), "--model", str(tmp_path / "missing.joblib")]) == 1
    assert capsys.readouterr().err.startswith("Error: Model file not found")
    assert predict_spectrum.main([str(tmp_path / "nothing.txt"), "--model", str(model)]) == 1
    assert "Spectrum file not found" in capsys.readouterr().err


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


# --- review fix: the loader checks the fields the prediction path reads -------------------------------------------

@pytest.mark.parametrize("field", ["model_version", "pipeline", "threshold", "n_features", "preprocessing",
                                   "feature_fingerprint", "species", "antibiotic"])
def test_a_bundle_missing_a_required_field_is_refused(tmp_path, bundle, field):
    broken = {k: v for k, v in bundle.items() if k != field}
    path = tmp_path / "m.joblib"
    joblib.dump(broken, path)
    with pytest.raises(ModelError, match=field):
        load_bundle(path)


@pytest.mark.parametrize("field, value", [("threshold", 1.5), ("threshold", float("nan")), ("n_features", 0),
                                          ("n_features", -10)])
def test_a_bundle_with_an_impossible_value_is_refused(tmp_path, bundle, field, value):
    path = tmp_path / "m.joblib"
    joblib.dump({**bundle, field: value}, path)
    with pytest.raises(ModelError):
        load_bundle(path)


def test_a_bundle_whose_model_cannot_give_probabilities_is_refused(tmp_path, bundle):
    from sklearn.preprocessing import StandardScaler  # a fitted transformer, not a classifier

    path = tmp_path / "m.joblib"
    joblib.dump({**bundle, "pipeline": StandardScaler()}, path)
    with pytest.raises(ModelError, match="probabilities"):
        load_bundle(path)


# --- review fix: a model file that changed after it was saved is not loaded ---------------------------------------

def test_a_tampered_model_file_is_refused(tmp_path, bundle):
    """joblib files execute code when loaded, so a file that no longer matches its checksum is refused."""
    path = save_bundle(bundle, tmp_path / "m.joblib")
    sidecar = checksum_path(path)
    assert sidecar.is_file() and len(sidecar.read_text(encoding="utf-8").split()[0]) == 64
    assert load_bundle(path)["model_version"] == "test-lr"          # unchanged file still loads

    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ModelError, match="does not match its checksum"):
        load_bundle(path)


def test_a_model_without_a_checksum_still_loads(tmp_path, bundle):
    """Bundles saved before checksums existed keep working; the check is an extra, not a gate."""
    path = save_bundle(bundle, tmp_path / "m.joblib")
    checksum_path(path).unlink()
    assert verify_digest(path) is None
    assert load_bundle(path)["model_version"] == "test-lr"
