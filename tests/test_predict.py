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
    EXPLAIN_REGIONS,
    ModelError,
    checksum_path,
    load_bundle,
    load_zones,
    predict_features,
    predict_spectrum_file,
    save_bundle,
    verify_digest,
)
from src.preprocessing import PreprocessingConfig
from src.train import ModelSpec, build_pipeline
from src.uncertainty import ADVICE, RESISTANT, SUSCEPTIBLE, UNCERTAIN, ZoneRule, Zones
from src.utils import load_config, project_path
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


# --- Version 0.6: confidence zones and the explanation of one prediction -----------------------------------------

@pytest.fixture()
def tree_bundle(bundle):
    """The same bundle with a small gradient-boosted tree model: only those give exact contributions."""
    rng = np.random.default_rng(2)
    X = rng.random((60, 6000)).astype(np.float32) * 1e-4
    y = np.tile([0, 1], 30)
    X[y == 1, :50] += 1e-3
    tree = build_pipeline(ModelSpec("lgbm", "lightgbm", {"n_estimators": 20, "min_child_samples": 5}),
                          seed=42, n_jobs=1).fit(X, y)
    return {**bundle, "pipeline": tree, "model_version": "test-lightgbm"}


def write_zones(path: Path, lower, upper, threshold: float = 0.5) -> Path:
    path.write_text(json.dumps(Zones(threshold, lower, upper, ZoneRule()).to_dict()), encoding="utf-8")
    return path


def test_load_zones_round_trips_a_written_file(tmp_path):
    zones = Zones(0.5, 0.2, 0.8, ZoneRule(), "validation", 123)
    path = tmp_path / "uncertainty.json"
    path.write_text(json.dumps(zones.to_dict()), encoding="utf-8")
    loaded = load_zones(path)
    assert loaded == zones
    assert (loaded.one(0.1), loaded.one(0.5), loaded.one(0.9)) == (SUSCEPTIBLE, UNCERTAIN, RESISTANT)


def test_load_zones_returns_none_when_there_is_no_zones_file(tmp_path):
    """No zones fitted yet is not an error: the prediction is then reported without a confidence label."""
    assert load_zones(tmp_path / "uncertainty.json") is None


@pytest.mark.parametrize("text", ['{"threshold": 0.5}',                          # no edges: not a zones record
                                  '{"threshold": 0.5, "lower": 0.8, "upper": 0.2}',   # edges crossed
                                  '{"threshold": "x", "lower": null, "upper": null}',
                                  '[0.2, 0.8]', 'not json at all'])
def test_load_zones_refuses_a_file_that_is_not_a_zones_record(tmp_path, text):
    path = tmp_path / "uncertainty.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ModelError, match="confidence"):
        load_zones(path)


def test_a_prediction_with_zones_carries_the_confidence_label(tmp_path, bundle):
    """Each side of each edge, with the edges placed around the probability this model actually gives."""
    mz, intensity = synthetic_spectrum(n=1500, seed=3)
    path = write_spectrum(tmp_path / "s.txt", mz, intensity)
    plain = predict_spectrum_file(bundle, path)
    p = plain.resistance_probability
    assert 0.05 < p + 0.02 and p + 0.02 < 1                 # the edges below stay inside [0, 1]

    def confidence(lower, upper):
        result = predict_spectrum_file(bundle, path, zones=Zones(0.5, lower, upper, ZoneRule()))
        assert result.prediction == plain.prediction and result.resistance_probability == p
        return result.confidence

    assert confidence(p + 0.01, p + 0.02) == SUSCEPTIBLE     # below the lower edge
    assert confidence(p - 0.01, p + 0.01) == UNCERTAIN       # between the two edges
    assert confidence(p - 0.02, p - 0.01) == RESISTANT       # above the upper edge
    assert confidence(None, p + 0.01) == UNCERTAIN           # no susceptible zone exists on that side
    assert confidence(p - 0.01, None) == UNCERTAIN           # no resistant zone exists on that side
    assert confidence(None, None) == UNCERTAIN


def test_a_prediction_without_zones_says_the_confidence_is_not_available(tmp_path, bundle):
    """Not the same as 'uncertain': nothing was computed, so nothing is claimed."""
    mz, intensity = synthetic_spectrum(n=1500, seed=3)
    result = predict_spectrum_file(bundle, write_spectrum(tmp_path / "s.txt", mz, intensity)).to_dict()
    assert result["confidence"] == "not available" and result["explanation"] is None
    assert result["disclaimer"] == DISCLAIMER


def test_explaining_a_model_without_exact_contributions_does_not_fail_the_prediction(tmp_path, bundle, capsys):
    """The fixture is a logistic regression, which has no TreeSHAP: the prediction is unchanged, the
    explanation is simply absent."""
    mz, intensity = synthetic_spectrum(n=1500, seed=3)
    spectrum = write_spectrum(tmp_path / "s.txt", mz, intensity)
    plain = predict_spectrum_file(bundle, spectrum)
    explained = predict_spectrum_file(bundle, spectrum, explain=True)
    assert explained.explanation is None
    assert (explained.prediction, explained.resistance_probability) == (plain.prediction,
                                                                        plain.resistance_probability)
    model = save_bundle(bundle, tmp_path / "m.joblib")
    assert predict_spectrum.main([str(spectrum), "--model", str(model), "--explain"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["explanation"] is None and out["prediction"] == plain.prediction
    assert out["disclaimer"] == DISCLAIMER


def test_explaining_a_tree_model_reports_signed_m_z_regions(tmp_path, tree_bundle):
    mz, intensity = synthetic_spectrum(n=1500, seed=3)
    spectrum = write_spectrum(tmp_path / "s.txt", mz, intensity)
    plain = predict_spectrum_file(tree_bundle, spectrum)
    result = predict_spectrum_file(tree_bundle, spectrum, explain=True)
    assert result.resistance_probability == plain.resistance_probability      # explaining changes nothing
    regions = result.explanation
    assert regions and len(regions) <= EXPLAIN_REGIONS
    assert [r["rank"] for r in regions] == list(range(1, len(regions) + 1))
    for region in regions:
        assert 2000.0 <= region["mz_start"] < region["mz_end"] <= 20000.0     # the preprocessing m/z range
        assert region["towards"] in ("resistant", "susceptible")
        if region["contribution"]:            # regions whose rounded contribution is 0 have no direction
            assert region["towards"] == ("resistant" if region["contribution"] > 0 else "susceptible")
    assert any(r["contribution"] for r in regions)


def test_predict_spectrum_command_reports_confidence_and_advice(tmp_path, bundle, capsys):
    mz, intensity = synthetic_spectrum(n=1500, seed=4)
    spectrum = write_spectrum(tmp_path / "s.txt", mz, intensity)
    model = save_bundle(bundle, tmp_path / "m.joblib")
    assert predict_spectrum.main([str(spectrum), "--model", str(model)]) == 0
    p = json.loads(capsys.readouterr().out)["resistance_probability"]

    uncertain = write_zones(tmp_path / "uncertain.json", p - 0.01, p + 0.01)
    assert predict_spectrum.main([str(spectrum), "--model", str(model), "--uncertainty", str(uncertain)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["confidence"] == UNCERTAIN and out["advice"] == ADVICE and out["disclaimer"] == DISCLAIMER

    confident = write_zones(tmp_path / "confident.json", p + 0.01, p + 0.02)
    assert predict_spectrum.main([str(spectrum), "--model", str(model), "--uncertainty", str(confident)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["confidence"] == SUSCEPTIBLE and "advice" not in out

    assert predict_spectrum.main([str(spectrum), "--model", str(model),
                                  "--uncertainty", str(tmp_path / "gone.json")]) == 1
    assert capsys.readouterr().err.startswith("Error: Confidence zones file not found")


def test_the_command_defaults_to_the_version_0_6_model_and_zones():
    config = load_config()
    model = predict_spectrum.default_model(config)
    assert model.is_relative_to(project_path(config["explain"]["model_dir"]))
    assert model.name == f"best_{config['explain']['split']}.joblib"
    zones = predict_spectrum.default_zones_path(config)
    assert zones.is_relative_to(project_path(config["explain"]["report_dir"])) and zones.suffix == ".json"

    older = {k: v for k, v in config.items() if k != "explain"}      # a checkout without the Version 0.6 section
    assert predict_spectrum.default_model(older).is_relative_to(project_path(config["baselines"]["model_dir"]))
    assert predict_spectrum.default_model(older) != model
    assert predict_spectrum.default_zones_path(older) is None
