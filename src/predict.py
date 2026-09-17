"""Save and load trained model bundles and make research predictions from spectra.

A bundle is one joblib file holding the fitted pipeline together with everything needed to reproduce
and describe it (preprocessing settings, threshold, dataset fingerprints, metrics, versions). A JSON
card with the same information, minus the pipeline, is written next to it.

Security: joblib files can execute code when loaded. Only load bundles produced by this project,
never files uploaded by users.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.preprocessing import PreprocessingConfig, preprocess_file

BUNDLE_FORMAT = "amr-model-bundle/1"
DISCLAIMER = ("AI research prediction from a research prototype. It is not a clinically validated diagnostic, "
              "does not replace laboratory antimicrobial susceptibility testing and is not a treatment "
              "recommendation.")


class ModelError(RuntimeError):
    """A model file is missing, unreadable or does not fit the input."""


def card(bundle: dict[str, Any]) -> dict[str, Any]:
    """The bundle without the fitted pipeline (safe to write as JSON)."""
    return {k: v for k, v in bundle.items() if k != "pipeline"}


def save_bundle(bundle: dict[str, Any], path: Path) -> Path:
    if bundle.get("format") != BUNDLE_FORMAT or "pipeline" not in bundle:
        raise ModelError("Not a model bundle (format marker or pipeline missing).")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    path.with_suffix(".json").write_text(json.dumps(card(bundle), indent=2, default=str), encoding="utf-8")
    return path


def load_bundle(path: str | Path, n_jobs: int | None = 1) -> dict[str, Any]:
    """Load a bundle. `n_jobs` sets the model's thread count (1 is fastest for single spectra because no
    threads are started per call; None keeps the saved setting). Predictions do not depend on it."""
    path = Path(path)
    if not path.is_file():
        raise ModelError(f"Model file not found: {path}. Train and save a model first with "
                         "'python scripts/train_baselines.py --evaluate-test'.")
    try:
        bundle = joblib.load(path)
    except Exception as exc:                      # joblib raises many different errors for damaged files
        raise ModelError(f"Could not read the model file {path} ({type(exc).__name__}).") from exc
    if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
        raise ModelError(f"{path} is not a model bundle of this project.")
    model = bundle["pipeline"].steps[-1][1]
    if n_jobs is not None and "n_jobs" in model.get_params():
        model.set_params(n_jobs=n_jobs)
    return bundle


def predict_features(bundle: dict[str, Any], features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resistance probabilities and 0/1 labels for one feature vector or a matrix of them."""
    X = np.asarray(features, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    n_features = int(bundle["n_features"])
    if X.ndim != 2 or X.shape[1] != n_features:
        raise ModelError(f"Wrong number of features: the model expects {n_features} per spectrum, "
                         f"got {X.shape[-1] if X.ndim else 0}.")
    if not np.isfinite(X).all():
        raise ModelError("The features contain missing or infinite values.")
    prob = bundle["pipeline"].predict_proba(X)[:, 1]
    return prob, (prob >= float(bundle["threshold"])).astype(np.int64)


@dataclass
class Prediction:
    species: str
    antibiotic: str
    prediction: str
    resistance_probability: float
    threshold: float
    model_version: str
    preprocessing_ms: float
    inference_ms: float
    total_ms: float
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def predict_spectrum_file(bundle: dict[str, Any], path: str | Path) -> Prediction:
    """Read and preprocess a raw spectrum file exactly as in training, then predict (timed)."""
    pcfg = PreprocessingConfig.from_dict(bundle["preprocessing"])
    if pcfg.fingerprint() != bundle["feature_fingerprint"]:
        raise ModelError("The preprocessing settings stored in the model do not match its feature fingerprint.")
    started = time.perf_counter()
    features, _ = preprocess_file(Path(path), pcfg)          # raises SpectrumFormatError for invalid files
    preprocessed = time.perf_counter()
    prob, label = predict_features(bundle, features)
    finished = time.perf_counter()
    return Prediction(
        species=bundle["species"], antibiotic=bundle["antibiotic"],
        prediction="Resistant" if label[0] == 1 else "Susceptible",
        resistance_probability=round(float(prob[0]), 4), threshold=round(float(bundle["threshold"]), 4),
        model_version=bundle["model_version"],
        preprocessing_ms=round((preprocessed - started) * 1000, 2),
        inference_ms=round((finished - preprocessed) * 1000, 2),
        total_ms=round((finished - started) * 1000, 2),
    )
