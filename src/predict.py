"""Save and load trained model bundles and make research predictions from spectra.

A bundle is one joblib file holding the fitted pipeline together with everything needed to reproduce
and describe it (preprocessing settings, threshold, dataset fingerprints, metrics, versions). A JSON
card with the same information, minus the pipeline, is written next to it.

Version 0.6 adds two optional parts to a single-spectrum prediction, neither of which changes the
probability or the cut-off: the three-way confidence label of a fitted zone file (src.uncertainty) and
the m/z regions that moved this one prediction (src.explain, imported only when asked for).

Security: joblib files can execute code when loaded. Only load bundles produced by this project,
never files uploaded by users.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.preprocessing import PreprocessingConfig, preprocess_file
from src.tuning import set_threads
from src.uncertainty import UncertaintyError, Zones

BUNDLE_FORMAT = "amr-model-bundle/1"
DISCLAIMER = ("AI research prediction from a research prototype. It is not a clinically validated diagnostic, "
              "does not replace laboratory antimicrobial susceptibility testing and is not a treatment "
              "recommendation.")
# What a prediction reports when no confidence zones were given: the three-way label is not simply absent,
# it was never computed, and saying so is not the same as calling the isolate uncertain.
NO_CONFIDENCE = "not available"
EXPLAIN_REGIONS = 5                   # m/z regions reported for one prediction


class ModelError(RuntimeError):
    """A model file is missing, unreadable or does not fit the input."""


def card(bundle: dict[str, Any]) -> dict[str, Any]:
    """The bundle without the fitted pipeline (safe to write as JSON)."""
    return {k: v for k, v in bundle.items() if k != "pipeline"}


# The fields the prediction path reads. Checking them at load time turns a confusing KeyError deep inside
# a prediction into one clear message about the file. Fields that only some scripts use (model, dataset,
# split, metrics) are not required here; those scripts check what they need.
REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "model_version": str, "pipeline": object, "threshold": (int, float, np.floating),
    "n_features": (int, np.integer), "preprocessing": dict, "feature_fingerprint": str,
    "species": str, "antibiotic": str,
}


def check_bundle(bundle: dict[str, Any], path: str | Path = "<bundle>") -> None:
    """Every field the prediction code reads is present, of the right type and in range."""
    for field, kind in REQUIRED_FIELDS.items():
        if field not in bundle:
            raise ModelError(f"{path} is missing the {field!r} field; it was not written by this project.")
        if kind is not object and not isinstance(bundle[field], kind):
            raise ModelError(f"{path}: {field!r} should be {kind}, found {type(bundle[field]).__name__}.")
    threshold = float(bundle["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ModelError(f"{path}: threshold {threshold} is not a probability between 0 and 1.")
    if int(bundle["n_features"]) < 1:
        raise ModelError(f"{path}: n_features must be positive, found {bundle['n_features']}.")
    if not hasattr(bundle["pipeline"], "predict_proba"):
        raise ModelError(f"{path}: the saved pipeline cannot produce probabilities.")


def file_digest(path: Path) -> str:
    """SHA-256 of a file, read in blocks so a large model does not have to fit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_bundle(bundle: dict[str, Any], path: Path) -> Path:
    if bundle.get("format") != BUNDLE_FORMAT or "pipeline" not in bundle:
        raise ModelError("Not a model bundle (format marker or pipeline missing).")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    path.with_suffix(".json").write_text(json.dumps(card(bundle), indent=2, default=str), encoding="utf-8")
    checksum_path(path).write_text(f"{file_digest(path)}  {path.name}\n", encoding="utf-8")
    return path


def checksum_path(path: Path) -> Path:
    return Path(path).with_name(Path(path).name + ".sha256")


def verify_digest(path: Path) -> str | None:
    """Compare the file with the checksum written beside it. Returns the digest, or None if there is none.

    Loading a joblib file executes the code inside it, so the file must be one this project wrote. The
    checksum catches a corrupted or swapped file; it is not a signature and cannot stop someone who can
    write both files, which is why the rule stays 'only load bundles produced by this project'.
    """
    path = Path(path)
    sidecar = checksum_path(path)
    if not sidecar.is_file():
        return None
    expected = sidecar.read_text(encoding="utf-8").split()[0].strip().lower()
    actual = file_digest(path)
    if expected != actual:
        raise ModelError(f"{path} does not match its checksum in {sidecar.name} (expected {expected[:16]}..., "
                         f"found {actual[:16]}...). The file changed after it was saved; do not load it.")
    return actual


def load_bundle(path: str | Path, n_jobs: int | None = 1) -> dict[str, Any]:
    """Load a bundle. `n_jobs` sets the model's thread count (1 is fastest for single spectra because no
    threads are started per call; None keeps the saved setting). Predictions do not depend on it."""
    path = Path(path)
    if not path.is_file():
        raise ModelError(f"Model file not found: {path}. Train and save a model first with "
                         "'python scripts/train_baselines.py --evaluate-test'.")
    verify_digest(path)                           # refuse a file that changed after this project saved it
    try:
        bundle = joblib.load(path)
    except Exception as exc:                      # joblib raises many different errors for damaged files
        raise ModelError(f"Could not read the model file {path} ({type(exc).__name__}).") from exc
    if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
        raise ModelError(f"{path} is not a model bundle of this project.")
    check_bundle(bundle, path)
    if n_jobs is not None:
        set_threads(bundle["pipeline"], n_jobs)       # also reaches models inside calibration wrappers
    return bundle


def load_zones(path: str | Path) -> Zones | None:
    """The confidence zones written by Version 0.6 (a JSON file holding `Zones.to_dict()`).

    Returns None when there is no such file: a prediction without zones is still a prediction, it just
    does not carry a three-way confidence label. A file that exists but is not a zones record is an
    error, because silently ignoring it would report "not available" for a model that has zones.
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelError(f"Could not read the confidence zones file {path} ({type(exc).__name__}).") from exc
    if not isinstance(data, dict):
        raise ModelError(f"{path} does not hold a confidence-zones record (found a "
                         f"{type(data).__name__}, expected an object with threshold, lower and upper).")
    try:
        return Zones.from_dict(data)
    except (UncertaintyError, TypeError, ValueError) as exc:   # UncertaintyError for a record that is not one,
        raise ModelError(f"{path} is not a usable confidence-zones record: {exc}") from exc  # the rest for junk


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


def explain_regions(bundle: dict[str, Any], features: np.ndarray, pcfg: PreprocessingConfig,
                    keep: int = EXPLAIN_REGIONS) -> list[dict[str, Any]] | None:
    """The m/z regions that moved this one prediction most, as plain dicts, or None if the saved model
    cannot be explained this way.

    `contribution` is signed on the model's margin (positive = pushed towards resistant), never a share
    of risk: the calibration step after it is monotone but not additive. Regions are ranked by absolute
    contribution and reported as src.explain returns them, so one the model barely used can appear with a
    contribution that rounds to 0.0 while `towards` still shows the sign it had. src.explain is imported
    here and not at module level, so an ordinary prediction does not depend on the explanation code.
    """
    from src.explain import ExplainError, explain_one

    try:
        frame = explain_one(bundle["pipeline"], features, pcfg, keep=keep)
    except ExplainError:
        return None            # exact contributions need a tree model; the prediction itself still stands
    return [{"rank": int(row.rank), "mz_start": round(float(row.mz_start), 1),
             "mz_end": round(float(row.mz_end), 1), "contribution": round(float(row.total_signed), 4),
             "towards": str(row.towards)} for row in frame.itertuples(index=False)]


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
    confidence: str = NO_CONFIDENCE                        # the three-way label, when zones were given
    explanation: list[dict[str, Any]] | None = None        # top m/z regions, when asked for
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def predict_spectrum_file(bundle: dict[str, Any], path: str | Path, *, zones: Zones | None = None,
                          explain: bool = False) -> Prediction:
    """Read and preprocess a raw spectrum file exactly as in training, then predict (timed).

    `zones` adds the three-way confidence label of a fitted zone file; it does not move the model's
    cut-off, so `prediction` is unchanged by it. `explain=True` adds the m/z regions that moved this
    prediction, and is left out (explanation stays None) for a model that cannot give exact
    contributions. The three reported times cover reading, preprocessing and inference only: labelling
    and explaining happen after the clock has stopped, so the numbers mean the same thing whatever the
    two options are set to.
    """
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
        confidence=NO_CONFIDENCE if zones is None else zones.one(float(prob[0])),
        explanation=explain_regions(bundle, features, pcfg) if explain else None,
    )
