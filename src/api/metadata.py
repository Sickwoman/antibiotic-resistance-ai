"""Everything `GET /model-info` serves, built from an explicit allow-list of declared fields.

Never a bundle or a saved report dumped wholesale: both carry values that must not be served — the dataset
and feature fingerprints, `x_sha256`, the git commit, the code fingerprint, absolute paths and the DRIAMS
archive checksums. Numbers taken from a saved report pass through `finite_or_none`, because a report can
hold a bare NaN and starlette renders JSON with `allow_nan=False`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.api.schemas import (
    AlgorithmInfo,
    ExternalResult,
    LatencyInfo,
    LimitsInfo,
    MetricSet,
    ModelInfoResponse,
    TargetInfo,
    TrainingDataInfo,
    ZoneInfo,
    finite_or_none,
)
from src.predict import DISCLAIMER
from src.uncertainty import RESISTANT, SUSCEPTIBLE, UNCERTAIN, Zones
from src.utils import get_logger

if TYPE_CHECKING:                        # annotation only; importing it at runtime would be a cycle
    from src.api.inference_service import Service

log = get_logger("api")

# The model whose external results the append-only log records. /model-info reports those results only
# when the served bundle is this model; serving anything else (a test bundle, say) reports none rather
# than attaching one model's numbers to another.
PROJECT_MODEL_VERSION = "v0.4.0-tuned_lightgbm-random-seed42"

# Rows of results/experiments/test_evaluations.csv that describe the served model away from home, and the
# internal test row. Read from the log so no number in a response is ever typed by hand.
EXTERNAL_EXPERIMENTS: dict[str, tuple[str, str]] = {
    "external__DRIAMS-B__saved_model": ("DRIAMS-B", "0.7"),
    "external__DRIAMS-D__saved_model": ("DRIAMS-D", "0.7"),
    "c_holdout__B1_baseline_saved": ("DRIAMS-C", "0.8"),
}
INTERNAL_TEST_EXPERIMENT = "random"

EXTERNAL_NOTE = (
    "Results for this same saved model at hospitals it was not trained on, read from the project's "
    "append-only evaluation log. They are reported here because the internal test figure alone would "
    "overstate the model. Version 0.7 found no measurable generalisation gap and Version 0.8's "
    "recalibration result was null, so these numbers are not evidence that the model transfers."
)
INTENDED_USE = (
    "Research only. This service returns a resistance probability for one species-antibiotic pair from a "
    "MALDI-TOF spectrum, for methodological research on adaptive AI. It does not diagnose, does not "
    "replace laboratory antimicrobial susceptibility testing, and never indicates which antibiotic a "
    "patient should receive."
)


def read_log_rows(log_path: Path, model_version: str) -> tuple[list[MetricSet], list[ExternalResult]]:
    """The served model's internal test row and its results at other sites, read from the log.

    Returns empty lists when the log is absent or the served bundle is not the project model, so a test
    bundle never inherits the project model's numbers.
    """
    if model_version != PROJECT_MODEL_VERSION or not log_path.is_file():
        return [], []
    try:
        table = pd.read_csv(log_path)
    except (OSError, pd.errors.ParserError):
        return [], []

    internal: list[MetricSet] = []
    rows = table[(table["experiment"] == INTERNAL_TEST_EXPERIMENT) & (table["model"] == "tuned_lightgbm")
                 & (table["seed"] == 42) & (table["stage"] == "v0.4-tuned")]
    if len(rows):
        r = rows.iloc[-1]
        internal.append(MetricSet(
            part="internal test (held-out patients, DRIAMS-A)", n=int(r["n"]), n_resistant=int(r["n_resistant"]),
            roc_auc=finite_or_none(r["roc_auc"]), pr_auc=finite_or_none(r["pr_auc"]),
            brier=finite_or_none(r["brier"]), sensitivity=finite_or_none(r["sensitivity"]),
            specificity=finite_or_none(r["specificity"]), precision=finite_or_none(r["precision"]),
        ))

    external: list[ExternalResult] = []
    for key, (site, version) in EXTERNAL_EXPERIMENTS.items():
        rows = table[table["experiment"] == key]
        if not len(rows):
            continue
        r = rows.iloc[-1]
        external.append(ExternalResult(
            site=site, version=version, n=int(r["n"]), n_resistant=int(r["n_resistant"]),
            roc_auc=finite_or_none(r["roc_auc"]), brier=finite_or_none(r["brier"]),
        ))
    return internal, external


def read_timing(path: Path) -> dict[str, Any]:
    """The Version 0.4 inference_timing.json, or an empty dict when it is not there."""
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def zone_info(zones: Zones | None, bundle: dict[str, Any]) -> ZoneInfo:
    """Describe the confidence zones, including the side that does not exist.

    The Version 0.6 fit reached its 0.95 target on the susceptible side only; the resistant side's best
    achievable precision at 5 % coverage was 0.84, so `upper` is null and no high-confidence-resistant
    zone exists. The API states that, because a consumer should not have to infer it from an absence.
    """
    if zones is None:
        return ZoneInfo(
            fitted_on="not loaded", n_fitted=0, target_npv=0.0, target_precision=0.0, min_coverage=0.0,
            susceptible_edge=None, resistant_edge=None, possible_labels=["not available"],
            note="No confidence zone file is loaded, so no three-way confidence label is reported.",
        )
    d = zones.to_dict()
    rule = d.get("rule") or {}
    lower, upper = finite_or_none(d.get("lower")), finite_or_none(d.get("upper"))

    # Exactly the labels Zones.label() can produce for a probability in [0, 1]: a side with no edge is a
    # label that is never returned. Verified against the fitted file, where only two of the three occur.
    labels: list[str] = []
    if lower is not None:
        labels.append(SUSCEPTIBLE)
    labels.append(UNCERTAIN)
    if upper is not None:
        labels.append(RESISTANT)

    missing = [name for name, edge in (("susceptible", lower), ("resistant", upper)) if edge is None]
    if not missing:
        note = "Probabilities between the two edges are reported as uncertain. "
    elif missing == ["resistant"]:
        note = "Every probability above the susceptible edge is reported as uncertain. "
    elif missing == ["susceptible"]:
        note = "Every probability below the resistant edge is reported as uncertain. "
    else:
        note = "Every probability is reported as uncertain. "
    if missing:
        label_word = "that label is" if len(missing) == 1 else "those labels are"
        note += (f"No high-confidence {' or '.join(missing)} zone exists: that side could not reach its "
                 f"pre-registered target at the minimum coverage, so {label_word} never returned. ")
    note += "The zones change what a prediction is called, never the decision threshold."
    return ZoneInfo(
        fitted_on=str(d.get("fitted_on", "validation")), n_fitted=int(d.get("n_fitted", 0) or 0),
        target_npv=float(rule.get("target_npv", 0.95)), target_precision=float(rule.get("target_precision", 0.95)),
        min_coverage=float(rule.get("min_coverage", 0.05)), susceptible_edge=lower, resistant_edge=upper,
        possible_labels=labels, note=note,
    )


def model_info(service: Service) -> ModelInfoResponse:
    """Everything /model-info serves, field by declared field. Never a bundle dumped wholesale."""
    bundle = service.require_model()
    pre = bundle.get("preprocessing") or {}
    cal = bundle.get("calibration") or {}
    ds = bundle.get("dataset") or {}
    split = bundle.get("split") or {}
    rule = bundle.get("threshold_rule") or {}
    t = service.timing

    n_bins = int(pre.get("n_bins") or bundle.get("n_features", 0))
    feature_definition = (f"{n_bins} bins of {pre.get('bin_width')} Da from m/z {pre.get('mz_min')} to "
                          f"{pre.get('mz_max')}, {pre.get('intensity_transform')} intensities, "
                          f"{pre.get('baseline_method')} baseline, {pre.get('normalization')} normalised")
    calibration = (f"{cal.get('method', 'none')} on {cal.get('fitted_on', 'unknown')}"
                   if cal else "none")
    threshold_rule = (f"{rule.get('rule', 'unknown')}"
                      + (f" (min_sensitivity={rule.get('min_sensitivity')})" if rule.get("min_sensitivity") else ""))

    return ModelInfoResponse(
        api_version=service.api_version,
        model_version=str(bundle["model_version"]),
        algorithm=AlgorithmInfo(
            name=str(bundle.get("model", "unknown")), kind=str(bundle.get("model_kind", "unknown")),
            hyperparameters=dict(bundle.get("params") or {}), calibration=calibration,
            n_features=int(bundle["n_features"]), feature_definition=feature_definition,
        ),
        target=TargetInfo(
            species=str(bundle["species"]), antibiotic=str(bundle["antibiotic"]),
            label_map={str(k): str(v) for k, v in (bundle.get("label_map") or {}).items()},
            intermediate_as=str(ds.get("intermediate_as", "resistant")),
        ),
        threshold=float(bundle["threshold"]),
        threshold_rule=threshold_rule,
        training_data=TrainingDataInfo(
            dataset=str(ds.get("name", "unknown")),
            source="DRIAMS, a public de-identified MALDI-TOF database (Weis et al.); no patient "
                   "identifiers are used or stored by this service",
            training_sites=[str(s) for s in (split.get("train_sites") or ds.get("sites") or [])],
            n_training_spectra=int(split.get("train_size", 0) or 0),
            grouping="splits are by patient, so no patient appears in more than one part",
        ),
        evaluation=service.internal,
        external_validation=service.external,
        external_validation_note=EXTERNAL_NOTE,
        confidence_zones=zone_info(service.zones, bundle),
        latency=LatencyInfo(
            measured_on_samples=int(t.get("samples", 0) or 0),
            preprocessing_ms_median=finite_or_none((t.get("preprocessing_ms") or {}).get("median")),
            inference_ms_median=finite_or_none((t.get("inference_ms") or {}).get("median")),
            total_ms_median=finite_or_none((t.get("total_ms") or {}).get("median")),
            total_ms_p95=finite_or_none((t.get("total_ms") or {}).get("p95")),
            batch_inference_ms_per_sample=finite_or_none(t.get("batch_inference_ms_per_sample")),
            note="Measured on this machine for the saved model, not a claim about other hardware. Each "
                 "/predict response also carries its own measured timings.",
        ),
        limits=LimitsInfo(
            max_upload_bytes=service.limits.max_upload_bytes, max_raw_points=service.limits.max_raw_points,
            max_batch_files=service.limits.max_batch_files, max_batch_bytes=service.limits.max_batch_bytes,
            allowed_suffixes=list(service.limits.allowed_suffixes),
        ),
        intended_use=INTENDED_USE,
        disclaimer=DISCLAIMER,
    )


# ------------------------------------------------------------------------------------------------
# The application
# ------------------------------------------------------------------------------------------------
