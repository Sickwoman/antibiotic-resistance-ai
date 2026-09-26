"""Declared response models for the Version 0.9 backend API, and the rule about what may never be served.

Every response is built from a model declared here. Nothing is serialised by reflection off a saved bundle
or a saved JSON report, because both carry fields the API must not expose: the dataset and feature
fingerprints, `x_sha256`, the git commit, the code fingerprint, absolute paths and the DRIAMS archive
checksums.

A saved report can also carry a bare `NaN` (`uncertainty.json` does, at `sides.resistant.achieved`), and
starlette renders JSON with `allow_nan=False`, so an un-sanitised number taken from a report would raise
`ValueError: Out of range float values are not JSON compliant` at render time and become a 500. Every
number that comes from a report therefore passes through `finite_or_none`.

See docs/v0.9_api_plan.md and protocol amendment 5.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# Strings that must not appear in any response body. The tests assert each one's absence from every
# endpoint, so this is a checked list rather than a remembered rule. The two fingerprints and the code
# fingerprint are 16 hex characters, and the dataset digest is given as a prefix, so none of them can
# collide with the 32-hex patient-identifier pattern that tests/test_privacy.py scans for.
FORBIDDEN_IN_RESPONSES: tuple[str, ...] = (
    "347cbd6d5d956ff9",        # feature_fingerprint
    "151415a4d03dbcc5",        # dataset.row_fingerprint / split.dataset_fingerprint
    "c27609b6f2663e0d",        # dataset.x_sha256 (prefix)
    "1eebdbe2681c1904",        # code_fingerprint
    "feature_fingerprint",
    "row_fingerprint",
    "dataset_fingerprint",
    "x_sha256",
    "code_fingerprint",
    "git_commit",
    "driams_root",
    "C:/DRIAMS",
    "C:\\DRIAMS",
)


def finite_or_none(value: Any) -> float | None:
    """A float safe to put in a response, or None where a saved report holds NaN or an infinity."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class ApiModel(BaseModel):
    """Base for every response model.

    `protected_namespaces=()` is required because several fields are named `model_*` (`model_version`,
    `model_loaded`), which collides with pydantic's own reserved prefix. Those names are part of the
    published contract and match what the command-line tool already prints, so the base class is
    adjusted rather than the field names.
    """

    model_config = ConfigDict(protected_namespaces=())


# ----------------------------------------------------------------------------------------- errors
# The public error vocabulary: a closed set, so a client can branch on it safely. These names are part of
# the published contract and may only be added to, never renamed or repurposed.
CODE_INVALID_REQUEST = "invalid_request"                # the request itself is malformed
CODE_INVALID_SPECTRUM = "invalid_spectrum"              # the spectrum cannot be read or preprocessed
CODE_PAYLOAD_TOO_LARGE = "payload_too_large"            # over a configured byte or file-count limit
CODE_UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"  # not an accepted file suffix
CODE_SERVICE_NOT_READY = "service_not_ready"            # a required artifact did not load
CODE_INTERNAL_ERROR = "internal_error"                  # anything unexpected

ERROR_CODES: tuple[str, ...] = (
    CODE_INVALID_REQUEST, CODE_INVALID_SPECTRUM, CODE_PAYLOAD_TOO_LARGE,
    CODE_UNSUPPORTED_MEDIA_TYPE, CODE_SERVICE_NOT_READY, CODE_INTERNAL_ERROR,
)


class ErrorBody(ApiModel):
    """A stable public error.

    `code` is the field to branch on: one of ERROR_CODES, and nothing else. `type` is the internal
    exception class name and is **deprecated** — it is retained for one release so existing callers keep
    working, and will be removed. Nothing in the message is a traceback or a filesystem path.
    """

    code: str
    message: str
    type: str | None = None          # deprecated; scheduled for removal


class ErrorResponse(ApiModel):
    error: ErrorBody


# ------------------------------------------------------------------------ liveness and readiness
class LivenessResponse(ApiModel):
    """`GET /health`: the process is running. Nothing more.

    Deliberately carries no artifact state and no internals. A liveness probe that sees a failure here is
    entitled to restart the process, so this must not fail merely because a model did not load — that is
    what readiness is for.
    """

    status: Literal["ok"]


class ReadinessResponse(ApiModel):
    """`GET /ready`: whether every artifact the service needs actually loaded.

    200 when ready, 503 when not. `detail` is one of the fixed public strings, never a raw load error, so
    it cannot carry a filesystem path.
    """

    status: Literal["ready", "not_ready"]
    model_loaded: bool
    zones_loaded: bool
    model_version: str | None
    api_version: str
    uptime_s: float
    detail: str | None = None          # why the service is not ready, when it is not


# ------------------------------------------------------------------------------------- prediction
class Region(ApiModel):
    """One m/z region that moved a prediction. Named by its interval only: no protein is identified."""

    rank: int
    mz_start: float
    mz_end: float
    contribution: float
    towards: str


class PredictionResponse(ApiModel):
    """The contract for /predict, identical to what scripts/predict_spectrum.py prints.

    `advice` is present only when the confidence label is the uncertain one, exactly as in the command-line
    output. The endpoint therefore returns the payload dict built by `src.predict.prediction_payload` and
    validates it against this model, rather than returning a model instance that would always carry an
    `advice` key; that keeps the two outputs byte-comparable, which a test asserts.
    """

    species: str
    antibiotic: str
    prediction: str
    resistance_probability: float
    threshold: float
    model_version: str
    preprocessing_ms: float
    inference_ms: float
    total_ms: float
    confidence: str
    explanation: list[Region] | None = None
    disclaimer: str
    advice: str | None = None


class BatchItem(ApiModel):
    """One result in a batch. Exactly one of `prediction` and `error` is set."""

    index: int
    prediction: PredictionResponse | None = None
    error: ErrorBody | None = None


class BatchResponse(ApiModel):
    n_submitted: int
    n_succeeded: int
    n_failed: int
    results: list[BatchItem]
    disclaimer: str


# ------------------------------------------------------------------------------------- model info
class AlgorithmInfo(ApiModel):
    name: str
    kind: str
    hyperparameters: dict[str, Any]
    calibration: str
    n_features: int
    feature_definition: str


class TargetInfo(ApiModel):
    species: str
    antibiotic: str
    label_map: dict[str, str]
    intermediate_as: str


class TrainingDataInfo(ApiModel):
    dataset: str
    source: str
    training_sites: list[str]
    n_training_spectra: int
    grouping: str


class MetricSet(ApiModel):
    part: str
    n: int
    n_resistant: int
    roc_auc: float | None
    pr_auc: float | None
    brier: float | None
    sensitivity: float | None
    specificity: float | None
    precision: float | None


class ExternalResult(ApiModel):
    """How the served model performed at a site it was not trained on."""

    site: str
    version: str
    n: int
    n_resistant: int
    roc_auc: float | None
    brier: float | None


class ZoneInfo(ApiModel):
    fitted_on: str
    n_fitted: int
    target_npv: float
    target_precision: float
    min_coverage: float
    susceptible_edge: float | None
    resistant_edge: float | None
    possible_labels: list[str]
    note: str


class LatencyInfo(ApiModel):
    measured_on_samples: int
    preprocessing_ms_median: float | None
    inference_ms_median: float | None
    total_ms_median: float | None
    total_ms_p95: float | None
    batch_inference_ms_per_sample: float | None
    note: str


class LimitsInfo(ApiModel):
    max_upload_bytes: int
    max_raw_points: int
    max_batch_files: int
    max_batch_bytes: int
    allowed_suffixes: list[str]


class ModelInfoResponse(ApiModel):
    api_version: str
    model_version: str
    algorithm: AlgorithmInfo
    target: TargetInfo
    threshold: float
    threshold_rule: str
    training_data: TrainingDataInfo
    evaluation: list[MetricSet]
    external_validation: list[ExternalResult]
    external_validation_note: str
    confidence_zones: ZoneInfo
    latency: LatencyInfo
    limits: LimitsInfo
    intended_use: str
    disclaimer: str
