"""The Version 0.9 backend API: the frozen serving path of src/predict.py exposed over HTTP.

Four endpoints, fixed in docs/v0.9_api_plan.md and protocol amendment 5: GET /health, POST /predict,
POST /batch-predict, GET /model-info.

This module adds transport, input validation and an explicit response contract, and nothing else. It may
not fit, calibrate, tune, score a dataset part, or append to any log, and it takes no dataset name, split
name or filesystem path from a client, so a protected test part cannot be reached through it.

Three safety rules drive the code below, each with a reason that was verified rather than assumed:

- **An uploaded filename is never used or logged.** Raw DRIAMS files are named after the spectrum UUID and
  repeat it in their '#' comment lines. The filename is read for its suffix and then discarded, and the
  upload is written to a generated name, so even a library error message quoting `path.name` cannot leak
  it.
- **Uploaded bytes are data, never code.** Nothing from a request is deserialised, imported or executed,
  and nothing reaches joblib.load. A '.joblib' upload is refused at the suffix gate without being opened.
- **No response may carry a non-finite float.** starlette renders JSON with allow_nan=False, so a NaN taken
  from a saved report (uncertainty.json holds one) would raise at render time and become a 500. Numbers
  from saved reports go through src.api_schemas.finite_or_none.

Input limits are passed to the serving path as keyword arguments. They are deliberately not
PreprocessingConfig fields: that dataclass's hash is the feature fingerprint every saved bundle is checked
against, so a field there would invalidate every bundle and every cache.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

from src.api_schemas import (
    CODE_INTERNAL_ERROR,
    CODE_INVALID_REQUEST,
    CODE_INVALID_SPECTRUM,
    CODE_PAYLOAD_TOO_LARGE,
    CODE_SERVICE_NOT_READY,
    CODE_UNSUPPORTED_MEDIA_TYPE,
    ERROR_CODES,
    AlgorithmInfo,
    BatchItem,
    BatchResponse,
    ErrorBody,
    ExternalResult,
    LatencyInfo,
    LimitsInfo,
    LivenessResponse,
    MetricSet,
    ModelInfoResponse,
    PredictionResponse,
    ReadinessResponse,
    TargetInfo,
    TrainingDataInfo,
    ZoneInfo,
    finite_or_none,
)
from src.data_loader import DataError
from src.predict import DISCLAIMER, ModelError, load_bundle, load_zones, predict_spectrum_file, prediction_payload
from src.uncertainty import ADVICE, RESISTANT, SUSCEPTIBLE, UNCERTAIN, Zones
from src.utils import ConfigError, get_logger, project_path

log = get_logger("api")

# The serving contract's own version, deliberately a constant here rather than config.yaml ->
# project.version. That field tracks the project/experiment state (it was stale at "0.7.0" through the
# whole of Version 0.8), so sourcing a public API version from it would let an unrelated edit change the
# contract's identity. The model carries its own separate version and the two are never conflated.
API_VERSION = "0.9.0"

# What a degraded /health may say. Load failures carry a filesystem path in their message, so the public
# text is one of these fixed strings and the real exception goes to the log only.
DETAIL_MODEL_UNAVAILABLE = "The configured model could not be loaded, so predictions are unavailable."
DETAIL_ZONES_UNUSABLE = ("The confidence-zone file exists but is not a usable zones record, so the "
                         "service will not serve predictions whose confidence label would silently "
                         "read 'not available'.")

CHUNK_BYTES = 64 * 1024                 # how much of an upload is read at a time
UPLOAD_STEM = "upload"                  # generated name for a saved upload: the client's is discarded

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


class UploadTooLarge(ValueError):
    """The request body is larger than the configured limit."""


class UnsupportedUpload(ValueError):
    """The uploaded file does not have an accepted suffix."""


class BadRequest(ValueError):
    """The request itself is malformed, independently of any spectrum's content."""


class TooManyFiles(ValueError):
    """More files were submitted in one batch than the configured limit."""


@dataclass
class Limits:
    max_upload_bytes: int = 4_000_000
    max_raw_points: int = 200_000
    max_batch_files: int = 20
    max_batch_bytes: int = 32_000_000
    allowed_suffixes: tuple[str, ...] = (".txt",)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Limits:
        lim = (cfg.get("api") or {}).get("limits") or {}
        return cls(
            max_upload_bytes=int(lim.get("max_upload_bytes", 4_000_000)),
            max_raw_points=int(lim.get("max_raw_points", 200_000)),
            max_batch_files=int(lim.get("max_batch_files", 20)),
            max_batch_bytes=int(lim.get("max_batch_bytes", 32_000_000)),
            allowed_suffixes=tuple(str(s).lower() for s in lim.get("allowed_suffixes", (".txt",))),
        )


@dataclass
class Service:
    """What the process loaded at start-up. A failure here degrades the service; it does not crash it."""

    limits: Limits
    api_version: str
    bundle: dict[str, Any] | None = None
    zones: Zones | None = None
    detail: str | None = None
    zones_failed: bool = False           # the file exists but is not a zones record: not the same as absent
    started_at: float = field(default_factory=time.perf_counter)
    external: list[ExternalResult] = field(default_factory=list)
    internal: list[MetricSet] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """Ready means every artifact the service needs loaded, not merely that a model is present.

        A *missing* zones file is fine — a prediction without a three-way label is still a prediction. A
        file that exists but cannot be parsed is not fine, because serving on would report "not available"
        for a model that does have zones, which is the silent failure load_zones exists to prevent.
        """
        return self.bundle is not None and not self.zones_failed

    def require_model(self) -> dict[str, Any]:
        """The loaded bundle, or a refusal whose message is safe to send to a client.

        It gates on `ready`, not merely on a bundle being present: an unusable zones file must stop
        predictions too, because serving on would report confidence "not available" for a model that does
        have zones. `self.detail` is already one of the fixed public strings, never a raw load error, so
        this message cannot carry a filesystem path.
        """
        if not self.ready:
            raise ModelError(self.detail or DETAIL_MODEL_UNAVAILABLE)
        return self.bundle


# ------------------------------------------------------------------------------------------------
# Uploads
# ------------------------------------------------------------------------------------------------

def check_suffix(filename: str | None, allowed: tuple[str, ...]) -> None:
    """Accept or refuse an upload on its suffix alone.

    Only the suffix is used; the filename itself is then discarded and never logged, because a raw DRIAMS
    file is named after its spectrum UUID. The refusal message names the allowed suffixes rather than
    echoing what was sent.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in allowed:
        raise UnsupportedUpload(f"Unsupported file type. This service accepts {', '.join(allowed)} "
                                "spectrum files only.")


async def stream_to_file(upload: UploadFile, dest: Path, max_bytes: int, budget: list[int] | None = None) -> int:
    """Write an upload to `dest` in chunks, stopping the moment it exceeds `max_bytes`.

    `budget` optionally carries a shared remaining-bytes allowance for a batch, so a batch cannot exceed
    its total by splitting the payload across files. Returns the number of bytes written.
    """
    total = 0
    with dest.open("wb") as handle:
        while True:
            chunk = await upload.read(CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise UploadTooLarge(f"The uploaded spectrum is larger than the {max_bytes}-byte limit.")
            if budget is not None:
                budget[0] -= len(chunk)
                if budget[0] < 0:
                    raise UploadTooLarge("The batch is larger than the configured total byte limit.")
            handle.write(chunk)
    return total


# ------------------------------------------------------------------------------------------------
# /model-info, built from an explicit allow-list
# ------------------------------------------------------------------------------------------------

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


def check_zone_bounds(zones: Zones) -> None:
    """Refuse a zones record whose edges are not usable probabilities.

    `Zones.from_dict` checks that two edges are ordered, but neither it nor `Zones.label` checks that an
    edge is finite or inside [0, 1] — `label` only validates the probability it is given. The two failures
    that follow are silent, which is why they are refused here rather than served:

    - an edge of NaN makes every comparison false, so that zone quietly disappears and every spectrum comes
      back "Uncertain" from a model that does have a zone;
    - an edge of +inf makes every comparison true, so every spectrum is labelled high-confidence.

    An out-of-range edge behaves like the first. All of them are treated exactly like an unparseable zones
    file: the service reports itself not ready rather than serving a label it cannot stand behind.
    """
    for name, edge in (("lower", zones.lower), ("upper", zones.upper)):
        if edge is None:
            continue                                  # a side that does not exist is a fitted outcome
        if not math.isfinite(float(edge)):
            raise ModelError(f"The {name} confidence-zone edge is {edge!r}, which is not a finite "
                             "probability, so every comparison against it would be meaningless.")
        if not 0.0 <= float(edge) <= 1.0:
            raise ModelError(f"The {name} confidence-zone edge is {edge!r}, outside the probability "
                             "range [0, 1].")


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

def default_model_path(config: dict[str, Any]) -> Path:
    api = config.get("api") or {}
    model_dir = api.get("model_dir") or (config.get("explain") or {}).get("model_dir")
    split = api.get("split") or (config.get("explain") or {}).get("split")
    if not model_dir or not split:
        raise ConfigError("config.yaml needs api.model_dir and api.split to say which bundle to serve.")
    return project_path(model_dir) / config["dataset"]["name"] / f"best_{split}.joblib"


def default_zones_path(config: dict[str, Any]) -> Path | None:
    api = config.get("api") or {}
    zones_dir = api.get("zones_dir") or (config.get("explain") or {}).get("report_dir")
    if not zones_dir:
        return None
    return project_path(zones_dir) / config["dataset"]["name"] / "uncertainty.json"


# The single mapping from an internal failure to what the client is told. Ordered most specific first,
# because DataError is the base of SpectrumFormatError and PreprocessingError. Keeping it in one table is
# the point: a handler cannot drift from the documented vocabulary if it does not choose the code itself.
ERROR_MAP: tuple[tuple[type[Exception], int, str], ...] = (
    (BadRequest, 400, CODE_INVALID_REQUEST),
    (UnsupportedUpload, 415, CODE_UNSUPPORTED_MEDIA_TYPE),
    (UploadTooLarge, 413, CODE_PAYLOAD_TOO_LARGE),
    (TooManyFiles, 413, CODE_PAYLOAD_TOO_LARGE),
    (ModelError, 503, CODE_SERVICE_NOT_READY),
    (DataError, 422, CODE_INVALID_SPECTRUM),
)


def classify(exc: Exception) -> tuple[int, str]:
    """(status, public code) for an exception, from ERROR_MAP; anything unmapped is an internal error."""
    for kind, status, code in ERROR_MAP:
        if isinstance(exc, kind):
            return status, code
    return 500, CODE_INTERNAL_ERROR


def error_response(exc: Exception) -> JSONResponse:
    """A code and a message, never a traceback and never a path.

    Mirrors the 'Error: <message>' convention of the project's scripts. `type` is still filled in for one
    release so existing callers do not break, but `code` is the field the contract promises.
    """
    status, code = classify(exc)
    body = ErrorBody(code=code, message=str(exc), type=type(exc).__name__)
    return JSONResponse(status_code=status, content={"error": body.model_dump()})


def create_app(config: dict[str, Any], *, model_path: Path | None = None,
               zones_path: Path | None = None) -> FastAPI:
    """Build the application. A factory, so tests can serve a synthetic bundle with no global state."""
    limits = Limits.from_config(config)
    api_cfg = config.get("api") or {}
    service = Service(limits=limits, api_version=API_VERSION)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = model_path or default_model_path(config)
        try:
            service.bundle = load_bundle(path, n_jobs=int(api_cfg.get("request_threads", 1)))
            log.info("serving model %s", service.bundle["model_version"])
        except (ModelError, DataError, ConfigError) as exc:
            # str(exc) names the file that was missing or unreadable, i.e. an absolute path. It must not
            # reach a response, so the public detail is a fixed string and the real reason is logged.
            service.detail = DETAIL_MODEL_UNAVAILABLE
            log.error("no model loaded, the service will report degraded: %s", exc)
        zp = zones_path if zones_path is not None else default_zones_path(config)
        if service.bundle is not None and zp is not None:
            try:
                loaded = load_zones(zp)
                if loaded is not None:
                    check_zone_bounds(loaded)         # finite, in range: from_dict checks neither
                service.zones = loaded
            except (ModelError, DataError) as exc:
                # The message names the file, so only the fixed public string reaches a response.
                service.zones_failed = True
                service.detail = DETAIL_ZONES_UNUSABLE
                log.error("confidence zones could not be loaded, the service will report degraded: %s", exc)
        if service.bundle is not None:
            model_version = str(service.bundle["model_version"])
            log_path = project_path((config.get("evaluation") or {}).get("test_log")
                                    or "results/experiments/test_evaluations.csv")
            service.internal, service.external = read_log_rows(log_path, model_version)
            timing_dir = api_cfg.get("model_timing_dir")
            if timing_dir:
                service.timing = read_timing(project_path(timing_dir) / config["dataset"]["name"]
                                             / "inference_timing.json")
        service.started_at = time.perf_counter()
        yield

    app = FastAPI(
        title="Antibiotic resistance research API",
        version=API_VERSION,
        summary="Research predictions of ciprofloxacin resistance in Escherichia coli from MALDI-TOF "
                "spectra. Not a clinical diagnostic.",
        lifespan=lifespan,
    )
    app.state.service = service

    # ---------------------------------------------------------------------------- error handling
    for _kind, _status, _code in ERROR_MAP:
        # One handler per mapped class, all sharing error_response, so the status and the code can only
        # come from ERROR_MAP. DataError last in the table also catches SpectrumFormatError and
        # PreprocessingError, which subclass it.
        app.add_exception_handler(_kind, lambda request, exc: error_response(exc))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("an unexpected error reached the API: %s", exc)     # traceback to the log only
        body = ErrorBody(code=CODE_INTERNAL_ERROR, type="InternalError",
                         message="The request could not be completed because of an internal error.")
        return JSONResponse(status_code=500, content={"error": body.model_dump()})

    # ---------------------------------------------------------------------------- body size guard
    @app.middleware("http")
    async def refuse_oversized_body(request: Request, call_next):
        """Refuse on a declared Content-Length before the body is consumed.

        The per-file streaming counter is the authoritative limit; this only avoids reading a body that
        has already announced it is too big.
        """
        cap = limits.max_batch_bytes if request.url.path == "/batch-predict" else limits.max_upload_bytes
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > cap:
            return error_response(UploadTooLarge(
                f"The request body is {declared} bytes, above the {cap}-byte limit for this endpoint."))
        return await call_next(request)

    # ---------------------------------------------------------------------------- endpoints
    @app.get("/health", response_model=LivenessResponse)
    async def health() -> JSONResponse:
        """Liveness only: this answers 200 for as long as the process is serving requests at all.

        It deliberately does not consult the model. A liveness probe treats a failure as "restart me", and
        restarting will not make a missing model appear; that state belongs to /ready.
        """
        return JSONResponse(status_code=200, content=LivenessResponse(status="ok").model_dump())

    @app.get("/ready", response_model=ReadinessResponse)
    async def ready() -> JSONResponse:
        is_ready = service.ready
        body = ReadinessResponse(
            status="ready" if is_ready else "not_ready", model_loaded=service.bundle is not None,
            zones_loaded=service.zones is not None,
            model_version=str(service.bundle["model_version"]) if service.bundle is not None else None,
            api_version=service.api_version,
            uptime_s=round(time.perf_counter() - service.started_at, 3),
            detail=None if is_ready else service.detail,
        )
        return JSONResponse(status_code=200 if is_ready else 503, content=body.model_dump())

    @app.get("/model-info", response_model=ModelInfoResponse)
    async def info() -> JSONResponse:
        return JSONResponse(content=model_info(service).model_dump())

    @app.post("/predict", responses={200: {"model": PredictionResponse}})
    async def predict(request: Request, file: Annotated[UploadFile, File()],
                      explain: bool = False) -> JSONResponse:
        bundle = service.require_model()
        # Exactly one spectrum. FastAPI binds a single UploadFile even when several were sent, so without
        # this the caller would get a 200 for a request only one part of which was used.
        submitted = len((await request.form()).getlist("file"))
        if submitted != 1:
            raise BadRequest(f"Send exactly one spectrum file in the 'file' field; {submitted} were sent. "
                             "Use /batch-predict for several.")
        check_suffix(file.filename, limits.allowed_suffixes)
        with TemporaryDirectory() as tmp:
            # A generated name: the client's filename is a DRIAMS UUID and must not reach a message or a log.
            dest = Path(tmp) / f"{UPLOAD_STEM}.txt"
            await stream_to_file(file, dest, limits.max_upload_bytes)
            result = predict_spectrum_file(bundle, dest, zones=service.zones, explain=explain,
                                           max_points=limits.max_raw_points,
                                           max_bytes=limits.max_upload_bytes)
        payload = prediction_payload(result)
        PredictionResponse.model_validate(payload)      # the declared contract, checked before it is sent
        return JSONResponse(content=payload)

    @app.post("/batch-predict", response_model=BatchResponse)
    async def batch_predict(files: Annotated[list[UploadFile], File()],
                            explain: bool = False) -> JSONResponse:
        bundle = service.require_model()
        if len(files) > limits.max_batch_files:
            raise TooManyFiles(f"{len(files)} files were submitted; this service accepts at most "
                               f"{limits.max_batch_files} per batch.")
        budget = [limits.max_batch_bytes]
        items: list[BatchItem] = []
        with TemporaryDirectory() as tmp:
            for index, upload in enumerate(files):
                dest = Path(tmp) / f"{UPLOAD_STEM}_{index}.txt"       # generated, never the client's name
                try:
                    check_suffix(upload.filename, limits.allowed_suffixes)
                    await stream_to_file(upload, dest, limits.max_upload_bytes, budget)
                    result = predict_spectrum_file(bundle, dest, zones=service.zones, explain=explain,
                                                   max_points=limits.max_raw_points,
                                                   max_bytes=limits.max_upload_bytes)
                    payload = prediction_payload(result)
                    items.append(BatchItem(index=index, prediction=PredictionResponse.model_validate(payload)))
                except (UnsupportedUpload, UploadTooLarge, DataError) as exc:
                    # One unusable file does not fail the batch; it is reported in its own slot.
                    items.append(BatchItem(index=index, error=ErrorBody(
                        code=classify(exc)[1], message=str(exc), type=type(exc).__name__)))
                finally:
                    dest.unlink(missing_ok=True)
        failed = sum(1 for i in items if i.error is not None)
        body = BatchResponse(n_submitted=len(files), n_succeeded=len(items) - failed, n_failed=failed,
                             results=items, disclaimer=DISCLAIMER)
        return JSONResponse(content=body.model_dump(exclude_none=True))

    return app


__all__ = ["ADVICE", "API_VERSION", "DETAIL_MODEL_UNAVAILABLE", "DETAIL_ZONES_UNUSABLE", "ERROR_CODES",
           "ERROR_MAP", "UNCERTAIN", "classify",
           "BadRequest", "Limits", "Service", "check_zone_bounds", "create_app", "default_model_path",
           "default_zones_path",
           "model_info"]
