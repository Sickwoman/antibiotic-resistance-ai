"""Loading the frozen artifacts, tracking readiness, and delegating a prediction to src.predict.

No scientific logic is reimplemented here: `predict_spectrum_file` does the whole prediction, exactly as it
does for the command line, and `prediction_payload` formats the result so the two cannot diverge.

`predict_spectrum_file` and `TemporaryDirectory` are imported into this module deliberately. They are what
the upload path calls, so this is the module a test must monkeypatch to intercept them — behind a
re-exporting package `__init__` such a patch would bind a name nothing reads and pass while testing nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fastapi import UploadFile

from src.api.errors import BadRequest, TooManyFiles, UnsupportedUpload, UploadTooLarge, classify
from src.api.metadata import read_log_rows, read_timing
from src.api.schemas import (
    BatchItem,
    BatchResponse,
    ErrorBody,
    ExternalResult,
    MetricSet,
    PredictionResponse,
)
from src.data_loader import DataError
from src.predict import (
    DISCLAIMER,
    ModelError,
    load_bundle,
    load_zones,
    predict_spectrum_file,
    prediction_payload,
    verify_digest,
)
from src.uncertainty import Zones
from src.utils import ConfigError, get_logger, project_path

log = get_logger("api")

# What a degraded /health may say. Load failures carry a filesystem path in their message, so the public
# text is one of these fixed strings and the real exception goes to the log only.
DETAIL_MODEL_UNAVAILABLE = "The configured model could not be loaded, so predictions are unavailable."
DETAIL_ZONES_UNUSABLE = ("The confidence-zone file exists but is not a usable zones record, so the "
                         "service will not serve predictions whose confidence label would silently "
                         "read 'not available'.")

CHUNK_BYTES = 64 * 1024                 # how much of an upload is read at a time
UPLOAD_STEM = "upload"                  # generated name for a saved upload: the client's is discarded

CHUNK_BYTES = 64 * 1024                 # how much of an upload is read at a time
UPLOAD_STEM = "upload"                  # generated name for a saved upload: the client's is discarded


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
    digest_verified: bool = False        # the bundle matched its .sha256 sidecar (False = no sidecar)
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
# Where the configured artifacts live
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


# ------------------------------------------------------------------------------------------------
# Start-up and the prediction path
# ------------------------------------------------------------------------------------------------

def load_artifacts(service: Service, config: dict[str, Any], *, model_path: Path | None = None,
                   zones_path: Path | None = None) -> None:
    """Load the bundle and the zones once, recording readiness. Never raises for a missing artifact.

    A failure leaves the process running in a deterministic not-ready state rather than crash-looping, and
    the public `detail` is one of the fixed strings because a load error's own message names the file, i.e.
    an absolute path.
    """
    api_cfg = config.get("api") or {}
    path = model_path or default_model_path(config)
    try:
        service.bundle = load_bundle(path, n_jobs=int(api_cfg.get("request_threads", 1)))
        # load_bundle already refused a mismatch; this records whether the check could run at all,
        # so a missing sidecar is visible on /ready instead of looking like a passed check.
        service.digest_verified = verify_digest(path) is not None
        log.info("serving model %s (digest %s)", service.bundle["model_version"],
                 "verified" if service.digest_verified else "unverified: no sidecar")
    except (ModelError, DataError, ConfigError) as exc:
        service.detail = DETAIL_MODEL_UNAVAILABLE
        log.error("no model loaded, the service will report degraded: %s", exc)
    zp = zones_path if zones_path is not None else default_zones_path(config)
    if service.bundle is not None and zp is not None:
        try:
            # src.uncertainty.Zones validates its own edges (issue #16), and load_zones wraps the
            # UncertaintyError as ModelError, so an unusable record lands in the except below.
            service.zones = load_zones(zp)
        except (ModelError, DataError) as exc:
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


async def predict_one(service: Service, file: UploadFile, *, submitted: int,
                      explain: bool = False) -> dict[str, Any]:
    """One spectrum in, the response payload out. `submitted` is how many files the request carried."""
    bundle = service.require_model()
    # Exactly one spectrum. FastAPI binds a single UploadFile even when several were sent, so without this
    # the caller would get a 200 for a request only one part of which was used.
    if submitted != 1:
        raise BadRequest(f"Send exactly one spectrum file in the 'file' field; {submitted} were sent. "
                         "Use /batch-predict for several.")
    check_suffix(file.filename, service.limits.allowed_suffixes)
    with TemporaryDirectory() as tmp:
        # A generated name: the client's filename is a DRIAMS UUID and must not reach a message or a log.
        dest = Path(tmp) / f"{UPLOAD_STEM}.txt"
        await stream_to_file(file, dest, service.limits.max_upload_bytes)
        result = predict_spectrum_file(bundle, dest, zones=service.zones, explain=explain,
                                       max_points=service.limits.max_raw_points,
                                       max_bytes=service.limits.max_upload_bytes)
    payload = prediction_payload(result)
    PredictionResponse.model_validate(payload)      # the declared contract, checked before it is sent
    return payload


async def predict_many(service: Service, files: list[UploadFile], *, explain: bool = False) -> BatchResponse:
    """Several spectra in one request. One unusable file is reported in its own slot, not raised."""
    bundle = service.require_model()
    limits = service.limits
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
                items.append(BatchItem(index=index, error=ErrorBody(
                    code=classify(exc)[1], message=str(exc), type=type(exc).__name__)))
            finally:
                dest.unlink(missing_ok=True)
    failed = sum(1 for i in items if i.error is not None)
    return BatchResponse(n_submitted=len(files), n_succeeded=len(items) - failed, n_failed=failed,
                         results=items, disclaimer=DISCLAIMER)
