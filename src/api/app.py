"""The Version 0.9 backend API: the frozen serving path of src/predict.py exposed over HTTP.

Five endpoints, fixed in docs/v0.9_api_plan.md and protocol amendment 5: GET /health (liveness),
GET /ready (readiness), POST /predict, POST /batch-predict, GET /model-info.

This module is transport only — application construction, lifespan wiring, routes, exception handlers and
the body-size guard. It holds no inference logic: loading, readiness and the prediction path live in
`inference_service`, the `/model-info` allow-list in `metadata`, and the error vocabulary in `errors`.

The service may not fit, calibrate, tune, score a dataset part, or append to any log, and it takes no
dataset name, split name or filesystem path from a client, so a protected test part cannot be reached
through it.

Three safety rules drive the code, each with a reason that was verified rather than assumed:

- **An uploaded filename is never used or logged.** Raw DRIAMS files are named after the spectrum UUID and
  repeat it in their '#' comment lines, so the filename is read for its suffix and then discarded and the
  upload is written to a generated name — even a library error quoting `path.name` cannot leak it.
- **Uploaded bytes are data, never code.** Nothing from a request is deserialised, imported or executed,
  and nothing reaches joblib.load. A '.joblib' upload is refused at the suffix gate without being opened.
- **No response may carry a non-finite float.** starlette renders JSON with allow_nan=False, so a NaN taken
  from a saved report (uncertainty.json holds one) would raise at render time and become a 500.

Input limits are passed to the serving path as keyword arguments. They are deliberately not
PreprocessingConfig fields: that dataclass's hash is the feature fingerprint every saved bundle is checked
against, so a field there would invalidate every bundle and every cache.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from src.api.errors import ERROR_MAP, UploadTooLarge, error_response
from src.api.inference_service import (
    Limits,
    Service,
    load_artifacts,
    predict_many,
    predict_one,
)
from src.api.metadata import model_info
from src.api.page import render_page
from src.api.schemas import (
    CODE_INTERNAL_ERROR,
    BatchResponse,
    ErrorBody,
    LivenessResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadinessResponse,
)
from src.utils import get_logger

log = get_logger("api")

# The serving contract's own version, deliberately a constant here rather than config.yaml ->
# project.version. That field tracks the project/experiment state (it was stale at "0.7.0" through the
# whole of Version 0.8), so sourcing a public API version from it would let an unrelated edit change the
# contract's identity. The model carries its own separate version and the three are never conflated.
#
# It moves when the contract moves: 0.9.1 split /health into liveness and /ready and added `code` to every
# error body, 0.9.2 added `digest_verified` to /ready, and 1.0.0 adds the result page at GET /. The rule the
# independence test protects is that this value is never *sourced* from configuration - not that it must
# differ numerically from anything else.
API_VERSION = "1.0.0"


def create_app(config: dict[str, Any], *, model_path: Path | None = None,
               zones_path: Path | None = None) -> FastAPI:
    """Build the application. A factory, so tests can serve a synthetic bundle with no global state."""
    limits = Limits.from_config(config)
    service = Service(limits=limits, api_version=API_VERSION)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        load_artifacts(service, config, model_path=model_path, zones_path=zones_path)
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
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        """The result page (Version 1.0). A client of this API, not a second serving path.

        It holds no scientific logic: the browser reads every value from a /predict response. The page is
        served whatever the readiness state, because a reader who arrives at an unready service should be
        told so by the page rather than meeting a bare 503.
        """
        return HTMLResponse(content=render_page())

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
            digest_verified=service.digest_verified,
            detail=None if is_ready else service.detail,
        )
        return JSONResponse(status_code=200 if is_ready else 503, content=body.model_dump())

    @app.get("/model-info", response_model=ModelInfoResponse)
    async def info() -> JSONResponse:
        return JSONResponse(content=model_info(service).model_dump())

    @app.post("/predict", responses={200: {"model": PredictionResponse}})
    async def predict(request: Request, file: Annotated[UploadFile, File()],
                      explain: bool = False) -> JSONResponse:
        submitted = len((await request.form()).getlist("file"))
        payload = await predict_one(service, file, submitted=submitted, explain=explain)
        return JSONResponse(content=payload)

    @app.post("/batch-predict", response_model=BatchResponse)
    async def batch_predict(files: Annotated[list[UploadFile], File()],
                            explain: bool = False) -> JSONResponse:
        body = await predict_many(service, files, explain=explain)
        return JSONResponse(content=body.model_dump(exclude_none=True))

    return app
