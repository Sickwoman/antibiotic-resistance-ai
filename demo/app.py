"""The demo's local web server: one page, three synthetic examples, and real inference through the unchanged CLI.

It is a small FastAPI application, separate from the served API in `src/api/`, which it neither imports nor
changes. It needs this Python process running: the page is not a static file.

Routes:
- `GET /` and `GET /static/...`: the page, with no third-party resource (a strict Content-Security-Policy makes the
  browser refuse any request to another origin);
- `GET /api/status`, `GET /api/examples`, `GET /api/evidence`: model presence, the synthetic spectra, and the
  committed aggregate results;
- `POST /api/predict`: `{"example": "<id>"}` for one of the three synthetic spectra. There is no upload route, so
  nothing but those three spectra can reach the model.

Nothing is stored. Each response is marked `Cache-Control: no-store`, and the server keeps no history. Requests whose
Host header is not a loopback name are refused, which guards a server bound to 127.0.0.1 against DNS rebinding.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.middleware.trustedhost import TrustedHostMiddleware

from demo import synthetic
from demo.evidence import EvidenceError, load_evidence
from demo.inference import DEFAULT_TIMEOUT, DemoInferenceError, default_model_path, model_status, predict

STATIC = Path(__file__).resolve().parent / "static"
ASSETS = {"app.js": "text/javascript", "style.css": "text/css"}
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
SECURITY_HEADERS = {
    "Content-Security-Policy": ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
                                "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    example: str


def error(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def create_app(*, model: Path | None = None, timeout: float = DEFAULT_TIMEOUT,
               runner: Callable[..., dict[str, Any]] = predict,
               allowed_hosts: tuple[str, ...] = LOOPBACK_HOSTS) -> FastAPI:
    """The demo application. `model` is passed to the CLI as --model when given; by default the CLI picks its own."""
    examples = {e.id: e for e in synthetic.generate()}
    shown_model = model if model is not None else default_model_path()      # resolved once, by the CLI's own code
    try:
        evidence, evidence_error = load_evidence(), None
    except (OSError, KeyError, ValueError, EvidenceError) as exc:
        evidence, evidence_error = None, f"The committed aggregate results could not be read ({type(exc).__name__})."

    app = FastAPI(title="Research demo: the frozen ciprofloxacin model", docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))

    @app.middleware("http")
    async def headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    @app.get("/", include_in_schema=False)
    def page() -> FileResponse:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.get("/static/{name}", include_in_schema=False)
    def asset(name: str):
        if name not in ASSETS:
            return error("not_found", "No such file.", 404)
        return FileResponse(STATIC / name, media_type=ASSETS[name])

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {"model": model_status(shown_model), "examples": len(examples),
                "evidence_available": evidence is not None,
                "inputs": "synthetic spectra only (seed 42); no upload route exists",
                "runner": "scripts/predict_spectrum.py, unchanged, one process per prediction"}

    @app.get("/api/examples")
    def list_examples() -> list[dict[str, Any]]:
        return [{"id": e.id, "title": e.title, "n_points": e.n_points, "mz_min": round(float(e.mz[0]), 1),
                 "mz_max": round(float(e.mz[-1]), 1), "seed": synthetic.SEED, "synthetic": True,
                 "plot": synthetic.plot_points(e)} for e in examples.values()]

    @app.get("/api/evidence")
    def get_evidence():
        if evidence is None:
            return error("evidence_unavailable", evidence_error, 503)
        return evidence

    @app.post("/api/predict")
    def run_prediction(body: PredictRequest):
        chosen = examples.get(body.example)
        if chosen is None:
            return error("unknown_example", "Only the three synthetic examples can be scored.", 404)
        try:
            result = runner(chosen.to_text(), model=model, timeout=timeout)
        except DemoInferenceError as exc:
            return error(exc.code, exc.message, exc.status)
        return {"example": chosen.id, "synthetic": True, **result}

    return app
