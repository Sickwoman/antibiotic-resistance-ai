"""Real local inference for the demo, through the project's unchanged prediction CLI (`scripts/predict_spectrum.py`).

**Why the CLI.** It is the project's own entry point for one spectrum, and it builds its output with
`src.predict.prediction_payload`, exactly as the served API does, so the two cannot disagree. It loads the frozen
bundle, verifying its SHA-256 sidecar. It reads and preprocesses the spectrum exactly as in training, and prints one
JSON payload. It writes no file and appends to no log: its only outputs are that JSON on stdout, or one error line
on stderr.

**What this module keeps: nothing.**
- The selected synthetic spectrum is written to a fresh temporary directory for the one call, because the CLI reads
  a path. The directory is deleted as soon as the CLI returns.
- No result is stored.

**What it passes on: an allow-list.**
- The CLI's words for the two sides of the cut-off and its confidence-zone label are never forwarded.
- The demo reports "above" or "below the research threshold" instead (docs/research_demo_brief.md, section 5). That
  status is taken from the CLI's own decision and checked against score >= threshold.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "predict_spectrum.py"
DEFAULT_TIMEOUT = 180.0
FORWARDED = ("model_version", "preprocessing_ms", "inference_ms", "total_ms", "disclaimer")
ABOVE, BELOW = "Resistant", "Susceptible"         # the CLI's decision values; read here, never forwarded
ROUNDING = 1e-4                                   # the payload's four-decimal rounding of score and threshold
TEMP_PREFIX = "amr-demo-"
STALE_AFTER_S = 15 * 60                           # a leftover older than this cannot belong to a live run
_lock = threading.Lock()                          # one prediction at a time: each one starts a fresh process


class DemoInferenceError(RuntimeError):
    """A prediction that could not be made: a code for the page, a message safe to show, an HTTP status."""

    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def _cli_module():
    spec = importlib.util.spec_from_file_location("demo_predict_spectrum_cli", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_model_path() -> Path:
    """The bundle the CLI loads when it is given no --model: computed by the CLI's own code, not repeated here."""
    cli = _cli_module()
    return Path(cli.default_model(cli.load_config(None)))


def model_status(path: Path) -> dict[str, Any]:
    """Whether the bundle the CLI will load is present, and whether it has its checksum sidecar (nothing is loaded)."""
    return {"path": shown(path), "present": path.is_file(),
            "sidecar": path.with_name(path.name + ".sha256").is_file()}


def shown(path: Path | str) -> str:
    """A path as the page may show it: relative to the repository where possible, never the full local path."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _safe(message: str, tmp: str | None) -> str:
    text = message.strip()
    for prefix, label in ((str(ROOT), "<repository>"), (tmp, "<temporary folder>"),
                          (tempfile.gettempdir(), "<temporary folder>")):
        if prefix:
            text = text.replace(prefix, label).replace(prefix.replace("\\", "/"), label)
    return text.replace("\\", "/")[:600]


def _classify(stderr: str, tmp: str | None) -> DemoInferenceError:
    message = _safe(stderr.removeprefix("Error:").strip() or "the prediction CLI failed without a message", tmp)
    if "Model file not found" in message:
        # The CLI adds "Train and save a model first with ...": retraining would not give the frozen model, and that
        # command scores spent test parts, so only its first sentence is relayed.
        message = message.split(". Train and save", 1)[0].rstrip(".") + "."
        return DemoInferenceError("model_unavailable", "The frozen model bundle was not found, so no prediction can "
                                  "be made. It is not distributed with the repository (models/ is not in Git). An "
                                  "authorised user obtains models/v0.4/ecoli_ciprofloxacin/best_random.joblib from "
                                  "the project owner through a trusted channel and checks its SHA-256 before use: "
                                  "see demo/README.md, \"The frozen model\". The rest of the page works without it. "
                                  f"The CLI reported: {message}", 503)
    if any(s in message for s in ("does not match its checksum", "Could not read the model file",
                                  "is not a model bundle", "confidence zones", "confidence-zones")):
        return DemoInferenceError("model_invalid", "The frozen model artifacts could not be used, so no prediction "
                                  f"was made. The CLI reported: {message}", 503)
    return DemoInferenceError("inference_failed", f"The prediction CLI failed. It reported: {message}", 500)


def sweep_stale(max_age_s: float = STALE_AFTER_S) -> int:
    """Remove this demo's temporary folders that a killed run left behind (only those older than `max_age_s`).

    A run deletes its folder when the CLI returns. A server killed mid-prediction cannot, so the next start removes
    what is left. The age limit, far above the longest allowed run, keeps any other instance's live folder safe.
    """
    removed, now = 0, time.time()
    for path in Path(tempfile.gettempdir()).glob(f"{TEMP_PREFIX}*"):
        try:
            if path.is_dir() and now - path.stat().st_mtime > max_age_s:
                shutil.rmtree(path)
                removed += 1
        except OSError:
            continue
    return removed


def child_environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"          # the child leaves no bytecode cache behind either
    return env


def predict(spectrum_text: str, *, model: Path | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Score one synthetic spectrum with the frozen model through the unchanged CLI. Returns the allow-listed result."""
    command = [sys.executable, str(CLI)]
    with _lock, TemporaryDirectory(prefix=TEMP_PREFIX) as tmp:
        path = Path(tmp) / "synthetic_spectrum.txt"
        path.write_text(spectrum_text, encoding="utf-8")
        started = time.perf_counter()
        try:
            done = subprocess.run([*command, str(path), *(["--model", str(model)] if model else [])],
                                  cwd=ROOT, env=child_environment(), capture_output=True, text=True,
                                  encoding="utf-8", timeout=timeout)
        except subprocess.TimeoutExpired:
            raise DemoInferenceError("timeout", f"The prediction CLI did not finish within {timeout:.0f} s.",
                                     504) from None
        except OSError as exc:
            raise DemoInferenceError("inference_failed", f"The prediction CLI could not be started "
                                     f"({type(exc).__name__}).", 500) from None
        elapsed = time.perf_counter() - started
        if done.returncode != 0:
            raise _classify(done.stderr, tmp)
        stdout = done.stdout
    try:
        payload = json.loads(stdout)
        score, threshold, decision = (float(payload["resistance_probability"]), float(payload["threshold"]),
                                      payload["prediction"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise DemoInferenceError("inference_failed", "The prediction CLI returned output the demo cannot read.",
                                 500) from None
    in_range = math.isfinite(score) and 0.0 <= score <= 1.0 and math.isfinite(threshold)
    if not in_range or decision not in (ABOVE, BELOW):
        raise DemoInferenceError("inference_failed", "The prediction CLI returned values outside their range.", 500)
    above = decision == ABOVE
    # The payload rounds the score and the threshold to four decimals; the decision is made on the unrounded score.
    # So the decision is checked against the rounded numbers only where rounding cannot explain a difference.
    if abs(score - threshold) > ROUNDING and above != (score >= threshold):
        raise DemoInferenceError("inference_failed", "The CLI's decision disagrees with its score and threshold.", 500)
    return {"score": score, "threshold": threshold, "above_threshold": above,
            **{key: payload.get(key) for key in FORWARDED}, "cli_wall_s": round(elapsed, 2),
            "runner": "scripts/predict_spectrum.py (unchanged)"}
