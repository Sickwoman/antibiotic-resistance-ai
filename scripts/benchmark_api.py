"""Measure the Version 0.9 API's request latency and write results/metrics/v0.9/<dataset>/api_timing.json.

Run from the project root with the virtual environment active:

    python scripts/benchmark_api.py
    python scripts/benchmark_api.py --samples 50

This records **durations only**. No probability, no label, no identifier and no filename is stored, so it
cannot become a covert scoring run: protocol amendment 5, point 3, which is also why it uses
**validation-part** spectra and never a test part. It appends nothing to the append-only evaluation log,
and asserts the log is byte-unchanged when it finishes.

A real uvicorn server is started on a free local port and driven over HTTP, so the reported numbers include
the transport and serialisation cost rather than only the in-process call. `timing_samples` in config.yaml
sets how many spectra are timed, matching what Versions 0.3 and 0.4 measured for the model alone.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import create_app, default_model_path, default_zones_path  # noqa: E402
from src.data_loader import DataError  # noqa: E402
from src.dataset import load_dataset, resolve_relpath  # noqa: E402
from src.predict import ModelError  # noqa: E402
from src.splits import SplitError, load_splits  # noqa: E402
from src.utils import ConfigError, driams_root, get_logger, load_config, project_path  # noqa: E402

log = get_logger("api")

LOG_RELPATH = "results/experiments/test_evaluations.csv"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class BackgroundServer:
    """A real uvicorn server on a background thread, so latency includes the HTTP round trip."""

    def __init__(self, app: Any, port: int) -> None:
        import uvicorn

        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> BackgroundServer:
        self.thread.start()
        deadline = time.time() + 60
        while time.time() < deadline:
            if getattr(self.server, "started", False):
                return self
            time.sleep(0.05)
        raise ModelError("The API did not start within 60 seconds, so nothing was measured.")

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=60)


def summarise(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"median": round(float(np.median(array)), 2), "p95": round(float(np.quantile(array, 0.95)), 2),
            "max": round(float(array.max()), 2)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the API's request latency (durations only).")
    parser.add_argument("--samples", type=int, default=None,
                        help="spectra to time (default: evaluation.timing_samples in config.yaml)")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        import httpx

        config = load_config(args.config)
        dataset_name = config["dataset"]["name"]
        api_cfg = config.get("api") or {}
        split_name = str(api_cfg.get("split") or (config.get("explain") or {})["split"])

        log_path = project_path(LOG_RELPATH)
        log_before = log_path.read_bytes() if log_path.is_file() else None

        data_dir = project_path(config["dataset"]["output_dir"]) / dataset_name
        _, meta, _ = load_dataset(data_dir, verify_x=False)
        splits = load_splits(data_dir / "splits", meta)
        if split_name not in splits:
            raise SplitError(f"split {split_name!r} is not built; run scripts/build_dataset.py first.")

        # Validation rows only. A test part is never read here, timing or not.
        rows = splits[split_name].validation
        wanted = int(args.samples if args.samples is not None else config["evaluation"]["timing_samples"])
        rng = np.random.default_rng(int(config["project"]["random_seed"]))
        timed = rng.choice(rows, size=min(wanted, rows.size), replace=False)
        root = driams_root(config)
        paths = [resolve_relpath(root, rel) for rel in meta["spectrum_relpath"].iloc[timed]]

        model_path = default_model_path(config)
        if not model_path.is_file():
            raise ModelError(f"Model file not found: {model_path}. Nothing was measured.")
        app = create_app(config, model_path=model_path, zones_path=default_zones_path(config))

        port = free_port()
        records: list[dict[str, float]] = []
        startup_started = time.perf_counter()
        with BackgroundServer(app, port) as _server:
            startup_ms = (time.perf_counter() - startup_started) * 1000
            base = f"http://127.0.0.1:{port}"
            with httpx.Client(base_url=base, timeout=120.0) as client:
                # Readiness, not liveness: /health only says the process is up, and the model version
                # lives on /ready. Measuring against a not-ready service would time refusals.
                ready = client.get("/ready")
                if ready.status_code != 200:
                    raise ModelError(f"the API reported {ready.status_code} at /ready; nothing was measured.")
                model_version = str(ready.json()["model_version"])

                cold_started = time.perf_counter()
                client.post("/predict", files={"file": ("warmup.txt", paths[0].read_bytes())})
                cold_ms = (time.perf_counter() - cold_started) * 1000      # first request, nothing warm

                log.info("timing %d single-spectrum requests", len(paths))
                for path in paths:
                    payload = path.read_bytes()
                    started = time.perf_counter()
                    response = client.post("/predict", files={"file": ("spectrum.txt", payload)})
                    round_trip = (time.perf_counter() - started) * 1000
                    response.raise_for_status()
                    body = response.json()
                    # Durations only: the probability and the label are deliberately not read.
                    records.append({"round_trip_ms": round_trip,
                                    "preprocessing_ms": float(body["preprocessing_ms"]),
                                    "inference_ms": float(body["inference_ms"]),
                                    "total_ms": float(body["total_ms"])})

                # The explanation path, timed separately: it runs after the prediction clock stops, so
                # its cost is invisible in the per-response timings.
                explain_ms: list[float] = []
                for path in paths[:min(20, len(paths))]:
                    payload = path.read_bytes()
                    started = time.perf_counter()
                    response = client.post("/predict?explain=true",
                                           files={"file": ("spectrum.txt", payload)})
                    explain_ms.append((time.perf_counter() - started) * 1000)
                    response.raise_for_status()
                    explained = response.json()["explanation"] is not None

                batch_size = min(int((api_cfg.get("limits") or {}).get("max_batch_files", 20)), len(paths))
                files = [("files", ("spectrum.txt", p.read_bytes())) for p in paths[:batch_size]]
                started = time.perf_counter()
                batch = client.post("/batch-predict", files=files)
                batch_ms = (time.perf_counter() - started) * 1000
                batch.raise_for_status()
                batch_ok = int(batch.json()["n_succeeded"])

        frame = pd.DataFrame(records)
        timing: dict[str, Any] = {
            "samples": len(frame), "model": model_version, "rows": "validation",
            "records": "durations only; no probability, label or identifier is stored",
            "transport": "real uvicorn server on 127.0.0.1, driven with httpx",
            "server_startup_ms": round(startup_ms, 2),
            "cold_first_request_ms": round(cold_ms, 2),
            "explanation_returned": explained,
            "batch_size": batch_size, "batch_succeeded": batch_ok,
            "batch_round_trip_ms": round(batch_ms, 2),
            "batch_round_trip_ms_per_sample": round(batch_ms / max(batch_size, 1), 3),
        }
        for column in ("round_trip_ms", "preprocessing_ms", "inference_ms", "total_ms"):
            timing[column] = summarise(frame[column].tolist())
        timing["explain_round_trip_ms"] = summarise(explain_ms)
        timing["explanation_cost_ms_median"] = round(timing["explain_round_trip_ms"]["median"]
                                                    - timing["round_trip_ms"]["median"], 2)
        timing["http_overhead_ms_median"] = round(timing["round_trip_ms"]["median"]
                                                  - timing["total_ms"]["median"], 2)

        report_dir = project_path(api_cfg.get("report_dir", "results/metrics/v0.9")) / dataset_name
        report_dir.mkdir(parents=True, exist_ok=True)
        out = report_dir / "api_timing.json"
        out.write_text(json.dumps(timing, indent=2), encoding="utf-8")
        log.info("wrote %s", out)
        print(json.dumps(timing, indent=2))

        log_after = log_path.read_bytes() if log_path.is_file() else None
        if log_after != log_before:
            raise ModelError("the append-only evaluation log changed during the benchmark; this is a bug.")
        log.info("the append-only log is byte-unchanged, as it must be")
    except (ModelError, DataError, ConfigError, SplitError) as exc:
        log.error("%s", exc)
        return 1
    except ImportError as exc:
        log.error("the benchmark needs fastapi, uvicorn and httpx: pip install -r requirements.txt (%s)", exc)
        return 1
    except KeyboardInterrupt:
        log.info("stopped")
        return 130
    except Exception as exc:                      # a run must never end silently
        log.exception("the benchmark stopped with an unexpected error: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
