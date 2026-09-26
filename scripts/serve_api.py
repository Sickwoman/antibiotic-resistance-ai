"""Run the Version 0.9 backend API locally.

Run from the project root with the virtual environment active:

    python scripts/serve_api.py
    python scripts/serve_api.py --port 8080
    python scripts/serve_api.py --model models/v0.4/ecoli_ciprofloxacin/best_random.joblib

Then open http://127.0.0.1:8000/docs for the interactive documentation, or:

    curl -F "file=@<spectrum.txt>" http://127.0.0.1:8000/predict
    curl http://127.0.0.1:8000/model-info

Defaults come from the `api` section of config.yaml. The service binds to 127.0.0.1 because it has **no
authentication**: it is a research prototype and must not be exposed to a network. It serves predictions
from a saved bundle and never fits, scores or logs anything (protocol amendment 5).

Only use model files produced by this project: a model file can run code when it is loaded. Uploaded
spectra are never deserialised or executed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import create_app, default_model_path, default_zones_path  # noqa: E402
from src.data_loader import DataError  # noqa: E402
from src.predict import ModelError  # noqa: E402
from src.utils import ConfigError, get_logger, load_config  # noqa: E402

log = get_logger("api")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve research predictions from the saved model over HTTP (not a clinical service).")
    parser.add_argument("--host", default=None, help="interface to bind (default: api.host in config.yaml)")
    parser.add_argument("--port", type=int, default=None, help="port (default: api.port in config.yaml)")
    parser.add_argument("--model", type=Path, default=None,
                        help="bundle to serve (default: the project's model, from api.model_dir)")
    parser.add_argument("--uncertainty", type=Path, default=None,
                        help="confidence zones JSON (default: the Version 0.6 report for this dataset)")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args(argv)


def resolve(config: dict[str, Any], args: argparse.Namespace) -> tuple[str, int, Path, Path | None]:
    api = config.get("api") or {}
    host = args.host or str(api.get("host", "127.0.0.1"))
    port = int(args.port if args.port is not None else api.get("port", 8000))
    model = args.model or default_model_path(config)
    zones = args.uncertainty if args.uncertainty is not None else default_zones_path(config)
    if args.uncertainty is not None and not args.uncertainty.is_file():
        raise ModelError(f"Confidence zones file not found: {args.uncertainty}.")
    if not model.is_file():
        raise ModelError(f"Model file not found: {model}. Train and save a model first, or pass --model.")
    return host, port, model, zones


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import uvicorn

        config = load_config(args.config)
        host, port, model, zones = resolve(config, args)
        app = create_app(config, model_path=model, zones_path=zones)
        log.info("serving %s on http://%s:%d (no authentication: localhost only)", model.name, host, port)
        log.info("interactive documentation: http://%s:%d/docs", host, port)
        uvicorn.run(app, host=host, port=port, log_level="info")
    except (ModelError, DataError, ConfigError) as exc:
        log.error("%s", exc)
        return 1
    except ImportError as exc:
        log.error("the API needs fastapi and uvicorn: pip install -r requirements.txt (%s)", exc)
        return 1
    except KeyboardInterrupt:
        log.info("stopped")
        return 130
    except Exception as exc:                      # a run must never end silently
        log.exception("the server stopped with an unexpected error: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
