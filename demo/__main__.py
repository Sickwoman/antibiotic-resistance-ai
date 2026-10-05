"""Launch the research demo on this computer only.

    python -m demo                    # http://127.0.0.1:8050/
    python -m demo --port 8051 --open

It binds to the loopback interface and refuses any other host. It scores only its three synthetic spectra, through
the project's unchanged prediction CLI, so the frozen model bundle must be present in `models/` (it is not in Git).
Stop it with Ctrl+C.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

from demo.app import create_app
from demo.inference import DEFAULT_TIMEOUT, default_model_path, model_status, sweep_stale

LOOPBACK = ("127.0.0.1", "localhost")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m demo", description="Run the local research demo "
                                     "(synthetic spectra only; not a clinical tool).")
    parser.add_argument("--host", default="127.0.0.1", help="loopback only: 127.0.0.1 (default) or localhost")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--model", type=Path, default=None,
                        help="bundle passed to the CLI as --model (default: the CLI's own default, the frozen model)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds allowed per prediction")
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = parser.parse_args(argv)
    if args.host not in LOOPBACK:
        print(f"Refusing to bind to {args.host!r}: the demo serves this computer only (127.0.0.1 or localhost).",
              file=sys.stderr)
        return 2
    url = f"http://{args.host}:{args.port}/"
    model = model_status(args.model if args.model is not None else default_model_path())
    print("Research demo of the frozen ciprofloxacin model: synthetic spectra only, not a clinical tool.")
    print(f"  model bundle: {model['path']} ({'present' if model['present'] else 'MISSING'})")
    if not model["present"]:
        print("  The frozen bundle is not distributed with the repository. The page and the evaluation panel\n"
              "  work, but every prediction will report the missing model. See demo/README.md, 'The frozen model'.")
    removed = sweep_stale()
    if removed:
        print(f"  removed {removed} temporary folder(s) that an interrupted earlier run left behind")
    print(f"  open {url}  (Ctrl+C to stop)")
    if args.open:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(model=args.model, timeout=args.timeout), host=args.host, port=args.port,
                log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
