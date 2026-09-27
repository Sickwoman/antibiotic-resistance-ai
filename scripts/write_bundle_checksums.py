"""Write the `.sha256` sidecar beside each saved model bundle, or verify the ones already there.

Run from the project root with the virtual environment active:

    python scripts/write_bundle_checksums.py --check     # verify only, changes nothing (default)
    python scripts/write_bundle_checksums.py --write      # create any missing sidecar

Why this exists: `src.predict.save_bundle` has written a sidecar beside every bundle it saved since
Version 0.3, and `load_bundle` calls `verify_digest` to refuse a file that changed after it was saved. The
bundles currently on disk predate that code, so no sidecar existed for them and `verify_digest` returned
None — the substitution check was inert for the one artifact the API actually serves (issue #18).

This script only ever **adds** a sidecar computed from a bundle's current bytes. It never writes, rewrites
or loads a bundle: the digest is taken by streaming the file, so nothing here can execute pickled code. A
sidecar that already exists is verified, never overwritten, because silently replacing one would defeat its
purpose — if a bundle really did change, the right outcome is a loud mismatch, not a fresh sidecar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import ModelError, checksum_path, file_digest, verify_digest  # noqa: E402
from src.utils import get_logger, project_path  # noqa: E402

log = get_logger("api")

BUNDLE_GLOB = "best_*.joblib"          # the saved bundles; the tuning cache holds search artifacts, not these


def bundles(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob(BUNDLE_GLOB) if p.is_file() and "cache" not in p.parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write or verify the .sha256 sidecar beside each bundle.")
    parser.add_argument("--write", action="store_true",
                        help="create any missing sidecar (default is to verify and change nothing)")
    parser.add_argument("--models-dir", type=Path, default=None, help="default: models/ in the project root")
    args = parser.parse_args(argv)

    try:
        root = args.models_dir or project_path("models")
        if not root.is_dir():
            raise ModelError(f"No models directory at {root}; nothing to do.")
        found = bundles(root)
        if not found:
            log.info("no bundles under %s", root.name)
            return 0

        written, verified, missing = 0, 0, 0
        for bundle in found:
            relative = bundle.relative_to(root.parent).as_posix()
            sidecar = checksum_path(bundle)
            if sidecar.is_file():
                digest = verify_digest(bundle)          # raises ModelError on a mismatch
                verified += 1
                log.info("verified %s  %s", (digest or "")[:16], relative)
            elif args.write:
                digest = file_digest(bundle)
                sidecar.write_text(f"{digest}  {bundle.name}\n", encoding="utf-8")
                # Read it back through the real verifier, so a sidecar is never reported as written
                # unless load_bundle would actually accept it.
                if verify_digest(bundle) != digest:
                    raise ModelError(f"the sidecar written for {relative} does not verify; refusing to continue.")
                written += 1
                log.info("wrote    %s  %s", digest[:16], relative)
            else:
                missing += 1
                log.info("MISSING sidecar for %s (run with --write)", relative)

        log.info("%d bundle(s): %d verified, %d written, %d missing", len(found), verified, written, missing)
        return 1 if missing else 0
    except (ModelError, OSError) as exc:
        log.error("%s", exc)
        return 1
    except Exception as exc:                      # a run must never end silently
        log.exception("the run stopped with an unexpected error: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
