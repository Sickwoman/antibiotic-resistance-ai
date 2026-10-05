"""Install the frozen model from a local package, verified before anything is written.

    python scripts/install_model.py <path to ecoli_ciprofloxacin-v0.4.0-tuned_lightgbm-random-seed42.zip>
    python scripts/install_model.py <extracted package folder>

What it checks:
- **The pin.** The bundle's SHA-256 must equal the value pinned in this repository (`src/model_package.py`, also in
  `docs/reproduction_guide.md`) before anything is written. The package's own checksum file and manifest can only
  refuse a package, never accept one.
- **No deserialisation.** It never loads the bundle: a joblib file is a pickle, and loading one runs code. The
  project's unchanged loader does that later, at prediction time.
- **Local files only.** It never downloads anything.

Where it writes: `models/v0.4/ecoli_ciprofloxacin/best_random.joblib` and its `.sha256` file, the path the unchanged
CLI and the demo read. It never replaces a different file already there.

Exit codes:
- 0: installed, or already installed;
- 2: verification failed (missing, damaged, inconsistent or different package; nothing written);
- 3: refused to overwrite a different existing model (left untouched).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_package import PackageError, install  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install the frozen model from a local package (verified first).")
    parser.add_argument("package", help="the package .zip, or a folder with its extracted members")
    parser.add_argument("--target", type=Path, default=None,
                        help="install somewhere else than models/v0.4/ecoli_ciprofloxacin/best_random.joblib")
    args = parser.parse_args(argv)
    try:
        result = install(args.package, target=args.target)
    except PackageError as exc:
        print(f"Not installed: {exc}", file=sys.stderr)
        return exc.code
    print(result.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
