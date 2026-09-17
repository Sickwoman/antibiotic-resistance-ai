"""Research prediction for one raw MALDI-TOF spectrum file with a saved model (not a clinical result).

Run from the project root with the virtual environment active:

    python scripts/predict_spectrum.py C:\\DRIAMS\\DRIAMS-A\\raw\\2018\\<code>.txt
    python scripts/predict_spectrum.py <spectrum.txt> --model models/v0.3/ecoli_ciprofloxacin/best_random.joblib

The spectrum must be a DRIAMS-style raw text file ('#' comment lines, a header line and "m/z intensity"
rows). Only use model files produced by this project: model files can run code when they are loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import DataError  # noqa: E402
from src.predict import ModelError, load_bundle, predict_spectrum_file  # noqa: E402
from src.utils import ConfigError, load_config, project_path  # noqa: E402


def default_model(config: dict[str, Any]) -> Path:
    """The model saved by scripts/train_baselines.py for the primary dataset."""
    bl = config["baselines"]
    return project_path(bl["model_dir"]) / config["dataset"]["name"] / f"best_{bl['splits'][0]}.joblib"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research prediction for one raw spectrum file "
                                                 "(not a clinically validated result).")
    parser.add_argument("spectrum", type=Path, help="raw spectrum .txt file")
    parser.add_argument("--model", type=Path, default=None, help="model file (default: the saved Version 0.3 model)")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        model_path = args.model or default_model(load_config(args.config))
        result = predict_spectrum_file(load_bundle(model_path), args.spectrum)
    except (ModelError, DataError, ConfigError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
