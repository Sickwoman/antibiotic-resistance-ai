"""Research prediction for one raw MALDI-TOF spectrum file with a saved model (not a clinical result).

Run from the project root with the virtual environment active:

    python scripts/predict_spectrum.py C:\\DRIAMS\\DRIAMS-A\\raw\\2018\\<code>.txt
    python scripts/predict_spectrum.py <spectrum.txt> --explain
    python scripts/predict_spectrum.py <spectrum.txt> --model models/v0.3/ecoli_ciprofloxacin/best_random.joblib

The default model is the project's model (Version 0.6 reports on the tuned Version 0.4 bundle named by
`explain.model_dir` / `explain.split` in config.yaml). Two optional parts come from Version 0.6 and change
nothing about the probability or the cut-off:

- the three-way confidence label, if the confidence zones fitted on validation are available: the zones
  file is read from `explain.report_dir` unless --uncertainty names another one;
- --explain, the m/z regions that moved this one prediction (only a tree model can give these exactly).

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
from src.predict import ModelError, load_bundle, load_zones, predict_spectrum_file  # noqa: E402
from src.uncertainty import ADVICE, UNCERTAIN  # noqa: E402
from src.utils import ConfigError, load_config, project_path  # noqa: E402

ZONES_FILENAME = "uncertainty.json"


def default_model(config: dict[str, Any]) -> Path:
    """The project's model: the bundle Version 0.6 describes (`explain.model_dir` / `explain.split`).

    Configurations without an `explain` section fall back to the Version 0.3 baseline bundle, so the
    script keeps working against an older checkout of config.yaml.
    """
    explain = config.get("explain")
    if explain:
        return project_path(explain["model_dir"]) / config["dataset"]["name"] / f"best_{explain['split']}.joblib"
    bl = config["baselines"]
    return project_path(bl["model_dir"]) / config["dataset"]["name"] / f"best_{bl['splits'][0]}.joblib"


def default_zones_path(config: dict[str, Any]) -> Path | None:
    """Where Version 0.6 writes the fitted confidence zones, or None if there is no `explain` section.

    The file need not exist; a missing one simply means no confidence label is reported.
    """
    explain = config.get("explain")
    if not explain:
        return None
    return project_path(explain["report_dir"]) / config["dataset"]["name"] / ZONES_FILENAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Research prediction for one raw spectrum file "
                                                 "(not a clinically validated result).")
    parser.add_argument("spectrum", type=Path, help="raw spectrum .txt file")
    parser.add_argument("--model", type=Path, default=None,
                        help="model file (default: the project's Version 0.6 model from config.yaml)")
    parser.add_argument("--uncertainty", type=Path, default=None,
                        help="confidence zones JSON fitted on validation (default: the Version 0.6 report "
                             "for this dataset, used when that file exists)")
    parser.add_argument("--explain", action="store_true",
                        help="also report the m/z regions that moved this prediction (tree models only)")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.uncertainty is not None and not args.uncertainty.is_file():
            raise ModelError(f"Confidence zones file not found: {args.uncertainty}.")
        zones_path = args.uncertainty or default_zones_path(config)
        zones = None if zones_path is None else load_zones(zones_path)
        result = predict_spectrum_file(load_bundle(args.model or default_model(config)), args.spectrum,
                                       zones=zones, explain=args.explain)
    except (ModelError, DataError, ConfigError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    out = result.to_dict()
    if result.confidence == UNCERTAIN:
        out["advice"] = ADVICE                 # a zone that declines to call the isolate says what to do next
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
