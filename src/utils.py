"""Shared helpers: configuration loading, path resolution, seeding and logging."""

from __future__ import annotations

import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing or incomplete."""


def read_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader for a local .env file (comments and blank lines ignored)."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml.

    paths.driams_root can be overridden by the DRIAMS_ROOT environment variable, or else by a
    DRIAMS_ROOT line in a .env file next to config.yaml.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict) or "paths" not in config:
        raise ConfigError(f"Configuration file {config_path} has no 'paths' section.")
    env_root = os.environ.get("DRIAMS_ROOT") or read_dotenv(config_path.parent / ".env").get("DRIAMS_ROOT")
    if env_root:
        config["paths"]["driams_root"] = env_root
    return config


def driams_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["driams_root"])


def archives_dir(config: dict[str, Any]) -> Path:
    return driams_root(config) / config["paths"]["archives_subdir"]


def manifests_dir(config: dict[str, Any]) -> Path:
    return driams_root(config) / config["paths"]["manifests_subdir"]


def project_path(relative: str | Path) -> Path:
    """Resolve a path from config.yaml relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def archive_info(config: dict[str, Any], site_letter: str) -> dict[str, Any]:
    """Return the archive entry for a site letter (A, B, C or D)."""
    letter = site_letter.upper().replace("DRIAMS-", "").replace("DRIAMS_", "")
    archives = config["driams"]["archives"]
    if letter not in archives:
        raise ConfigError(f"Unknown DRIAMS site '{site_letter}'. Choose from {sorted(archives)}.")
    return archives[letter]


def free_disk_gb(path: Path) -> float:
    """Free space (GB) on the drive holding `path` (walks up to an existing parent)."""
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def set_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs (deep-learning libraries are seeded in their own modules)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def human_bytes(n: float) -> str:
    """Decimal units (1 GB = 10^9 bytes), matching the sizes published by Zenodo and Dryad."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1000
    return f"{n:.2f} TB"
