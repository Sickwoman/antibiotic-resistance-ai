"""Tests for configuration loading and small utilities."""

from __future__ import annotations

import sys

import pytest

from src.utils import ConfigError, archive_info, human_bytes, keep_awake, load_config

CONFIG = "paths:\n  driams_root: C:/DRIAMS\ndriams:\n  archives:\n    B: {site: DRIAMS-B}\n"


def test_project_config_loads():
    config = load_config()
    assert config["target"]["species"] == "Escherichia coli"
    assert archive_info(config, "B")["size_bytes"] == 3688166065
    assert archive_info(config, "DRIAMS-B")["site"] == "DRIAMS-B"


def test_unknown_site_raises():
    with pytest.raises(ConfigError):
        archive_info(load_config(), "Z")


def test_reserved_sites_are_not_used_by_any_version_that_reserved_them():
    """A reserved site must be genuinely out of reach, not merely described as reserved in prose.

    DRIAMS-C is kept for Version 0.8 as an adaptation target whose labels have never been spent
    (docs/v0.7_generalisation_plan.md, addendum 2026-09-24). If it ever appears in the built sites or in a
    split's test part, Version 0.7 would spend it and Version 0.8 would lose the only site that can answer
    its question.
    """
    config = load_config()
    for key, entry in (config.get("reservation") or {}).items():
        site = entry["site"]
        assert not entry["used_in_v07"], f"{key}: the reservation says it is used after all"
        assert site not in config["dataset"]["sites"], f"{site} is built into the primary dataset"
        for name, split in config["splits"].items():
            assert site not in (split.get("test_sites") or []), f"{site} is a test site of {name}"
            assert site not in (split.get("train_sites") or []), f"{site} is a training site of {name}"
            assert site not in (split.get("sites") or []), f"{site} is used by {name}"
        assert entry["adaptation_fraction"] + entry["holdout_fraction"] == pytest.approx(1.0)
        assert not entry["rebuilds_primary_dataset"]


def test_missing_config_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_env_variable_overrides_root(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setenv("DRIAMS_ROOT", "D:/elsewhere")
    assert load_config(cfg)["paths"]["driams_root"] == "D:/elsewhere"


def test_dotenv_overrides_root_when_no_env_variable(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG, encoding="utf-8")
    (tmp_path / ".env").write_text("# local settings\nDRIAMS_ROOT=\"E:/data\"\n", encoding="utf-8")
    monkeypatch.delenv("DRIAMS_ROOT", raising=False)
    assert load_config(cfg)["paths"]["driams_root"] == "E:/data"


def test_keep_awake_context_manager():
    with keep_awake() as awake:
        assert awake is (sys.platform == "win32")


def test_human_bytes_uses_decimal_units():
    assert human_bytes(86161411795) == "86.16 GB"
    assert human_bytes(512) == "512 B"
