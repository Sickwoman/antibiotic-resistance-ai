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
