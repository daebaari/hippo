"""Tests for the toggle-related additions in hippo.config."""
from __future__ import annotations

import pytest

from hippo import config as cfg


class TestConfigPaths:
    def test_config_path_in_claude_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        assert cfg.config_path() == tmp_path / "hippo-config.toml"

    def test_secrets_path_in_claude_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        assert cfg.secrets_path() == tmp_path / "hippo-secrets"

    def test_default_config_dir_is_claude_home(self, monkeypatch):
        monkeypatch.delenv("HIPPO_CONFIG_DIR", raising=False)
        assert cfg.config_path().parent == cfg.CLAUDE_HOME


class TestConfigError:
    def test_config_error_is_runtime_error(self):
        assert issubclass(cfg.ConfigError, RuntimeError)

    def test_config_error_carries_message(self):
        err = cfg.ConfigError("boom")
        assert str(err) == "boom"
