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


class TestConfig:
    def test_load_missing_file_returns_defaults(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        c = cfg.load_config()
        assert c.backend == "local"

    def test_legacy_qwen_value_aliases_to_local(self, monkeypatch, tmp_path):
        """Existing config files with backend=qwen keep working."""
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        (tmp_path / "hippo-config.toml").write_text('backend = "qwen"\n')
        c = cfg.load_config()
        assert c.backend == "local"

    def test_round_trip(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        cfg.write_config(
            cfg.Config(
                backend="gemini",
                gemini_model_id="x",
                gemini_default_thinking_level="low",
            )
        )
        c = cfg.load_config()
        assert c.backend == "gemini"
        assert c.gemini_model_id == "x"
        assert c.gemini_default_thinking_level == "low"

    def test_load_malformed_toml_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        (tmp_path / "hippo-config.toml").write_text("not valid toml [[[")
        with pytest.raises(cfg.ConfigError):
            cfg.load_config()

    def test_load_unknown_backend_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        (tmp_path / "hippo-config.toml").write_text('backend = "pigeon"\n')
        with pytest.raises(cfg.ConfigError, match="backend must be"):
            cfg.load_config()

    def test_write_is_atomic(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        cfg.write_config(cfg.Config(backend="gemini"))
        leftovers = list(tmp_path.glob("*.tmp*"))
        assert leftovers == []


class TestApiKey:
    def test_no_key_anywhere_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        assert cfg.load_api_key() is None

    def test_google_env_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GOOGLE_API_KEY", "from-google")
        monkeypatch.setenv("GEMINI_API_KEY", "from-gemini")
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        (tmp_path / "hippo-secrets").write_text("GOOGLE_API_KEY=from-file\n")
        assert cfg.load_api_key() == "from-google"

    def test_gemini_env_when_no_google(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "from-gemini")
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        assert cfg.load_api_key() == "from-gemini"

    def test_secrets_file_when_no_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        (tmp_path / "hippo-secrets").write_text(
            "# comment\n\nGOOGLE_API_KEY=from-file\n"
        )
        assert cfg.load_api_key() == "from-file"

    def test_secrets_file_gemini_key_alias(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        (tmp_path / "hippo-secrets").write_text("GEMINI_API_KEY=alias-file\n")
        assert cfg.load_api_key() == "alias-file"
