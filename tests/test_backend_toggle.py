"""Tests for the /hippo-backend slash-command CLI module."""
from __future__ import annotations

import pytest

from hippo.cli import backend_toggle


class TestStatus:
    def test_status_default(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        rc = backend_toggle.main([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "backend: qwen" in out
        assert "api_key: not detected" in out

    def test_status_with_env_key(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("GOOGLE_API_KEY", "secret")
        rc = backend_toggle.main([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "api_key: detected (env)" in out


class TestSwitch:
    def test_switch_to_qwen(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        rc = backend_toggle.main(["qwen"])
        assert rc == 0
        from hippo.config import load_config
        assert load_config().backend == "qwen"
        out = capsys.readouterr().out
        assert "switched to qwen" in out.lower()

    def test_switch_to_gemini_with_key(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        rc = backend_toggle.main(["gemini"])
        assert rc == 0
        from hippo.config import load_config
        assert load_config().backend == "gemini"

    def test_switch_to_gemini_without_key_warns(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        rc = backend_toggle.main(["gemini"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "no API key" in captured.err

    def test_invalid_backend(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        with pytest.raises(SystemExit) as exc_info:
            backend_toggle.main(["pigeon"])
        assert exc_info.value.code == 2
