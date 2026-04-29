"""Tests for select_llm() dispatch and GeminiLLM retry hardening."""
from __future__ import annotations

import pytest

from hippo.models.llm import GeminiLLM


class _FakeAPIError(Exception):
    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(f"http {code}")


class _FakeModels:
    def __init__(self, errors_to_raise: list[Exception]) -> None:
        self._errors = list(errors_to_raise)
        self.calls = 0

    def generate_content(self, *, model, contents, config):
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        class _Resp:
            text = "ok"
        return _Resp()


class _FakeClient:
    def __init__(self, errors_to_raise: list[Exception]) -> None:
        self.models = _FakeModels(errors_to_raise)


class TestGeminiRetry:
    def _llm(self, errors):
        return GeminiLLM(
            client=_FakeClient(errors),
            model_id="x",
            default_thinking_level="high",
            max_attempts=4,
        )

    def test_retries_on_oserror(self, monkeypatch):
        monkeypatch.setattr(
            "hippo.models.llm._sleep", lambda *_a, **_k: None, raising=False
        )
        llm = self._llm([OSError("conn reset")])
        out = llm._call_with_retry(contents="hi", config=None)
        assert out == "ok"
        assert llm.client.models.calls == 2

    def test_retries_on_timeout(self, monkeypatch):
        monkeypatch.setattr(
            "hippo.models.llm._sleep", lambda *_a, **_k: None, raising=False
        )
        llm = self._llm([TimeoutError("slow")])
        out = llm._call_with_retry(contents="hi", config=None)
        assert out == "ok"

    def test_propagates_after_max_attempts(self, monkeypatch):
        monkeypatch.setattr(
            "hippo.models.llm._sleep", lambda *_a, **_k: None, raising=False
        )
        llm = self._llm([OSError("a"), OSError("b"), OSError("c"), OSError("d")])
        with pytest.raises(OSError):
            llm._call_with_retry(contents="hi", config=None)
        assert llm.client.models.calls == 4

    def test_does_not_retry_non_retryable_apierror(self, monkeypatch):
        from google.genai import errors  # type: ignore[import-untyped]
        monkeypatch.setattr(
            "hippo.models.llm._sleep", lambda *_a, **_k: None, raising=False
        )
        e = errors.APIError.__new__(errors.APIError)
        e.code = 401  # type: ignore[attr-defined]
        llm = self._llm([e])
        with pytest.raises(errors.APIError):
            llm._call_with_retry(contents="hi", config=None)
        assert llm.client.models.calls == 1


class TestSelectLLM:
    def _patch_loaders(self, monkeypatch):
        from hippo.models import llm as llm_mod
        local_marker = object()
        gemini_calls: list[dict] = []

        def _fake_local_load() -> object:
            return local_marker

        def _fake_gemini_load(*, api_key, model_id, default_thinking_level):
            gemini_calls.append(
                dict(api_key=api_key, model_id=model_id,
                     default_thinking_level=default_thinking_level)
            )
            return ("gemini", api_key, model_id)

        monkeypatch.setattr(llm_mod.LocalLLM, "load", staticmethod(_fake_local_load))
        monkeypatch.setattr(llm_mod.GeminiLLM, "load", staticmethod(_fake_gemini_load))
        return local_marker, gemini_calls

    def test_default_returns_local(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        local, _ = self._patch_loaders(monkeypatch)
        from hippo.models.llm import select_llm
        assert select_llm() is local

    def test_qwen_explicit(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        (tmp_path / "hippo-config.toml").write_text('backend = "qwen"\n')
        local, _ = self._patch_loaders(monkeypatch)
        from hippo.models.llm import select_llm
        assert select_llm() is local

    def test_gemini_with_env_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("GOOGLE_API_KEY", "from-env")
        (tmp_path / "hippo-config.toml").write_text(
            'backend = "gemini"\n[gemini]\nmodel_id = "m1"\n'
        )
        _, calls = self._patch_loaders(monkeypatch)
        from hippo.models.llm import select_llm
        out = select_llm()
        assert out[0] == "gemini"
        assert calls[0]["api_key"] == "from-env"
        assert calls[0]["model_id"] == "m1"

    def test_gemini_with_secrets_file_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        (tmp_path / "hippo-config.toml").write_text('backend = "gemini"\n')
        (tmp_path / "hippo-secrets").write_text("GOOGLE_API_KEY=from-file\n")
        _, calls = self._patch_loaders(monkeypatch)
        from hippo.models.llm import select_llm
        select_llm()
        assert calls[0]["api_key"] == "from-file"

    def test_gemini_no_key_strict_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        (tmp_path / "hippo-config.toml").write_text('backend = "gemini"\n')
        self._patch_loaders(monkeypatch)
        from hippo.config import ConfigError
        from hippo.models.llm import select_llm
        with pytest.raises(ConfigError, match="API key"):
            select_llm(strict=True)

    def test_gemini_no_key_nonstrict_falls_back(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        (tmp_path / "hippo-config.toml").write_text('backend = "gemini"\n')
        local, _ = self._patch_loaders(monkeypatch)
        from hippo.models.llm import select_llm
        out = select_llm(strict=False)
        assert out is local
        err = capsys.readouterr().err
        assert "WARNING" in err and "qwen" in err.lower()


def test_gemini_missing_dep_raises_configerror(monkeypatch, tmp_path):
    """If google-genai is missing, GeminiLLM.load raises ConfigError with install hint."""
    monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    (tmp_path / "hippo-config.toml").write_text('backend = "gemini"\n')
    # Force ImportError by blocking the `google` package
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "google", None)
    from hippo.config import ConfigError
    from hippo.models.llm import select_llm
    with pytest.raises(ConfigError, match="uv sync --extra gemini"):
        select_llm(strict=True)
