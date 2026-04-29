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
