"""Integration test for the local Qwen 32B LLM wrapper.

Skipped unless RUN_LLM_TESTS=1 in env, because loading 18GB of weights
is too slow for default test runs.
"""
from __future__ import annotations

import os

import pytest

from hippo.models.llm import LocalLLM


@pytest.mark.skipif(os.environ.get("RUN_LLM_TESTS") != "1", reason="set RUN_LLM_TESTS=1 to run")
class TestLLM:
    @pytest.fixture(scope="class")
    def llm(self) -> LocalLLM:
        return LocalLLM.load()

    def test_generate_returns_string(self, llm: LocalLLM) -> None:
        out = llm.generate("Say hello.", temperature=0.0, max_tokens=20)
        assert isinstance(out, str)
        assert len(out) > 0

    def test_generate_json_mode_returns_parseable(self, llm: LocalLLM) -> None:
        import json
        out = llm.generate(
            'Return JSON: {"result": "ok"}. ONLY JSON, no other text.',
            temperature=0.0, max_tokens=50,
        )
        # The model may wrap in markdown fences; tolerate that
        s = out.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        assert json.loads(s)["result"] == "ok"
