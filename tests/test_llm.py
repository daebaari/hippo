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


@pytest.mark.skipif(
    os.environ.get("RUN_LLM_TESTS") != "1",
    reason="set RUN_LLM_TESTS=1 to run",
)
def test_atomize_noise_field_real_llm(tmp_path, monkeypatch):
    """Real-LLM smoke: noise atoms get dropped, durable atoms survive."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")

    from hippo.dream.atomize import atomize_session
    from hippo.models.llm import select_llm
    from hippo.storage.bodies import list_bodies_by_scope
    from hippo.storage.capture import CaptureRecord, enqueue_capture
    from hippo.storage.multi_store import Scope, open_store

    s = open_store(Scope.global_())
    # One durable + one noise-y interaction
    enqueue_capture(s.conn, CaptureRecord(
        session_id="sess-mix",
        user_message="we use postgres for the main DB and that's a hard requirement",
        assistant_message="got it; recording",
    ))
    enqueue_capture(s.conn, CaptureRecord(
        session_id="sess-mix",
        user_message="status",
        assistant_message="ok",
    ))
    s.conn.close()

    class _DaemonStub:
        def embed(self, texts):
            from hippo.config import EMBEDDING_DIM
            return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]

    llm = select_llm()
    s = open_store(Scope.global_())
    n = atomize_session(
        store=s, session_id="sess-mix", project=None, run_id=1,
        llm=llm, daemon=_DaemonStub(),
    )
    # We expect the durable atom to be kept and the "status / ok" to be dropped.
    # The exact n depends on how the LLM splits the durable interaction,
    # but it should be >= 1 and the bodies should NOT include obvious noise.
    bodies = list_bodies_by_scope(s.conn, "global")
    assert n >= 1
    assert all("status" not in b.title.lower() for b in bodies)
    assert all(len(b.title) > 4 for b in bodies)  # weak sanity that noise isn't a title
    s.conn.close()
