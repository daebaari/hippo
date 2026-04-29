"""Test atomize phase with a mock LLM."""
from __future__ import annotations

import json

from hippo.dream.atomize import atomize_session
from hippo.storage.bodies import list_bodies_by_scope
from hippo.storage.capture import CaptureRecord, enqueue_capture
from hippo.storage.heads import list_heads_for_body
from hippo.storage.multi_store import Scope, open_store


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[str] = []
    def generate_chat(self, messages, *, temperature, max_tokens, thinking_level=None):
        self.calls.append(messages[-1]["content"])
        return self.response


class FakeDaemon:
    def embed(self, texts):
        from hippo.config import EMBEDDING_DIM
        return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]


def test_atomize_creates_bodies_and_heads(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    enqueue_capture(s.conn, CaptureRecord(
        session_id="sess-A", user_message="we use postgres", assistant_message="ok",
    ))
    s.conn.close()

    fake_response = json.dumps([{
        "title": "We use postgres",
        "body": "Project uses postgres for the main DB.",
        "scope": "global",
        "heads": ["uses postgres", "main DB is postgres"],
    }])
    llm = FakeLLM(fake_response)
    daemon = FakeDaemon()

    s = open_store(Scope.global_())
    n = atomize_session(
        store=s, session_id="sess-A", project=None, run_id=1,
        llm=llm, daemon=daemon,
    )
    assert n == 1
    bodies = list_bodies_by_scope(s.conn, "global")
    assert len(bodies) == 1
    assert bodies[0].title == "We use postgres"
    heads = list_heads_for_body(s.conn, bodies[0].body_id)
    assert len(heads) == 2
    s.conn.close()


def test_atomize_handles_empty_response(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    enqueue_capture(s.conn, CaptureRecord(
        session_id="sess-A", user_message="hi", assistant_message="hello",
    ))
    s.conn.close()

    s = open_store(Scope.global_())
    n = atomize_session(
        store=s, session_id="sess-A", project=None, run_id=1,
        llm=FakeLLM("[]"), daemon=FakeDaemon(),
    )
    assert n == 0
    s.conn.close()
