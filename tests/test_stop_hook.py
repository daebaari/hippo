"""Test the Stop hook handler with mock daemon."""
from __future__ import annotations

import json

from hippo.capture.stop_hook import handle_stop
from hippo.config import EMBEDDING_DIM
from hippo.storage.capture import list_unprocessed_captures
from hippo.storage.multi_store import Scope, open_store
from hippo.storage.turn_embeddings import vector_search_turns


class FakeDaemon:
    def embed(self, texts):
        return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]
    def rerank(self, pairs):
        return [0.5] * len(pairs)


def test_handle_stop_persists_capture_and_turn_embedding(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    payload = json.dumps({
        "session_id": "sess-1",
        "user_message": "hello",
        "assistant_message": "hi back",
        "transcript_path": "/tmp/fake.jsonl",
        "cwd": str(tmp_path),
    })
    out = handle_stop(stdin_text=payload, daemon=FakeDaemon())
    assert out == ""

    s = open_store(Scope.global_())
    captures = list_unprocessed_captures(s.conn)
    assert len(captures) == 1
    assert captures[0].user_message == "hello"
    assert captures[0].assistant_message == "hi back"
    q = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    results = vector_search_turns(s.conn, q, top_k=5)
    assert len(results) == 1
    s.conn.close()


def test_handle_stop_empty_messages_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    payload = json.dumps({"session_id": "s", "cwd": str(tmp_path)})
    out = handle_stop(stdin_text=payload, daemon=FakeDaemon())
    assert out == ""
    s = open_store(Scope.global_())
    assert list_unprocessed_captures(s.conn) == []
    s.conn.close()
