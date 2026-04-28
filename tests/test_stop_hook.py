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


def test_handle_stop_real_envelope_reads_user_from_transcript(tmp_path, monkeypatch):
    """Claude Code's actual Stop envelope: last_assistant_message in payload,
    user message in JSONL transcript at transcript_path."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "What's 2+2?"}}) + "\n"
        + json.dumps({"type": "assistant", "message": {"role": "assistant",
            "content": [{"type": "text", "text": "4."}]}}) + "\n"
    )

    payload = json.dumps({
        "session_id": "abc-123",
        "transcript_path": str(transcript),
        "cwd": str(tmp_path),
        "hook_event_name": "Stop",
        "stop_hook_active": "False",
        "permission_mode": "default",
        "last_assistant_message": "4.",
    })
    out = handle_stop(stdin_text=payload, daemon=FakeDaemon())
    assert out == ""

    s = open_store(Scope.global_())
    captures = list_unprocessed_captures(s.conn)
    assert len(captures) == 1
    assert captures[0].user_message == "What's 2+2?"
    assert captures[0].assistant_message == "4."
    s.conn.close()


def test_handle_stop_real_envelope_handles_user_with_attachment_blocks(tmp_path, monkeypatch):
    """User messages with attachments arrive as list-of-content-blocks."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": [
            {"type": "text", "text": "Look at this:"},
            {"type": "image", "source": {}},
            {"type": "text", "text": "what is it?"},
        ]}}) + "\n"
    )

    payload = json.dumps({
        "session_id": "abc",
        "transcript_path": str(transcript),
        "cwd": str(tmp_path),
        "last_assistant_message": "It's a screenshot.",
    })
    out = handle_stop(stdin_text=payload, daemon=FakeDaemon())
    assert out == ""

    s = open_store(Scope.global_())
    captures = list_unprocessed_captures(s.conn)
    assert len(captures) == 1
    assert "Look at this:" in (captures[0].user_message or "")
    assert "what is it?" in (captures[0].user_message or "")
    s.conn.close()


def test_handle_stop_missing_transcript_falls_back_to_assistant_only(tmp_path, monkeypatch):
    """If transcript path is missing/unreadable, still record the assistant turn."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    payload = json.dumps({
        "session_id": "x",
        "transcript_path": "/nonexistent/path.jsonl",
        "cwd": str(tmp_path),
        "last_assistant_message": "I responded.",
    })
    out = handle_stop(stdin_text=payload, daemon=FakeDaemon())
    assert out == ""
    s = open_store(Scope.global_())
    captures = list_unprocessed_captures(s.conn)
    assert len(captures) == 1
    assert captures[0].user_message is None
    assert captures[0].assistant_message == "I responded."
    s.conn.close()
