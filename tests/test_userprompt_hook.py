"""Test the UserPromptSubmit hook handler with mock daemon."""
from __future__ import annotations

import io
import json

from hippo.capture import userprompt_hook
from hippo.capture.userprompt_hook import handle_userprompt_submit, main


class FakeDaemon:
    def __init__(self):
        self.embed_calls: list[list[str]] = []
        self.rerank_calls: list[list[tuple[str, str]]] = []

    def embed(self, texts):
        from hippo.config import EMBEDDING_DIM
        self.embed_calls.append(list(texts))
        out = []
        for t in texts:
            v = [0.0] * EMBEDDING_DIM
            v[len(t) % EMBEDDING_DIM] = 1.0
            out.append(v)
        return out

    def rerank(self, pairs):
        self.rerank_calls.append(list(pairs))
        return [1.0 / (1 + i) for i, _ in enumerate(pairs)]


def _full_envelope(prompt: str, cwd: str) -> str:
    return json.dumps({
        "session_id": "sess-1",
        "transcript_path": "/tmp/transcript.jsonl",
        "cwd": cwd,
        "permission_mode": "default",
        "hook_event_name": "UserPromptSubmit",
        "prompt": prompt,
    })


def test_hook_emits_memory_block_when_results_exist(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    # Insert head with summary "x" (len 1) → one-hot at index 1.
    # Query "q" (len 1) → same one-hot at index 1 → vector match (distance 0).
    from hippo.config import EMBEDDING_DIM
    from hippo.storage.bodies import BodyRecord, insert_body
    from hippo.storage.heads import HeadRecord, insert_head
    from hippo.storage.multi_store import Scope, open_store
    from hippo.storage.vec import insert_head_embedding
    s = open_store(Scope.global_())
    insert_body(
        s.conn,
        BodyRecord(
            body_id="b1", file_path="bodies/b1.md", title="t",
            scope="global", source="manual",
        ),
    )
    insert_head(s.conn, HeadRecord(head_id="h1", body_id="b1", summary="x"))
    v = [0.0] * EMBEDDING_DIM
    v[len("x") % EMBEDDING_DIM] = 1.0
    insert_head_embedding(s.conn, "h1", v)
    s.conn.close()

    daemon = FakeDaemon()
    stdin_payload = _full_envelope("q", str(tmp_path))
    out = handle_userprompt_submit(stdin_text=stdin_payload, daemon=daemon)

    assert "<memory>" in out
    assert "h1" in out
    assert daemon.embed_calls == [["q"]]
    assert len(daemon.rerank_calls) == 1
    # rerank pair should contain the query as first element
    assert daemon.rerank_calls[0][0][0] == "q"


def test_hook_emits_nothing_when_no_results(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    daemon = FakeDaemon()
    stdin_payload = _full_envelope("hi", str(tmp_path))
    out = handle_userprompt_submit(stdin_text=stdin_payload, daemon=daemon)
    assert out == ""
    # Daemon was called to embed the query, but rerank had nothing to rank.
    assert daemon.embed_calls == [["hi"]]
    assert daemon.rerank_calls == []


def test_hook_emits_nothing_on_empty_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    daemon = FakeDaemon()
    stdin_payload = json.dumps({"prompt": "", "cwd": str(tmp_path)})
    out = handle_userprompt_submit(stdin_text=stdin_payload, daemon=daemon)
    assert out == ""
    # Empty prompt short-circuits before any daemon call.
    assert daemon.embed_calls == []
    assert daemon.rerank_calls == []


def test_main_swallows_exceptions(monkeypatch, capsys):
    monkeypatch.setattr(userprompt_hook.sys, "stdin", io.StringIO("not json"))
    rc = main()
    assert rc == 0
    captured = capsys.readouterr()
    assert "hippo userprompt-hook error" in captured.err
