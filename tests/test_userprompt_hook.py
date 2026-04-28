"""Test the UserPromptSubmit hook handler with mock daemon."""
from __future__ import annotations

import json

from hippo.capture.userprompt_hook import handle_userprompt_submit


class FakeDaemon:
    def embed(self, texts):
        from hippo.config import EMBEDDING_DIM
        out = []
        for t in texts:
            v = [0.0] * EMBEDDING_DIM
            v[len(t) % EMBEDDING_DIM] = 1.0
            out.append(v)
        return out
    def rerank(self, pairs):
        return [1.0 / (1 + i) for i, _ in enumerate(pairs)]


def test_hook_emits_memory_block_when_results_exist(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    # populate one head
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
    insert_head(s.conn, HeadRecord(head_id="h1", body_id="b1", summary="something"))
    v = [0.0] * EMBEDDING_DIM
    v[len("something") % EMBEDDING_DIM] = 1.0
    insert_head_embedding(s.conn, "h1", v)
    s.conn.close()

    stdin_payload = json.dumps({
        "user_message": "tell me about something",
        "session_id": "sess-1",
        "cwd": str(tmp_path),
    })
    out = handle_userprompt_submit(stdin_text=stdin_payload, daemon=FakeDaemon())
    assert out == "" or "<memory>" in out


def test_hook_emits_nothing_when_no_results(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    stdin_payload = json.dumps({
        "user_message": "hi", "session_id": "s", "cwd": str(tmp_path),
    })
    out = handle_userprompt_submit(stdin_text=stdin_payload, daemon=FakeDaemon())
    assert out == ""
