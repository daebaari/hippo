"""Tests for light dream session-meta generation."""
from __future__ import annotations

from hippo.config import EMBEDDING_DIM
from hippo.dream.light import run_light_dream
from hippo.storage.bodies import list_bodies_by_scope
from hippo.storage.capture import CaptureRecord, enqueue_capture
from hippo.storage.multi_store import Scope, open_store


class FakeDaemon:
    def embed(self, texts):
        return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]


def test_light_dream_creates_session_meta_atom(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    enqueue_capture(
        s.conn,
        CaptureRecord(session_id="sess-A", user_message="hi", assistant_message="hello"),
    )
    enqueue_capture(
        s.conn,
        CaptureRecord(session_id="sess-A", user_message="more", assistant_message="ok"),
    )
    enqueue_capture(
        s.conn,
        CaptureRecord(session_id="sess-B", user_message="other", assistant_message="other"),
    )
    s.conn.close()

    stats = run_light_dream(scope=Scope.global_(), daemon=FakeDaemon())

    s2 = open_store(Scope.global_())
    sess_meta_bodies = [
        b
        for b in list_bodies_by_scope(s2.conn, "global")
        if b.scope == "global" and b.title.startswith("session-meta:")
    ]
    assert len(sess_meta_bodies) == 2  # one per unique session
    assert stats["sessions_summarized"] == 2
    s2.conn.close()


def test_light_dream_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    enqueue_capture(
        s.conn,
        CaptureRecord(session_id="sess-1", user_message="x", assistant_message="y"),
    )
    s.conn.close()

    run_light_dream(scope=Scope.global_(), daemon=FakeDaemon())
    stats = run_light_dream(scope=Scope.global_(), daemon=FakeDaemon())
    assert stats["sessions_summarized"] == 0  # already done
