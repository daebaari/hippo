"""Tests for dream cleanup phase (finalize_processed_captures)."""
from __future__ import annotations

from hippo.config import EMBEDDING_DIM
from hippo.dream.cleanup import finalize_processed_captures
from hippo.storage.capture import CaptureRecord, enqueue_capture
from hippo.storage.multi_store import Scope, open_store
from hippo.storage.turn_embeddings import insert_turn_embedding


def test_finalize_processed_captures_marks_processed_and_deletes_embeddings(
    tmp_path, monkeypatch
) -> None:
    """With one capture and one turn_embedding linked to it, verify after
    finalize_processed_captures(...):
    - capture's processed_at is non-null
    - the turn_embedding row is gone
    """
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())

    # Enqueue a capture
    qid = enqueue_capture(
        s.conn,
        CaptureRecord(session_id="sess-A", user_message="hi"),
    )

    # Insert a turn embedding linked to it
    insert_turn_embedding(
        s.conn,
        capture_id=qid,
        summary="hi",
        embedding=[1.0] + [0.0] * (EMBEDDING_DIM - 1),
    )

    # Verify setup: capture not yet processed, embedding exists
    row = s.conn.execute(
        "SELECT processed_at FROM capture_queue WHERE queue_id=?", (qid,)
    ).fetchone()
    assert row["processed_at"] is None

    count = s.conn.execute(
        "SELECT COUNT(*) AS c FROM turn_embeddings WHERE capture_id=?", (qid,)
    ).fetchone()
    assert count["c"] == 1

    # Call finalize_processed_captures
    finalize_processed_captures(store=s, queue_ids=[qid], run_id=42)

    # Verify: processed_at is now set
    row = s.conn.execute(
        "SELECT processed_at FROM capture_queue WHERE queue_id=?", (qid,)
    ).fetchone()
    assert row["processed_at"] is not None

    # Verify: turn_embedding is gone
    count = s.conn.execute(
        "SELECT COUNT(*) AS c FROM turn_embeddings WHERE capture_id=?", (qid,)
    ).fetchone()
    assert count["c"] == 0

    s.conn.close()


def test_finalize_processed_captures_empty_list_is_noop(tmp_path, monkeypatch) -> None:
    """finalize_processed_captures with empty queue_ids should be a no-op."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())

    # Should not raise or do anything
    finalize_processed_captures(store=s, queue_ids=[], run_id=1)

    s.conn.close()
