"""Tests for capture_queue CRUD."""
from __future__ import annotations

import sqlite3

import pytest

from hippo.storage.capture import (
    CaptureRecord,
    enqueue_capture,
    list_unprocessed_captures,
    mark_captures_processed,
)
from hippo.storage.migrations import run_migrations


@pytest.fixture
def conn(sqlite_conn: sqlite3.Connection) -> sqlite3.Connection:
    run_migrations(sqlite_conn)
    return sqlite_conn


def test_enqueue_returns_id_and_persists(conn: sqlite3.Connection) -> None:
    queue_id = enqueue_capture(
        conn,
        CaptureRecord(
            session_id="sess1",
            project="kaleon",
            user_message="user msg",
            assistant_message="asst msg",
            transcript_path="/path/to/transcript.jsonl",
        ),
    )
    assert queue_id > 0
    rows = list_unprocessed_captures(conn)
    assert len(rows) == 1
    assert rows[0].session_id == "sess1"


def test_list_unprocessed_excludes_processed(conn: sqlite3.Connection) -> None:
    a = enqueue_capture(
        conn, CaptureRecord(session_id="s1", user_message="a", assistant_message="b")
    )
    b = enqueue_capture(
        conn, CaptureRecord(session_id="s1", user_message="c", assistant_message="d")
    )
    mark_captures_processed(conn, [a], run_id=1)
    remaining = list_unprocessed_captures(conn)
    assert {r.queue_id for r in remaining} == {b}


def test_unprocessed_ordered_by_created_at(conn: sqlite3.Connection) -> None:
    ids = []
    for i in range(3):
        ids.append(
            enqueue_capture(
                conn,
                CaptureRecord(
                    session_id=f"s{i}", user_message="u", assistant_message="a"
                ),
            )
        )
    rows = list_unprocessed_captures(conn)
    assert [r.queue_id for r in rows] == ids
