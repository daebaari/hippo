"""Tests for turn-level embeddings (raw turn searchability)."""
from __future__ import annotations

import sqlite3

import pytest

from hippo.config import EMBEDDING_DIM
from hippo.storage.capture import CaptureRecord, enqueue_capture
from hippo.storage.migrations import run_migrations
from hippo.storage.turn_embeddings import (
    delete_turn_embeddings_for_captures,
    insert_turn_embedding,
    vector_search_turns,
)


@pytest.fixture
def conn(sqlite_conn: sqlite3.Connection) -> sqlite3.Connection:
    run_migrations(sqlite_conn)
    return sqlite_conn


def test_insert_then_search_returns_closest(conn: sqlite3.Connection) -> None:
    cap_a = enqueue_capture(conn, CaptureRecord(session_id="s", user_message="u", assistant_message="a"))
    cap_b = enqueue_capture(conn, CaptureRecord(session_id="s", user_message="u", assistant_message="a"))
    v_a = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    v_b = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
    insert_turn_embedding(conn, capture_id=cap_a, summary="apples", embedding=v_a)
    insert_turn_embedding(conn, capture_id=cap_b, summary="oranges", embedding=v_b)
    query = [0.95, 0.05] + [0.0] * (EMBEDDING_DIM - 2)
    results = vector_search_turns(conn, query, top_k=2)
    assert len(results) == 2
    assert results[0].capture_id == cap_a


def test_delete_turn_embeddings_for_captures(conn: sqlite3.Connection) -> None:
    cap_a = enqueue_capture(conn, CaptureRecord(session_id="s", user_message="u", assistant_message="a"))
    cap_b = enqueue_capture(conn, CaptureRecord(session_id="s", user_message="u", assistant_message="a"))
    v = [0.5] * EMBEDDING_DIM
    insert_turn_embedding(conn, capture_id=cap_a, summary="x", embedding=v)
    insert_turn_embedding(conn, capture_id=cap_b, summary="y", embedding=v)
    delete_turn_embeddings_for_captures(conn, [cap_a])
    results = vector_search_turns(conn, v, top_k=10)
    cap_ids = {r.capture_id for r in results}
    assert cap_a not in cap_ids
    assert cap_b in cap_ids
