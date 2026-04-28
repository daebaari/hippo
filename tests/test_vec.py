"""Tests for vector embeddings + sqlite-vec search."""
from __future__ import annotations

import sqlite3

import pytest

from hippo.config import EMBEDDING_DIM
from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.heads import HeadRecord, insert_head
from hippo.storage.migrations import run_migrations
from hippo.storage.vec import (
    delete_head_embedding,
    insert_head_embedding,
    vector_search_heads,
)


@pytest.fixture
def conn(sqlite_conn: sqlite3.Connection) -> sqlite3.Connection:
    run_migrations(sqlite_conn)
    insert_body(
        sqlite_conn,
        BodyRecord(body_id="b1", file_path="bodies/b1.md", title="t",
                   scope="global", source="manual"),
    )
    for hid, summary in [("h1", "apples"), ("h2", "oranges"), ("h3", "bananas")]:
        insert_head(sqlite_conn, HeadRecord(head_id=hid, body_id="b1", summary=summary))
    return sqlite_conn


def test_insert_and_search_returns_closest(conn: sqlite3.Connection) -> None:
    # Three orthogonal-ish vectors
    v1 = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    v2 = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
    v3 = [0.0, 0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 3)
    insert_head_embedding(conn, "h1", v1)
    insert_head_embedding(conn, "h2", v2)
    insert_head_embedding(conn, "h3", v3)
    # Query closer to v1
    query = [0.99, 0.01, 0.0] + [0.0] * (EMBEDDING_DIM - 3)
    results = vector_search_heads(conn, query, top_k=2)
    assert len(results) == 2
    assert results[0].head_id == "h1"
    assert results[0].distance < results[1].distance


def test_dimension_mismatch_raises(conn: sqlite3.Connection) -> None:
    bad = [1.0] * (EMBEDDING_DIM - 1)
    with pytest.raises(ValueError):
        insert_head_embedding(conn, "h1", bad)


def test_delete_head_embedding(conn: sqlite3.Connection) -> None:
    v = [0.5] * EMBEDDING_DIM
    insert_head_embedding(conn, "h1", v)
    insert_head_embedding(conn, "h2", v)
    delete_head_embedding(conn, "h1")
    results = vector_search_heads(conn, v, top_k=10)
    head_ids = {r.head_id for r in results}
    assert "h1" not in head_ids
    assert "h2" in head_ids
