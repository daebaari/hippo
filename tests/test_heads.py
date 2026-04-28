"""Tests for heads table CRUD."""
from __future__ import annotations

import sqlite3

import pytest

from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.heads import (
    HeadRecord,
    archive_head,
    get_head,
    increment_retrieval,
    insert_head,
    list_heads_for_body,
)
from hippo.storage.migrations import run_migrations


@pytest.fixture
def conn(sqlite_conn: sqlite3.Connection) -> sqlite3.Connection:
    run_migrations(sqlite_conn)
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="b1", file_path="bodies/b1.md", title="t",
            scope="global", source="manual",
        ),
    )
    return sqlite_conn


def test_insert_and_get(conn: sqlite3.Connection) -> None:
    head = HeadRecord(head_id="h1", body_id="b1", summary="rules of the road")
    insert_head(conn, head)
    loaded = get_head(conn, "h1")
    assert loaded is not None
    assert loaded.summary == "rules of the road"
    assert loaded.archived is False
    assert loaded.retrieval_count == 0


def test_get_missing_head_returns_none(conn: sqlite3.Connection) -> None:
    assert get_head(conn, "missing") is None


def test_archive_head(conn: sqlite3.Connection) -> None:
    insert_head(conn, HeadRecord(head_id="h1", body_id="b1", summary="x"))
    archive_head(conn, "h1", reason="redundant:h2")
    loaded = get_head(conn, "h1")
    assert loaded is not None
    assert loaded.archived is True
    assert loaded.archive_reason == "redundant:h2"


def test_increment_retrieval_updates_count_and_timestamp(conn: sqlite3.Connection) -> None:
    insert_head(conn, HeadRecord(head_id="h1", body_id="b1", summary="x"))
    increment_retrieval(conn, "h1")
    increment_retrieval(conn, "h1")
    loaded = get_head(conn, "h1")
    assert loaded is not None
    assert loaded.retrieval_count == 2
    assert loaded.last_retrieved_at is not None


def test_list_heads_for_body_excludes_archived(conn: sqlite3.Connection) -> None:
    insert_head(conn, HeadRecord(head_id="h1", body_id="b1", summary="A"))
    insert_head(conn, HeadRecord(head_id="h2", body_id="b1", summary="B"))
    insert_head(conn, HeadRecord(head_id="h3", body_id="b1", summary="C"))
    archive_head(conn, "h2", reason="redundant")
    active = list_heads_for_body(conn, "b1")
    ids = {h.head_id for h in active}
    assert ids == {"h1", "h3"}
