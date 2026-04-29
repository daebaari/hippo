"""Tests for bodies table CRUD."""
from __future__ import annotations

import sqlite3

import pytest

from hippo.storage.bodies import (
    BodyRecord,
    archive_body,
    get_body,
    insert_body,
    list_bodies_by_scope,
)
from hippo.storage.migrations import run_migrations


@pytest.fixture
def conn(sqlite_conn: sqlite3.Connection) -> sqlite3.Connection:
    run_migrations(sqlite_conn)
    return sqlite_conn


def test_insert_and_get_round_trip(conn: sqlite3.Connection) -> None:
    record = BodyRecord(
        body_id="01HZK1234567890ABCDEFGHIJK",
        file_path="bodies/01HZK1234567890ABCDEFGHIJK.md",
        title="Kalshi taker fee",
        scope="project:kaleon",
        source="manual",
    )
    insert_body(conn, record)
    loaded = get_body(conn, record.body_id)
    assert loaded is not None
    assert loaded.body_id == record.body_id
    assert loaded.title == record.title
    assert loaded.scope == record.scope
    assert loaded.archived is False


def test_get_missing_returns_none(conn: sqlite3.Connection) -> None:
    assert get_body(conn, "missing") is None


def test_archive_body_marks_archived_with_reason(conn: sqlite3.Connection) -> None:
    record = BodyRecord(
        body_id="b1", file_path="bodies/b1.md", title="t", scope="global", source="manual"
    )
    insert_body(conn, record)
    archive_body(conn, "b1", reason="contradicted_by:b2", in_favor_of="b2")
    loaded = get_body(conn, "b1")
    assert loaded is not None
    assert loaded.archived is True
    assert loaded.archive_reason == "contradicted_by:b2"
    assert loaded.archived_in_favor_of == "b2"


def test_list_bodies_by_scope_excludes_archived(conn: sqlite3.Connection) -> None:
    insert_body(
        conn,
        BodyRecord(
            body_id="a",
            file_path="bodies/a.md",
            title="A",
            scope="global",
            source="manual",
        ),
    )
    insert_body(
        conn,
        BodyRecord(
            body_id="b",
            file_path="bodies/b.md",
            title="B",
            scope="global",
            source="manual",
        ),
    )
    insert_body(
        conn,
        BodyRecord(
            body_id="c",
            file_path="bodies/c.md",
            title="C",
            scope="project:kaleon",
            source="manual",
        ),
    )
    archive_body(conn, "b", reason="superseded")
    globals_active = list_bodies_by_scope(conn, "global")
    ids = {r.body_id for r in globals_active}
    assert ids == {"a"}


def test_body_record_includes_last_reviewed_at(temp_memory_dir, sqlite_conn):
    """BodyRecord exposes last_reviewed_at; defaults to None on insert."""
    from hippo.storage.bodies import BodyRecord, get_body, insert_body
    from hippo.storage.migrations import run_migrations

    run_migrations(sqlite_conn)
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="bid-1", file_path="bodies/bid-1.md", title="t",
            scope="global", source="test",
        ),
    )
    rec = get_body(sqlite_conn, "bid-1")
    assert rec is not None
    assert rec.last_reviewed_at is None
