"""CRUD for the bodies table.

bodies stores metadata for atom bodies; the actual content lives in
markdown files under <memory_dir>/bodies/<body_id>.md (managed by
body_files.py). Callers should keep the two in sync.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass


@dataclass
class BodyRecord:
    body_id: str
    file_path: str
    title: str
    scope: str
    source: str
    archived: bool = False
    archive_reason: str | None = None
    archived_in_favor_of: str | None = None
    created_at: int | None = None
    updated_at: int | None = None
    last_reviewed_at: int | None = None


def insert_body(conn: sqlite3.Connection, record: BodyRecord) -> None:
    now = record.created_at or int(time.time())
    conn.execute(
        "INSERT INTO bodies("
        "  body_id, file_path, title, scope, archived,"
        "  archive_reason, archived_in_favor_of, source, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.body_id,
            record.file_path,
            record.title,
            record.scope,
            int(record.archived),
            record.archive_reason,
            record.archived_in_favor_of,
            record.source,
            now,
            record.updated_at or now,
        ),
    )
    conn.commit()


def get_body(conn: sqlite3.Connection, body_id: str) -> BodyRecord | None:
    row = conn.execute("SELECT * FROM bodies WHERE body_id = ?", (body_id,)).fetchone()
    if row is None:
        return None
    return _row_to_record(row)


def archive_body(
    conn: sqlite3.Connection,
    body_id: str,
    *,
    reason: str,
    in_favor_of: str | None = None,
) -> None:
    conn.execute(
        "UPDATE bodies SET archived = 1, archive_reason = ?, "
        "archived_in_favor_of = ?, updated_at = ? WHERE body_id = ?",
        (reason, in_favor_of, int(time.time()), body_id),
    )
    conn.commit()


def list_bodies_by_scope(conn: sqlite3.Connection, scope: str) -> list[BodyRecord]:
    rows = conn.execute(
        "SELECT * FROM bodies WHERE scope = ? AND archived = 0 ORDER BY updated_at DESC",
        (scope,),
    ).fetchall()
    return [_row_to_record(r) for r in rows]


def update_last_reviewed_at(conn: sqlite3.Connection, body_id: str) -> None:
    """Stamp last_reviewed_at = now for a body."""
    conn.execute(
        "UPDATE bodies SET last_reviewed_at = ? WHERE body_id = ?",
        (int(time.time()), body_id),
    )
    conn.commit()


def find_oldest_unreviewed_active(
    conn: sqlite3.Connection, *, scope: str, limit: int
) -> list[BodyRecord]:
    """Active bodies in scope, NULL last_reviewed_at first, then oldest first.

    Tie-breaks deterministically by body_id ASC.
    """
    rows = conn.execute(
        "SELECT * FROM bodies "
        "WHERE archived = 0 AND scope = ? "
        "ORDER BY COALESCE(last_reviewed_at, 0) ASC, body_id ASC "
        "LIMIT ?",
        (scope, limit),
    ).fetchall()
    return [_row_to_record(r) for r in rows]


def _row_to_record(row: sqlite3.Row) -> BodyRecord:
    return BodyRecord(
        body_id=row["body_id"],
        file_path=row["file_path"],
        title=row["title"],
        scope=row["scope"],
        archived=bool(row["archived"]),
        archive_reason=row["archive_reason"],
        archived_in_favor_of=row["archived_in_favor_of"],
        source=row["source"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_reviewed_at=row["last_reviewed_at"],
    )
