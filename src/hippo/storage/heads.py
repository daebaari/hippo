"""CRUD for the heads table.

Heads are search affordances: each head has a 1-2 sentence summary and
points to a body. Multiple heads per body is the design — each captures a
different angle of the body's content, increasing retrieval surface area.

Embedding storage is in head_embeddings (vec0 virtual table) and is
managed by hippo.storage.vec.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass


@dataclass
class HeadRecord:
    head_id: str
    body_id: str
    summary: str
    archived: bool = False
    archive_reason: str | None = None
    last_retrieved_at: int | None = None
    retrieval_count: int = 0
    created_at: int | None = None


def insert_head(conn: sqlite3.Connection, record: HeadRecord) -> None:
    now = record.created_at or int(time.time())
    conn.execute(
        "INSERT INTO heads("
        "  head_id, body_id, summary, archived, archive_reason,"
        "  last_retrieved_at, retrieval_count, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.head_id,
            record.body_id,
            record.summary,
            int(record.archived),
            record.archive_reason,
            record.last_retrieved_at,
            record.retrieval_count,
            now,
        ),
    )
    conn.commit()


def get_head(conn: sqlite3.Connection, head_id: str) -> HeadRecord | None:
    row = conn.execute("SELECT * FROM heads WHERE head_id = ?", (head_id,)).fetchone()
    if row is None:
        return None
    return _row_to_record(row)


def archive_head(conn: sqlite3.Connection, head_id: str, *, reason: str) -> None:
    conn.execute(
        "UPDATE heads SET archived = 1, archive_reason = ? WHERE head_id = ?",
        (reason, head_id),
    )
    conn.commit()


def increment_retrieval(conn: sqlite3.Connection, head_id: str) -> None:
    """Bump retrieval_count and stamp last_retrieved_at."""
    conn.execute(
        "UPDATE heads SET retrieval_count = retrieval_count + 1, "
        "last_retrieved_at = ? WHERE head_id = ?",
        (int(time.time()), head_id),
    )
    conn.commit()


def list_heads_for_body(conn: sqlite3.Connection, body_id: str) -> list[HeadRecord]:
    rows = conn.execute(
        "SELECT * FROM heads WHERE body_id = ? AND archived = 0 ORDER BY created_at",
        (body_id,),
    ).fetchall()
    return [_row_to_record(r) for r in rows]


def _row_to_record(row: sqlite3.Row) -> HeadRecord:
    return HeadRecord(
        head_id=row["head_id"],
        body_id=row["body_id"],
        summary=row["summary"],
        archived=bool(row["archived"]),
        archive_reason=row["archive_reason"],
        last_retrieved_at=row["last_retrieved_at"],
        retrieval_count=row["retrieval_count"],
        created_at=row["created_at"],
    )
