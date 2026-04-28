"""CRUD for capture_queue (Stop hook writes; heavy dream consumes)."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class CaptureRecord:
    session_id: str
    user_message: Optional[str] = None
    assistant_message: Optional[str] = None
    project: Optional[str] = None
    transcript_path: Optional[str] = None
    queue_id: Optional[int] = None
    created_at: Optional[int] = None
    processed_at: Optional[int] = None
    processed_by_run: Optional[int] = None


def enqueue_capture(conn: sqlite3.Connection, record: CaptureRecord) -> int:
    now = record.created_at or int(time.time())
    cur = conn.execute(
        "INSERT INTO capture_queue("
        "  session_id, project, user_message, assistant_message,"
        "  transcript_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            record.session_id,
            record.project,
            record.user_message,
            record.assistant_message,
            record.transcript_path,
            now,
        ),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def list_unprocessed_captures(conn: sqlite3.Connection) -> list[CaptureRecord]:
    rows = conn.execute(
        "SELECT * FROM capture_queue WHERE processed_at IS NULL ORDER BY created_at, queue_id"
    ).fetchall()
    return [_row_to_record(r) for r in rows]


def mark_captures_processed(
    conn: sqlite3.Connection, queue_ids: list[int], *, run_id: int
) -> None:
    if not queue_ids:
        return
    placeholders = ",".join("?" for _ in queue_ids)
    conn.execute(
        f"UPDATE capture_queue SET processed_at = ?, processed_by_run = ? "
        f"WHERE queue_id IN ({placeholders})",
        (int(time.time()), run_id, *queue_ids),
    )
    conn.commit()


def _row_to_record(row: sqlite3.Row) -> CaptureRecord:
    return CaptureRecord(
        queue_id=row["queue_id"],
        session_id=row["session_id"],
        project=row["project"],
        user_message=row["user_message"],
        assistant_message=row["assistant_message"],
        transcript_path=row["transcript_path"],
        created_at=row["created_at"],
        processed_at=row["processed_at"],
        processed_by_run=row["processed_by_run"],
    )
