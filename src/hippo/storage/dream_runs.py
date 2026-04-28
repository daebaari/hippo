"""CRUD for dream_runs audit log."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Literal, Optional

DreamType = Literal["light", "heavy"]
DreamStatus = Literal["running", "completed", "failed"]


@dataclass
class DreamRunRecord:
    run_id: int
    type: str
    started_at: int
    completed_at: Optional[int]
    status: str
    atoms_created: int
    heads_created: int
    edges_created: int
    contradictions_resolved: int
    error_message: Optional[str]


def start_run(conn: sqlite3.Connection, dream_type: DreamType) -> int:
    cur = conn.execute(
        "INSERT INTO dream_runs(type, started_at) VALUES (?, ?)",
        (dream_type, int(time.time())),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def complete_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    atoms_created: int = 0,
    heads_created: int = 0,
    edges_created: int = 0,
    contradictions_resolved: int = 0,
) -> None:
    conn.execute(
        "UPDATE dream_runs SET status = 'completed', completed_at = ?, "
        "atoms_created = ?, heads_created = ?, edges_created = ?, "
        "contradictions_resolved = ? WHERE run_id = ?",
        (
            int(time.time()),
            atoms_created,
            heads_created,
            edges_created,
            contradictions_resolved,
            run_id,
        ),
    )
    conn.commit()


def fail_run(conn: sqlite3.Connection, run_id: int, *, error_message: str) -> None:
    conn.execute(
        "UPDATE dream_runs SET status = 'failed', completed_at = ?, error_message = ? "
        "WHERE run_id = ?",
        (int(time.time()), error_message, run_id),
    )
    conn.commit()


def get_recent_runs(conn: sqlite3.Connection, limit: int) -> list[DreamRunRecord]:
    rows = conn.execute(
        "SELECT * FROM dream_runs ORDER BY started_at DESC, run_id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        DreamRunRecord(
            run_id=int(r["run_id"]),
            type=r["type"],
            started_at=int(r["started_at"]),
            completed_at=r["completed_at"],
            status=r["status"],
            atoms_created=int(r["atoms_created"] or 0),
            heads_created=int(r["heads_created"] or 0),
            edges_created=int(r["edges_created"] or 0),
            contradictions_resolved=int(r["contradictions_resolved"] or 0),
            error_message=r["error_message"],
        )
        for r in rows
    ]
