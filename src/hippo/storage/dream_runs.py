"""CRUD for dream_runs audit log."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Literal

DreamType = Literal["light", "heavy"]
DreamStatus = Literal["running", "completed", "failed"]


@dataclass
class DreamRunRecord:
    run_id: int
    type: str
    started_at: int
    completed_at: int | None
    status: str
    atoms_created: int
    heads_created: int
    edges_created: int
    contradictions_resolved: int
    bodies_archived_review: int
    error_message: str | None
    current_phase: str | None
    phase_done: int | None
    phase_total: int | None
    phase_started_at: int | None
    last_progress_at: int | None


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
    bodies_archived_review: int = 0,
) -> None:
    conn.execute(
        "UPDATE dream_runs SET status = 'completed', completed_at = ?, "
        "atoms_created = ?, heads_created = ?, edges_created = ?, "
        "contradictions_resolved = ?, bodies_archived_review = ? "
        "WHERE run_id = ?",
        (
            int(time.time()),
            atoms_created,
            heads_created,
            edges_created,
            contradictions_resolved,
            bodies_archived_review,
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


def start_phase(
    conn: sqlite3.Connection, run_id: int, *, phase: str, total: int
) -> None:
    """Record the start of a new phase. Resets phase_done to 0."""
    now = int(time.time())
    conn.execute(
        "UPDATE dream_runs SET current_phase = ?, phase_done = 0, "
        "phase_total = ?, phase_started_at = ?, last_progress_at = ? "
        "WHERE run_id = ?",
        (phase, total, now, now, run_id),
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
            bodies_archived_review=int(r["bodies_archived_review"] or 0),
            error_message=r["error_message"],
            current_phase=r["current_phase"],
            phase_done=r["phase_done"],
            phase_total=r["phase_total"],
            phase_started_at=r["phase_started_at"],
            last_progress_at=r["last_progress_at"],
        )
        for r in rows
    ]
