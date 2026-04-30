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


def update_progress(conn: sqlite3.Connection, run_id: int, *, done: int) -> None:
    """Update phase_done and last_progress_at for the given run."""
    conn.execute(
        "UPDATE dream_runs SET phase_done = ?, last_progress_at = ? "
        "WHERE run_id = ?",
        (done, int(time.time()), run_id),
    )
    conn.commit()


def get_running_run(conn: sqlite3.Connection) -> DreamRunRecord | None:
    """Return the single in-progress run (status='running') or None.

    A row is considered live only if its most recent timestamp (started_at
    or last_progress_at) is within STALE_RUN_AGE_SECONDS. Orphaned rows from
    killed processes are filtered out so callers don't see ghost dreams.
    """
    from hippo.config import STALE_RUN_AGE_SECONDS

    cutoff = int(time.time()) - STALE_RUN_AGE_SECONDS
    rows = conn.execute(
        "SELECT * FROM dream_runs WHERE status = 'running' "
        "AND (last_progress_at >= ? OR (last_progress_at IS NULL AND started_at >= ?)) "
        "ORDER BY started_at DESC, run_id DESC LIMIT 1",
        (cutoff, cutoff),
    ).fetchall()
    if not rows:
        return None
    r = rows[0]
    return DreamRunRecord(
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


def mark_orphan_runs_failed(conn: sqlite3.Connection) -> int:
    """Flip any status='running' rows to 'failed' with reason='abandoned'.

    Intended to be called once at the start of every heavy dream, AFTER the
    per-scope lock has been acquired — at that point the lock guarantees no
    other live process is writing to this DB, so any 'running' row is by
    definition the artifact of a previously-killed process.

    Returns the number of rows updated.
    """
    cur = conn.execute(
        "UPDATE dream_runs SET status = 'failed', "
        "completed_at = ?, error_message = ? "
        "WHERE status = 'running'",
        (int(time.time()), "abandoned: process exited without completing"),
    )
    conn.commit()
    return int(cur.rowcount or 0)


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
