"""Light dream: PreCompact session-level metadata generation.

Per scope, scan capture_queue for session_ids that don't yet have a
session-meta body. For each, write one body (mechanical summary) + one
head (mechanical summary as keyword sentence). No LLM.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from hippo.config import LIGHT_LOCK_FILENAME
from hippo.lock import LockHeldError, acquire_lock, release_lock
from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.body_files import BodyFile, write_body_file
from hippo.storage.dream_runs import complete_run, fail_run, start_run
from hippo.storage.heads import HeadRecord, insert_head
from hippo.storage.multi_store import Scope, open_store
from hippo.storage.vec import insert_head_embedding


class DaemonClientProto(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _mechanical_session_summary(
    session_id: str, project: str | None, captures: list[sqlite3.Row]
) -> tuple[str, str]:
    """Return (title, body_content)."""
    title = f"session-meta:{session_id}"
    n = len(captures)
    if not captures:
        return title, f"Session {session_id} (empty)."
    timestamps = [c["created_at"] for c in captures if c["created_at"] is not None]
    if timestamps:
        first = min(timestamps)
        last = max(timestamps)
        duration_min = max(0, (last - first) // 60)
    else:
        duration_min = 0
    sample_msg = (captures[0]["user_message"] or "")[:120]
    body = (
        f"Session: {session_id}\n"
        f"Project: {project or '(none)'}\n"
        f"Turn count: {n}\n"
        f"Duration: {duration_min} minutes\n"
        f"First user message excerpt: {sample_msg}\n"
    )
    return title, body


def run_light_dream(
    *, scope: Scope, daemon: DaemonClientProto
) -> dict[str, object]:
    """Execute the light dream for one scope. Returns a stats dict.

    Keys: ``sessions_summarized`` (int), ``run_id`` (int) on success;
    ``sessions_summarized`` (int) and ``skipped_locked`` (bool) when the
    lock is held by another live process.
    """
    store = open_store(scope)
    lock_path = store.memory_dir / LIGHT_LOCK_FILENAME
    try:
        handle = acquire_lock(lock_path)
    except LockHeldError:
        store.conn.close()
        return {"sessions_summarized": 0, "skipped_locked": True}

    run_id = start_run(store.conn, "light")
    try:
        # Find all session_ids in capture_queue
        rows = store.conn.execute(
            "SELECT DISTINCT session_id, project FROM capture_queue"
        ).fetchall()

        sessions_summarized = 0
        for row in rows:
            session_id = row["session_id"]
            project = row["project"]
            # Skip if a session-meta body already exists
            existing = store.conn.execute(
                "SELECT body_id FROM bodies WHERE title = ? AND archived = 0",
                (f"session-meta:{session_id}",),
            ).fetchone()
            if existing is not None:
                continue

            # Gather captures for this session
            cap_rows = store.conn.execute(
                "SELECT * FROM capture_queue WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            ).fetchall()

            title, body_content = _mechanical_session_summary(session_id, project, cap_rows)
            now = datetime.now(UTC)
            body_id = uuid4().hex

            write_body_file(
                store.memory_dir,
                BodyFile(
                    body_id=body_id,
                    title=title,
                    scope=scope.as_string(),
                    created=now,
                    updated=now,
                    content=body_content,
                ),
            )
            insert_body(
                store.conn,
                BodyRecord(
                    body_id=body_id,
                    file_path=f"bodies/{body_id}.md",
                    title=title,
                    scope=scope.as_string(),
                    source=f"light-dream-run:{run_id}",
                ),
            )
            head_id = uuid4().hex
            head_summary = (
                f"Session {session_id} ({project or '(none)'}, {len(cap_rows)} turns)"
            )
            insert_head(
                store.conn,
                HeadRecord(head_id=head_id, body_id=body_id, summary=head_summary),
            )
            vec = daemon.embed([head_summary])[0]
            insert_head_embedding(store.conn, head_id, vec)
            sessions_summarized += 1

        complete_run(
            store.conn,
            run_id,
            atoms_created=sessions_summarized,
            heads_created=sessions_summarized,
        )
        return {"sessions_summarized": sessions_summarized, "run_id": run_id}
    except Exception as e:
        fail_run(store.conn, run_id, error_message=str(e))
        raise
    finally:
        release_lock(handle)
        store.conn.close()
