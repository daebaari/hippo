"""Tests for dream_runs audit log."""
from __future__ import annotations

import sqlite3

import pytest

from hippo.storage.dream_runs import (
    complete_run,
    fail_run,
    get_recent_runs,
    start_run,
)
from hippo.storage.migrations import run_migrations


@pytest.fixture
def conn(sqlite_conn: sqlite3.Connection) -> sqlite3.Connection:
    run_migrations(sqlite_conn)
    return sqlite_conn


def test_start_returns_run_id_and_running_status(conn: sqlite3.Connection) -> None:
    run_id = start_run(conn, "heavy")
    assert run_id > 0
    runs = get_recent_runs(conn, limit=1)
    assert runs[0].status == "running"
    assert runs[0].type == "heavy"


def test_complete_run_records_stats(conn: sqlite3.Connection) -> None:
    run_id = start_run(conn, "heavy")
    complete_run(
        conn,
        run_id,
        atoms_created=3,
        heads_created=8,
        edges_created=5,
        contradictions_resolved=1,
    )
    runs = get_recent_runs(conn, limit=1)
    r = runs[0]
    assert r.status == "completed"
    assert r.atoms_created == 3
    assert r.heads_created == 8
    assert r.edges_created == 5
    assert r.contradictions_resolved == 1
    assert r.completed_at is not None


def test_fail_run_records_error(conn: sqlite3.Connection) -> None:
    run_id = start_run(conn, "light")
    fail_run(conn, run_id, error_message="LLM unreachable")
    runs = get_recent_runs(conn, limit=1)
    assert runs[0].status == "failed"
    assert runs[0].error_message == "LLM unreachable"


def test_get_recent_runs_orders_descending_by_started(conn: sqlite3.Connection) -> None:
    a = start_run(conn, "light")
    b = start_run(conn, "heavy")
    c = start_run(conn, "heavy")
    runs = get_recent_runs(conn, limit=3)
    assert [r.run_id for r in runs] == [c, b, a]


def test_complete_run_persists_bodies_archived_review(sqlite_conn):
    from hippo.storage.dream_runs import complete_run, get_recent_runs, start_run
    from hippo.storage.migrations import run_migrations

    run_migrations(sqlite_conn)
    run_id = start_run(sqlite_conn, "heavy")
    complete_run(sqlite_conn, run_id, bodies_archived_review=4)

    runs = get_recent_runs(sqlite_conn, limit=1)
    assert len(runs) == 1
    assert runs[0].bodies_archived_review == 4


def test_dream_run_record_exposes_progress_fields(conn):
    run_id = start_run(conn, "heavy")
    runs = get_recent_runs(conn, limit=1)
    r = runs[0]
    assert r.current_phase is None
    assert r.phase_done is None
    assert r.phase_total is None
    assert r.phase_started_at is None
    assert r.last_progress_at is None


def test_start_phase_sets_phase_columns_and_resets_done(conn):
    from hippo.storage.dream_runs import start_phase

    run_id = start_run(conn, "heavy")
    start_phase(conn, run_id, phase="atomize", total=12)

    runs = get_recent_runs(conn, limit=1)
    r = runs[0]
    assert r.current_phase == "atomize"
    assert r.phase_done == 0
    assert r.phase_total == 12
    assert r.phase_started_at is not None
    assert r.last_progress_at is not None


def test_update_progress_writes_done_and_timestamp(conn):
    from hippo.storage.dream_runs import start_phase, update_progress

    run_id = start_run(conn, "heavy")
    start_phase(conn, run_id, phase="edge_proposal", total=100)

    update_progress(conn, run_id, done=42)

    runs = get_recent_runs(conn, limit=1)
    r = runs[0]
    assert r.current_phase == "edge_proposal"
    assert r.phase_done == 42
    assert r.last_progress_at is not None
