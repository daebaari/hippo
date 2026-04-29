"""Tests for schema migrations and connection helpers."""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import sqlite_vec

from hippo.storage.connection import open_connection
from hippo.storage.migrations import current_version, run_migrations


def test_run_migrations_creates_all_tables(temp_memory_dir: Path) -> None:
    db_path = temp_memory_dir / "memory.db"
    conn = open_connection(db_path)
    run_migrations(conn)
    table_names = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    expected = {
        "bodies", "heads", "edges", "capture_queue",
        "turn_embeddings", "dream_runs", "schema_versions",
    }
    assert expected.issubset(table_names)


def test_run_migrations_is_idempotent(temp_memory_dir: Path) -> None:
    db_path = temp_memory_dir / "memory.db"
    conn = open_connection(db_path)
    run_migrations(conn)
    run_migrations(conn)  # second call must not raise
    assert current_version(conn) == 2


def test_sqlite_vec_extension_loaded(temp_memory_dir: Path) -> None:
    db_path = temp_memory_dir / "memory.db"
    conn = open_connection(db_path)
    # vec_version() comes from sqlite-vec
    row = conn.execute("SELECT vec_version() AS v").fetchone()
    assert row["v"] is not None


def test_head_embeddings_virtual_table_works(temp_memory_dir: Path) -> None:
    db_path = temp_memory_dir / "memory.db"
    conn = open_connection(db_path)
    run_migrations(conn)
    # Insert a dummy embedding (length-1024 vector required)
    vec = [0.1] * 1024
    conn.execute(
        "INSERT INTO head_embeddings(head_id, embedding) VALUES (?, ?)",
        ("h1", _to_vec_blob(vec)),
    )
    row = conn.execute(
        "SELECT head_id FROM head_embeddings WHERE head_id = 'h1'"
    ).fetchone()
    assert row["head_id"] == "h1"


def _to_vec_blob(vec: list[float]) -> bytes:
    """sqlite-vec accepts vectors as either JSON or float32 packed bytes."""
    return struct.pack(f"{len(vec)}f", *vec)


def test_migration_002_adds_prune_columns(tmp_path: Path) -> None:
    """002 adds bodies.last_reviewed_at and dream_runs.bodies_archived_review."""
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row

    run_migrations(conn)

    body_cols = {row["name"] for row in conn.execute("PRAGMA table_info(bodies)").fetchall()}
    assert "last_reviewed_at" in body_cols

    run_cols = {row["name"] for row in conn.execute("PRAGMA table_info(dream_runs)").fetchall()}
    assert "bodies_archived_review" in run_cols

    # Partial index exists
    idx = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_bodies_review_queue'"
    ).fetchone()
    assert idx is not None

    # Re-running is idempotent
    run_migrations(conn)
    conn.close()
