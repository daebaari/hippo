"""Tests for schema migrations and connection helpers."""
from __future__ import annotations

import struct
from pathlib import Path

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
    assert current_version(conn) == 1


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
