"""Idempotent schema migration runner.

Migrations are numbered .sql files in <repo>/schema/. Runner reads each in
order, skips if already applied (per schema_versions table), then records
the version. Each migration is wrapped in a transaction.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

# Path to schema/ relative to this module (../../schema/)
_SCHEMA_DIR = Path(__file__).resolve().parents[3] / "schema"


def _ensure_versions_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_versions ("
        "  version INTEGER PRIMARY KEY,"
        "  applied_at INTEGER NOT NULL)"
    )


def current_version(conn: sqlite3.Connection) -> int:
    _ensure_versions_table(conn)
    row = conn.execute("SELECT MAX(version) AS v FROM schema_versions").fetchone()
    return int(row["v"]) if row["v"] is not None else 0


def _list_migrations() -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in sorted(_SCHEMA_DIR.glob("*.sql")):
        prefix = p.name.split("_", 1)[0]
        try:
            n = int(prefix)
        except ValueError:
            continue
        out.append((n, p))
    return out


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply all pending migrations idempotently."""
    _ensure_versions_table(conn)
    applied = current_version(conn)
    for version, path in _list_migrations():
        if version <= applied:
            continue
        sql = path.read_text()
        with conn:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_versions(version, applied_at) VALUES (?, ?)",
                (version, int(time.time())),
            )
