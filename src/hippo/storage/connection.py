"""SQLite connection factory with sqlite-vec extension auto-loaded."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec  # type: ignore[import-untyped]


def open_connection(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with sqlite-vec loaded and Row factory enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn
