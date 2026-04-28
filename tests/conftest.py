"""Shared pytest fixtures."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import sqlite_vec


@pytest.fixture
def temp_memory_dir(tmp_path: Path) -> Path:
    """Empty memory dir suitable for one store (global or project)."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "bodies").mkdir()
    return memory_dir


@pytest.fixture
def sqlite_conn(temp_memory_dir: Path) -> sqlite3.Connection:
    """Raw sqlite3 connection with sqlite-vec loaded."""
    conn = sqlite3.connect(temp_memory_dir / "memory.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn
