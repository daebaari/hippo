"""Scope-aware memory store resolution and lazy DB creation.

A "store" is one (memory_dir, sqlite connection) pair. There are two
scope kinds:
  - global: ~/.claude/memory/
  - project:<name>: ~/.claude/projects/<name>/memory/

Project DBs are lazy-created on first access. open_store() handles the
"create dir if needed, create DB if needed, run migrations" sequence so
callers never have to think about it.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hippo import config
from hippo.config import DB_FILENAME, BODIES_SUBDIR
from hippo.storage.connection import open_connection
from hippo.storage.migrations import run_migrations


@dataclass(frozen=True)
class Scope:
    kind: str               # "global" or "project"
    project_name: Optional[str] = None

    @staticmethod
    def global_() -> "Scope":
        return Scope(kind="global")

    @staticmethod
    def project(name: str) -> "Scope":
        return Scope(kind="project", project_name=name)

    def as_string(self) -> str:
        if self.kind == "global":
            return "global"
        return f"project:{self.project_name}"


@dataclass
class Store:
    scope: Scope
    memory_dir: Path
    conn: sqlite3.Connection


def resolve_memory_dir(scope: Scope) -> Path:
    if scope.kind == "global":
        return config.GLOBAL_MEMORY_DIR
    if scope.kind == "project":
        assert scope.project_name is not None
        return config.project_memory_dir(scope.project_name)
    raise ValueError(f"unknown scope kind: {scope.kind}")


def open_store(scope: Scope) -> Store:
    """Open (or lazy-create) the store for this scope. Runs migrations idempotently."""
    memory_dir = resolve_memory_dir(scope)
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / BODIES_SUBDIR).mkdir(parents=True, exist_ok=True)
    db_path = memory_dir / DB_FILENAME
    conn = open_connection(db_path)
    run_migrations(conn)
    return Store(scope=scope, memory_dir=memory_dir, conn=conn)
