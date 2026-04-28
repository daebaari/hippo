"""Tests for multi-store resolution and lazy DB creation."""
from __future__ import annotations

from pathlib import Path

import pytest

from hippo.storage.multi_store import (
    Scope,
    open_store,
    resolve_memory_dir,
)


def test_resolve_global_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    expected = tmp_path / "global"
    assert resolve_memory_dir(Scope.global_()) == expected


def test_resolve_project_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hippo.config.PROJECTS_ROOT", tmp_path / "projects")
    expected = tmp_path / "projects" / "kaleon" / "memory"
    assert resolve_memory_dir(Scope.project("kaleon")) == expected


def test_open_store_creates_dir_db_and_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    # DB file created
    assert (tmp_path / "global" / "memory.db").exists()
    # bodies/ created
    assert (tmp_path / "global" / "bodies").exists()
    # Schema applied — bodies table exists
    row = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='bodies'"
    ).fetchone()
    assert row is not None
    store.conn.close()


def test_open_store_idempotent_on_existing_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store_a = open_store(Scope.global_())
    store_a.conn.close()
    # Second open must not error or duplicate-apply
    store_b = open_store(Scope.global_())
    store_b.conn.close()


def test_scope_string_representation() -> None:
    assert Scope.global_().as_string() == "global"
    assert Scope.project("kaleon").as_string() == "project:kaleon"
