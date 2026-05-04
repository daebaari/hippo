"""Tests for hippo.cli.scope_args — shared scope-resolution helpers."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hippo.cli.scope_args import (
    CommandKind,
    add_scope_args,
    resolve_scopes,
)
from hippo.storage.multi_store import Scope


def _parse(argv: list[str], *, kind: CommandKind) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_scope_args(p, kind=kind)
    return p.parse_args(argv)


def test_explicit_scope_replaces_cwd_detection(tmp_path: Path) -> None:
    repo = tmp_path / "myrepo"
    (repo / ".git").mkdir(parents=True)
    args = _parse(["--scope", "explicit"], kind="cross_read")
    scopes = resolve_scopes(args, kind="cross_read", cwd=str(repo))
    assert scopes == [Scope.project("explicit")]


def test_explicit_scope_repeatable(tmp_path: Path) -> None:
    args = _parse(["--scope", "a", "--scope", "b"], kind="cross_read")
    scopes = resolve_scopes(args, kind="cross_read", cwd=str(tmp_path))
    assert scopes == [Scope.project("a"), Scope.project("b")]


def test_explicit_scope_global_value(tmp_path: Path) -> None:
    args = _parse(["--scope", "global"], kind="scoped_write")
    scopes = resolve_scopes(args, kind="scoped_write", cwd=str(tmp_path))
    assert scopes == [Scope.global_()]


def test_scoped_write_in_project_returns_project_only(tmp_path: Path) -> None:
    repo = tmp_path / "hippo"
    (repo / ".git").mkdir(parents=True)
    args = _parse([], kind="scoped_write")
    scopes = resolve_scopes(args, kind="scoped_write", cwd=str(repo))
    assert scopes == [Scope.project("hippo")]


def test_cross_read_in_project_returns_global_plus_project(tmp_path: Path) -> None:
    repo = tmp_path / "hippo"
    (repo / ".git").mkdir(parents=True)
    args = _parse([], kind="cross_read")
    scopes = resolve_scopes(args, kind="cross_read", cwd=str(repo))
    assert scopes == [Scope.global_(), Scope.project("hippo")]


def test_targeted_in_project_returns_global_plus_project(tmp_path: Path) -> None:
    repo = tmp_path / "hippo"
    (repo / ".git").mkdir(parents=True)
    args = _parse([], kind="targeted")
    scopes = resolve_scopes(args, kind="targeted", cwd=str(repo))
    assert scopes == [Scope.global_(), Scope.project("hippo")]


def test_no_project_no_flag_errors(tmp_path: Path) -> None:
    bare = tmp_path / "nothing"
    bare.mkdir()
    args = _parse([], kind="scoped_write")
    with pytest.raises(SystemExit):
        resolve_scopes(args, kind="scoped_write", cwd=str(bare))


def test_no_project_no_flag_errors_for_cross_read(tmp_path: Path) -> None:
    bare = tmp_path / "nothing"
    bare.mkdir()
    args = _parse([], kind="cross_read")
    with pytest.raises(SystemExit):
        resolve_scopes(args, kind="cross_read", cwd=str(bare))


def test_all_scopes_enumerates_global_and_every_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects_root = tmp_path / "projects"
    for name in ["alpha", "beta"]:
        memdir = projects_root / name / "memory"
        memdir.mkdir(parents=True)
        (memdir / "memory.db").touch()
    monkeypatch.setattr("hippo.cli.scope_args.PROJECTS_ROOT", projects_root)

    args = _parse(["--all-scopes"], kind="cross_read")
    scopes = resolve_scopes(args, kind="cross_read", cwd=str(tmp_path))
    assert scopes == [
        Scope.global_(),
        Scope.project("alpha"),
        Scope.project("beta"),
    ]


def test_all_scopes_works_outside_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setattr("hippo.cli.scope_args.PROJECTS_ROOT", projects_root)
    bare = tmp_path / "nothing"
    bare.mkdir()
    args = _parse(["--all-scopes"], kind="scoped_write")
    # Should NOT error even though cwd has no project.
    scopes = resolve_scopes(args, kind="scoped_write", cwd=str(bare))
    assert scopes == [Scope.global_()]


def test_single_scope_write_rejects_multi_scope() -> None:
    p = argparse.ArgumentParser()
    add_scope_args(p, kind="single_scope_write")
    with pytest.raises(SystemExit):
        p.parse_args(["--scope", "a", "--scope", "b"])


def test_single_scope_write_rejects_all_scopes() -> None:
    p = argparse.ArgumentParser()
    add_scope_args(p, kind="single_scope_write")
    # add_scope_args must NOT register --all-scopes for single_scope_write.
    with pytest.raises(SystemExit):
        p.parse_args(["--all-scopes"])


def test_single_scope_write_accepts_one_scope(tmp_path: Path) -> None:
    args = _parse(["--scope", "foo"], kind="single_scope_write")
    scopes = resolve_scopes(args, kind="single_scope_write", cwd=str(tmp_path))
    assert scopes == [Scope.project("foo")]


def test_single_scope_write_auto_detects_in_project(tmp_path: Path) -> None:
    repo = tmp_path / "myproj"
    (repo / ".git").mkdir(parents=True)
    args = _parse([], kind="single_scope_write")
    scopes = resolve_scopes(args, kind="single_scope_write", cwd=str(repo))
    assert scopes == [Scope.project("myproj")]
