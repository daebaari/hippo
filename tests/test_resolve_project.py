"""Tests for _resolve_project, including the git-worktree case."""
from __future__ import annotations

from pathlib import Path

from hippo.capture.userprompt_hook import _resolve_project


def test_regular_git_repo(tmp_path: Path) -> None:
    repo = tmp_path / "myrepo"
    repo.mkdir()
    (repo / ".git").mkdir()
    sub = repo / "src" / "lib"
    sub.mkdir(parents=True)
    assert _resolve_project(str(sub)) == "myrepo"


def test_claude_md_marker(tmp_path: Path) -> None:
    repo = tmp_path / "noteproject"
    repo.mkdir()
    (repo / "CLAUDE.md").write_text("# notes\n")
    assert _resolve_project(str(repo)) == "noteproject"


def test_no_marker_returns_none(tmp_path: Path) -> None:
    bare = tmp_path / "nothing"
    bare.mkdir()
    assert _resolve_project(str(bare)) is None


def test_git_worktree_resolves_to_main_repo_basename(tmp_path: Path) -> None:
    """A git worktree's `.git` is a FILE pointing back to the main repo's
    `.git/worktrees/<name>`. The hook must resolve to the MAIN repo's
    basename, not the worktree directory's name.
    """
    main_repo = tmp_path / "myrepo"
    main_repo.mkdir()
    (main_repo / ".git").mkdir()
    worktree_dir = main_repo / ".worktrees" / "feature-branch"
    worktree_dir.mkdir(parents=True)
    (worktree_dir / ".git").write_text(
        f"gitdir: {main_repo}/.git/worktrees/feature-branch\n"
    )
    sub = worktree_dir / "src"
    sub.mkdir()
    assert _resolve_project(str(sub)) == "myrepo"


def test_worktree_pointer_with_relative_path(tmp_path: Path) -> None:
    """Some Git versions write absolute paths; some write relative.
    Make sure relative paths still resolve correctly (rare but possible).
    """
    main_repo = tmp_path / "myrepo"
    main_repo.mkdir()
    (main_repo / ".git").mkdir()
    worktree_dir = main_repo / ".worktrees" / "feature"
    worktree_dir.mkdir(parents=True)
    (worktree_dir / ".git").write_text(
        "gitdir: ../../.git/worktrees/feature\n"
    )
    assert _resolve_project(str(worktree_dir)) == "myrepo"


def test_worktree_with_malformed_pointer_falls_back(tmp_path: Path) -> None:
    """If we can't parse the pointer, fall back to the worktree's own basename
    rather than crashing or returning None.
    """
    main_repo = tmp_path / "myrepo"
    main_repo.mkdir()
    worktree_dir = main_repo / ".worktrees" / "broken"
    worktree_dir.mkdir(parents=True)
    (worktree_dir / ".git").write_text("totally not a gitdir line\n")
    assert _resolve_project(str(worktree_dir)) == "broken"
