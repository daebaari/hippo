"""Detect the current project from a working directory.

Walks up from cwd looking for a project boundary marker (`.git` directory,
`.git` worktree-pointer file, or `CLAUDE.md`). The basename of that
directory is the project name used by Hippo's scope system.

This module is shared between the Stop/UserPromptSubmit capture hooks and
the CLI commands so capture-side and CLI-side scope detection always agree.
"""
from __future__ import annotations

from pathlib import Path


def resolve_project(cwd: str) -> str | None:
    """Walk up from cwd; return the basename of the first dir that looks like
    a project root, or None if none found.

    Git worktrees: when ``.git`` is a file (a worktree pointer like
    ``gitdir: /path/to/main/.git/worktrees/<name>``), resolve to the MAIN
    repo's basename so captures from a worktree land in the same scope as
    the main checkout. Falls back to the worktree directory's own name if
    the pointer can't be parsed.
    """
    p = Path(cwd).resolve()
    for candidate in [p, *p.parents]:
        git_entry = candidate / ".git"
        if git_entry.is_dir() or (candidate / "CLAUDE.md").exists():
            return candidate.name
        if git_entry.is_file():
            main_repo = _read_worktree_pointer(git_entry)
            return main_repo.name if main_repo else candidate.name
    return None


def _read_worktree_pointer(git_file: Path) -> Path | None:
    """Parse a worktree's `.git` file. Returns the MAIN repo path, or None.

    Format (per `man gitrepository-layout`):
        gitdir: <absolute-or-relative-path-to>/.git/worktrees/<name>
    The main repo is the parent of the `/.git/worktrees/<name>` segment.
    """
    try:
        first_line = git_file.read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    prefix = "gitdir:"
    if not first_line.startswith(prefix):
        return None
    pointer = first_line[len(prefix):].strip()
    if not pointer:
        return None
    pointer_path = Path(pointer)
    if not pointer_path.is_absolute():
        pointer_path = (git_file.parent / pointer_path).resolve()
    # pointer_path is .../<main>/.git/worktrees/<name>
    parts = pointer_path.parts
    try:
        idx = len(parts) - 1 - list(reversed(parts)).index(".git")
    except ValueError:
        return None
    if idx < 1:
        return None
    return Path(*parts[:idx])
