"""UserPromptSubmit hook handler.

Reads JSON envelope (Claude Code's UserPromptSubmit format) from stdin,
runs the retrieval pipeline against global + project scopes, and emits a
``<memory>`` block on stdout that Claude Code injects into the next turn.

The ``main()`` entry point swallows all exceptions and returns 0 — this
hook must NEVER block the user's prompt. Errors are logged to stderr.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Protocol

from hippo.config import (
    RETRIEVAL_HOP_LIMIT_PER_SEED,
    RETRIEVAL_RERANK_TOP_K,
    RETRIEVAL_TOTAL_CAP,
    RETRIEVAL_VECTOR_TOP_K_PER_SCOPE,
)
from hippo.daemon.client import DaemonClient
from hippo.retrieval.inject import format_memory_block, load_body_preview
from hippo.retrieval.pipeline import RetrievalPipeline
from hippo.storage.multi_store import Scope, resolve_memory_dir


class DaemonClientProto(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def rerank(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def _resolve_project(cwd: str) -> str | None:
    """Walk up from cwd; return the basename of the first dir that looks like
    a project root.

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


def handle_userprompt_submit(
    *, stdin_text: str, daemon: DaemonClientProto | None = None,
) -> str:
    payload = json.loads(stdin_text)
    user_message = payload.get("prompt", "")
    cwd = payload.get("cwd", os.getcwd())
    if not user_message.strip():
        return ""

    if daemon is None:
        daemon = DaemonClient(socket_path=Path.home() / ".claude" / "memory-daemon.sock")

    scopes = [Scope.global_()]
    project = _resolve_project(cwd)
    if project:
        scopes.append(Scope.project(project))

    pipeline = RetrievalPipeline(
        daemon=daemon,
        scopes=scopes,
        vector_top_k_per_scope=RETRIEVAL_VECTOR_TOP_K_PER_SCOPE,
        hop_limit_per_seed=RETRIEVAL_HOP_LIMIT_PER_SEED,
        total_cap=RETRIEVAL_TOTAL_CAP,
        rerank_top_k=RETRIEVAL_RERANK_TOP_K,
    )
    result = pipeline.run(user_message)
    scope_to_dir = {scope.as_string(): resolve_memory_dir(scope) for scope in scopes}
    return format_memory_block(
        result,
        body_resolver=lambda hit: load_body_preview(
            scope_to_dir.get(hit.scope, scope_to_dir["global"]), hit.head.body_id,
        ),
    )


def main() -> int:
    text = sys.stdin.read()
    try:
        out = handle_userprompt_submit(stdin_text=text)
        if out:
            sys.stdout.write(out)
        return 0
    except Exception as e:
        sys.stderr.write(f"hippo userprompt-hook error: {e}\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
