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
from hippo.scope_detect import resolve_project as _resolve_project
from hippo.storage.multi_store import Scope, resolve_memory_dir

# Re-export for callers that still import _resolve_project from this module
# (capture/stop_hook, cli/search, cli/get, dream/precompact_hook). Those will
# migrate to ``hippo.scope_detect`` directly in follow-up tasks of the
# CLI-scope-flag plan; this re-export keeps them working in the meantime.
__all__ = ["DaemonClientProto", "_resolve_project", "handle_userprompt_submit", "main"]


class DaemonClientProto(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def rerank(self, pairs: list[tuple[str, str]]) -> list[float]: ...


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
