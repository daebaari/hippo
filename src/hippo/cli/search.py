"""memory-search '<query>': run full retrieval pipeline; print the <memory> block."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from hippo.capture.userprompt_hook import _resolve_project
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

DEFAULTS = dict(
    vector_top_k_per_scope=RETRIEVAL_VECTOR_TOP_K_PER_SCOPE,
    hop_limit_per_seed=RETRIEVAL_HOP_LIMIT_PER_SEED,
    total_cap=RETRIEVAL_TOTAL_CAP,
    rerank_top_k=RETRIEVAL_RERANK_TOP_K,
)


def memory_search_cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="memory-search")
    p.add_argument("query")
    p.add_argument("--project", action="append", default=[])
    p.add_argument("--socket", default=str(Path.home() / ".claude" / "memory-daemon.sock"))
    args = p.parse_args(argv)

    daemon = DaemonClient(socket_path=Path(args.socket))
    scopes = [Scope.global_()]
    if args.project:
        scopes.extend(Scope.project(proj) for proj in args.project)
    else:
        cwd_project = _resolve_project(os.getcwd())
        if cwd_project:
            scopes.append(Scope.project(cwd_project))
    pipeline = RetrievalPipeline(daemon=daemon, scopes=scopes, **DEFAULTS)
    result = pipeline.run(args.query)
    # body_resolver: figure out which scope's bodies/ dir to read from
    scope_to_dir = {scope.as_string(): resolve_memory_dir(scope) for scope in scopes}
    block = format_memory_block(
        result,
        body_resolver=lambda hit: load_body_preview(scope_to_dir[hit.scope], hit.head.body_id),
    )
    if not block:
        print("(no memory candidates)")
        return 0
    print(block)
    return 0


def main() -> int:
    return memory_search_cli(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
