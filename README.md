# hippo

Atomic memory system for Claude Code: per-turn retrieval, multi-head atoms, head-level typed graph edges, two-tier dream consolidation.

See `docs/architecture.md` for the full design.

## Status

**Milestone 4 of 8: capture pipeline.** Storage, model daemon, retrieval pipeline, and capture pipeline are complete. A `UserPromptSubmit` hook injects retrieved memories on every user turn, and a `Stop` hook persists every completed turn to `capture_queue` and writes a turn-level embedding for immediate retrievability. Dream loops (consolidation of captures into atomic memories) are not yet implemented.

## Quick start

```bash
# install
uv sync

# run tests
uv run pytest

# install daemon (launchd user agent) and Claude Code hooks
scripts/install-daemon.sh
scripts/install-hooks.sh

# inspect storage state
uv run memory-stats --project kaleon --json
```

### Daemon

The daemon holds embedder + reranker models resident in memory and exposes them
over a Unix socket so hooks don't pay the model load cost per invocation.
`scripts/install-daemon.sh` installs a launchd user agent that starts the
daemon at login and keeps it alive. Logs at `~/.claude/debug/memory-daemon.{log,err}`.

### Hooks

`scripts/install-hooks.sh` registers the `UserPromptSubmit` and `Stop` hooks
in Claude Code's settings. The `UserPromptSubmit` hook injects retrieved
memory on each user turn; the `Stop` hook persists each completed turn to
`capture_queue` and writes a turn-level embedding so the turn is retrievable
immediately (until the dream loop atomizes it). Tunables (scopes searched,
top-k per stage, hop limit, total cap) live in `src/hippo/config.py`.

## Layout

```
src/hippo/
  config.py              # paths, dimensions, edge relations, retrieval tunables
  lock.py                # file-based lock with stale recovery
  storage/
    connection.py        # sqlite + sqlite-vec
    migrations.py        # idempotent runner
    body_files.py        # markdown + frontmatter I/O
    bodies.py            # bodies table CRUD
    heads.py             # heads table CRUD
    vec.py               # head_embeddings vector ops
    edges.py             # edges table CRUD
    capture.py           # capture_queue CRUD
    turn_embeddings.py   # turn-level vector store
    dream_runs.py        # audit log
    multi_store.py       # scope resolver, lazy DB creation
  models/
    embedder.py          # mxbai-embed-large wrapper
    reranker.py          # mxbai-rerank-large wrapper
  daemon/
    protocol.py          # newline-delimited JSON request/response
    server.py            # Unix-socket server, model lifecycle
    client.py            # sync client for hooks
  retrieval/
    vector_search.py     # dual-DB ANN over head embeddings
    graph_expand.py      # 1-hop graph expansion via edges
    rerank.py            # cross-encoder rerank with edge-type boosts
    pipeline.py          # orchestrator (embed -> search -> expand -> rerank)
    inject.py            # render results for hook injection
  capture/
    userprompt_hook.py   # UserPromptSubmit handler (retrieval injection)
    stop_hook.py         # Stop handler (capture + turn embedding)
  cli/
    stats.py             # memory-stats
    get.py               # memory-get (fetch a head/body by id)
    search.py            # memory-search (ad-hoc retrieval)
    archive.py           # memory-archive (mark heads archived)
bin/
  daemon                 # daemon entrypoint
  memory-get             # CLI shim
  memory-search          # CLI shim
  memory-archive         # CLI shim
  userprompt-retrieve    # UserPromptSubmit hook entrypoint
  stop-capture           # Stop hook entrypoint
hooks/
  userprompt-submit.sh   # shell wrapper invoked by Claude Code
  stop.sh                # shell wrapper invoked by Claude Code
launchd/
  memory-daemon.plist.template
scripts/
  install-daemon.sh      # installs launchd user agent
  install-hooks.sh       # registers Claude Code hooks
schema/
  001_initial.sql        # initial schema migration
tests/                   # mirrors src/ structure
```

## Next milestone

Light dream / `PreCompact` handler — atomize raw turns into per-head memories.
