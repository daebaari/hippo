# hippo

Atomic memory system for Claude Code: per-turn retrieval, multi-head atoms, head-level typed graph edges, two-tier dream consolidation.

See `docs/architecture.md` for the full design.

## Status

**Milestone 2 of 8: model daemon.** Storage layer and model daemon are complete. The daemon holds the embedder and reranker resident and serves them over a Unix socket via a sync client. Hooks and dream loops are not yet implemented.

## Quick start

```bash
# install
uv sync

# run tests
uv run pytest

# inspect storage state
uv run memory-stats --project kaleon --json
```

### Daemon (optional, for hook integration)

The daemon holds embedder + reranker models resident in memory and exposes them
over a Unix socket so hooks don't pay the model load cost per invocation.

    scripts/install-daemon.sh

This installs a launchd user agent that starts the daemon at login and keeps
it alive. Logs at `~/.claude/debug/memory-daemon.{log,err}`.

## Layout

```
src/hippo/
  config.py              # paths, dimensions, edge relations
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
  cli/stats.py           # memory-stats command
bin/
  daemon                 # daemon entrypoint
launchd/
  memory-daemon.plist.template
scripts/
  install-daemon.sh      # installs launchd user agent
schema/
  001_initial.sql        # initial schema migration
tests/                   # mirrors src/ structure
```

## Next milestone

Per-turn retrieval pipeline.
