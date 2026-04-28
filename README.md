# hippo

Atomic memory system for Claude Code: per-turn retrieval, multi-head atoms, head-level typed graph edges, two-tier dream consolidation.

See `docs/architecture.md` for the full design.

## Status

**Milestone 1 of 8: storage layer.** Schema, CRUD modules, sqlite-vec integration, and CLI smoke-test are complete and tested. Models, hooks, daemon, and dream loops are not yet implemented.

## Quick start

```bash
# install
uv sync

# run tests
uv run pytest

# inspect storage state
uv run memory-stats --project kaleon --json
```

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
  cli/stats.py           # memory-stats command
schema/
  001_initial.sql        # initial schema migration
tests/                   # mirrors src/ structure
```

## Next milestone

Model daemon (mxbai-embed-large + mxbai-rerank-large via MLX, Unix-socket server).
