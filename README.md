# hippo

Atomic memory system for Claude Code: per-turn retrieval, multi-head atoms, head-level typed graph edges, two-tier dream consolidation.

See `docs/architecture.md` for the full design.

## Status

**Milestone 3 of 8: per-turn retrieval pipeline.** Storage, model daemon, and the retrieval pipeline are complete. A `UserPromptSubmit` hook is now installed: on every user turn, the hook embeds the prompt via the daemon, runs vector search + 1-hop graph expansion across the global and per-project scopes, reranks the candidates, and injects the top results back into the conversation. Capture-side hooks and dream loops are not yet implemented.

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

`scripts/install-hooks.sh` registers the `UserPromptSubmit` hook in Claude
Code's settings so retrieval runs automatically on each user turn. Tunables
(scopes searched, top-k per stage, hop limit, total cap) live in
`src/hippo/config.py`.

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
    userprompt_hook.py   # UserPromptSubmit handler
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
  userprompt-retrieve    # hook entrypoint
hooks/
  userprompt-submit.sh   # shell wrapper invoked by Claude Code
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

Capture pipeline + `Stop` hook + turn embeddings.
