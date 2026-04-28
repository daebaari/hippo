# hippo

Atomic memory system for Claude Code: per-turn retrieval, multi-head atoms, head-level typed graph edges, two-tier dream consolidation.

See `docs/architecture.md` for the full design.

## Status

**Milestone 7 of 8: bootstrap migration.** Storage, model daemon, retrieval pipeline, capture pipeline, light dream, heavy dream, and the bootstrap migration tool are complete. A `UserPromptSubmit` hook injects retrieved memories on every user turn; a `Stop` hook persists each completed turn to `capture_queue` and writes a turn-level embedding; a `PreCompact` hook runs the light dream, which mechanically (no LLM) creates one `session-meta:<session_id>` atom per unique session seen in the capture queue. The heavy dream — LLM-driven atomization of raw turns into atoms, multi-head expansion, embedding-clustered LLM-judged typed edges, contradiction resolution, and cleanup — runs nightly at 3am on AC power via launchd, or on demand via the `/hippo-dream` slash command. The model is Qwen 2.5 32B Instruct (4-bit MLX, ~18GB resident) loaded once per run and unloaded after.

## Quick start

```bash
# install
uv sync

# run tests
uv run pytest

# install daemon (launchd user agent), Claude Code hooks, and nightly dream
scripts/install-daemon.sh
scripts/install-hooks.sh
scripts/install-dream.sh

# inspect storage state
uv run memory-stats --project kaleon --json
```

### Daemon

The daemon holds embedder + reranker models resident in memory and exposes them
over a Unix socket so hooks don't pay the model load cost per invocation.
`scripts/install-daemon.sh` installs a launchd user agent that starts the
daemon at login and keeps it alive. Logs at `~/.claude/debug/memory-daemon.{log,err}`.

### Hooks

`scripts/install-hooks.sh` registers the `UserPromptSubmit`, `Stop`, and
`PreCompact` hooks in Claude Code's settings. The `UserPromptSubmit` hook
injects retrieved memory on each user turn; the `Stop` hook persists each
completed turn to `capture_queue` and writes a turn-level embedding so the
turn is retrievable immediately (until the dream loop atomizes it); the
`PreCompact` hook fires when the conversation is about to be compacted and
runs the light dream — a fast (<30s), no-LLM pass that creates one
`session-meta:<session_id>` body + head per unique session in the capture
queue, so session-level metadata is preserved before context is dropped.
The PreCompact hook coexists with any other PreCompact hook the user has
registered (e.g. a vanilla Claude Code dream); the install script only
manages its own entry. Tunables (scopes searched, top-k per stage, hop
limit, total cap, cluster cosine threshold) live in `src/hippo/config.py`.

### Heavy dream

The heavy dream is the LLM-driven consolidation pass. Per scope it:

1. atomizes each unprocessed session's transcript into bodies + heads,
2. expands heads on bodies that are retrieved often but have few search-affordances,
3. clusters active heads by embedding cosine similarity and asks the LLM to type each within-cluster pair (`causes`, `supersedes`, `contradicts`, `applies_when`, `related`),
4. resolves `contradicts` edges by asking the LLM to pick the current body and archiving the loser plus its heads,
5. marks captures processed and deletes their now-redundant turn embeddings.

`scripts/install-dream.sh` registers a launchd agent
(`com.<user>.dream-heavy`) that fires at 3am via
`/usr/bin/caffeinate -i bin/dream-heavy`. `bin/dream-heavy` itself does
an explicit `pmset` AC-power check and exits 0 if on battery (use
`--force` to override). Logs at `~/.claude/debug/dream-heavy.{log,err}`.

The `/hippo-dream` slash command (installed by `install-hooks.sh`,
namespaced to coexist with any vanilla `/dream`) runs `bin/dream-heavy
--force` interactively. The model loads in ~30-60s, then a typical run
takes a few minutes per scope depending on capture volume.

### Bootstrap migration

`bin/dream-bootstrap` atomizes pre-existing markdown memory files into
the new Hippo schema as a one-time migration. Each file is fed to the
atomize prompt with a scope hint derived from its filename prefix; the
LLM extracts atoms with title/body/heads and chooses `global` vs
`project:<name>` scope. After atomization, the bootstrap runs the
normal multi-head expansion, edge proposal, and contradiction
resolution phases over the bootstrapped corpus, then archives the
original files to `<memory_dir>/.legacy/<timestamp>/`.

```bash
bin/dream-bootstrap \
  --memory-dir ~/.claude/projects/<encoded-project-path>/memory \
  --project <project-name>
```

Acquires the `.heavy-lock` on both global and project stores for the
duration; aborts cleanly if either is held by another process.
Per-file progress is streamed to stdout. Idempotent: re-running picks
up any files left in the legacy directory.

> **Runtime note:** propose-edges scales as N² within each cosine
> cluster. On large corpora a single 22-head cluster expands to 231
> LLM-judged pairs; the local-LLM bootstrap of a ~30-file kaleon
> memory took ~110 minutes wall-time. A per-cluster pair cap is the
> obvious follow-up.

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
    llm.py               # Qwen 2.5 32B Instruct (4-bit MLX) wrapper
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
  dream/
    light.py             # PreCompact session-meta generator (no LLM)
    precompact_hook.py   # PreCompact handler (invokes light dream)
    atomize.py           # transcript -> bodies+heads via LLM
    multi_head.py        # head expansion for retrieved-often bodies
    cluster.py           # cosine-threshold single-link head clustering
    edge_proposal.py     # within-cluster LLM-typed edges
    contradiction.py     # LLM-confirmed contradiction resolution
    cleanup.py           # mark captures processed + drop turn embeddings
    heavy.py             # heavy dream orchestrator
    prompts/             # markdown templates for each LLM-driven phase
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
  precompact-light-dream # PreCompact hook entrypoint
  dream-heavy            # heavy dream entrypoint (AC-gated)
  dream-bootstrap        # one-shot legacy-files-to-atoms migration
commands/
  dream.md               # /hippo-dream slash command
hooks/
  userprompt-submit.sh   # shell wrapper invoked by Claude Code
  stop.sh                # shell wrapper invoked by Claude Code
  precompact.sh          # shell wrapper invoked by Claude Code
launchd/
  memory-daemon.plist.template
  dream-heavy.plist.template
scripts/
  install-daemon.sh      # installs launchd user agent
  install-hooks.sh       # registers Claude Code hooks + slash commands
  install-dream.sh       # installs nightly dream-heavy launchd agent
schema/
  001_initial.sql        # initial schema migration
tests/                   # mirrors src/ structure
```

## Next milestone

Install + wiring — top-level `scripts/install.sh` that orchestrates
deps sync, schema migrations, daemon launchd, dream-heavy launchd,
hooks, slash commands, and an optional bootstrap step in one shot.
Plus an end-to-end smoke and a `v0.1.0` tag.
