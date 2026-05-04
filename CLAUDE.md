# Hippo — Claude Code Instructions

Atomic memory system for Claude Code. Stop hook captures every assistant turn,
nightly heavy dream atomizes captures into bodies + multi-head + typed edges,
UserPromptSubmit hook injects retrieval results on every user prompt.

See `README.md` for full architecture and `docs/operations.md` for day-to-day
operations. The canonical design spec lives at
`~/.claude/docs/specs/2026-04-28-atomic-memory-system-design.md` and is the
source of truth when this repo's docs and the spec disagree.

## Running services on this machine

- `com.<user>.memory-daemon` (launchd user agent, RunAtLoad + KeepAlive) —
  embedder + reranker resident, ~1.6GB RAM, listens on
  `~/.claude/memory-daemon.sock`. Manage via `launchctl kickstart -k`.
- `com.<user>.dream-heavy` (launchd user agent, StartCalendarInterval 3am AC-only) —
  fires nightly. Loads Qwen 2.5 32B (~18GB) only at run time.
- 3 hooks registered in `~/.claude/settings.json` under `hippo-*` labels
  (UserPromptSubmit, Stop, PreCompact). Coexist with any pre-existing hooks.
- `autoMemoryEnabled` and `autoDreamEnabled` are **off** in user settings —
  Hippo replaces both. Do not re-enable without checking with the user.
- LLM backend: switch with `/hippo-backend`; default qwen; persisted in
  `~/.claude/hippo-config.toml`. Gemini also reads
  `~/.claude/hippo-secrets` (mode 600) for the API key. See
  `src/hippo/config.py` and `src/hippo/models/llm.py` for current values.

## Stack pattern (important)

The launchd daemon plist and dream-heavy plist call uv with an absolute path
detected at install time, then `uv run --directory <repo>` to resolve the
project venv. Bare `bin/foo` invocations from launchd would fail because
launchd's PATH excludes `~/.local/bin`. CLI entry points (`bin/dream-heavy`,
`bin/dream-bootstrap`, `bin/userprompt-retrieve`, etc.) are bash shims that
also `exec uv run --project <repo>`; they only work for direct invocation
when `uv` is on the caller's PATH.

When adding a new launchd job or new bin entry point, follow this pattern.
Don't use `#!/usr/bin/env python3` shebangs — they resolve to system Python
without the project deps and crash on `import frontmatter` etc.

## Hooks: payload shape

Real Claude Code hook envelopes differ from naive expectations:

- **UserPromptSubmit** sends `{prompt, cwd, session_id, ...}` — `prompt`, not
  `user_message`.
- **Stop** sends `{session_id, transcript_path, cwd, last_assistant_message,
  hook_event_name="Stop", stop_hook_active, permission_mode}` — no
  `user_message` in payload; the user's message lives in the JSONL transcript
  at `transcript_path`. Hippo's Stop hook walks the JSONL backwards to find
  the most recent `type=user` entry.

Project detection (see `src/hippo/scope_detect.py`) walks up from cwd to
find a `.git` or `CLAUDE.md` and returns the basename as the project name.
That basename becomes the scope for capture / retrieval. CLI commands
auto-detect the same way, with `--scope <name>` as an explicit override
and `--all-scopes` for ops/cron use.

## Known gaps (real, affect usability)

- **Raw-turn retrieval is not wired into the pipeline.** Stop hook embeds
  every turn into `turn_embeddings_vec`, but `vector_search.py` only queries
  `head_embeddings`. Net effect: new captures are invisible to retrieval
  until heavy dream atomizes them. See `KNOWN_ISSUES.md`.
- **Edge-proposal N² scaling.** `propose_edges` evaluates every within-cluster
  pair via LLM. Largest cosine cluster of 22 heads → 231 LLM calls → minutes.
  See `KNOWN_ISSUES.md`.
- **Atomize-prompt noise.** The atomize prompt instructs the LLM to skip
  in-the-moment chatter, but a few noise atoms still leak through per dream
  run. Soft-archive via `bin/memory-archive <head_id> --reason "noise"` when
  spotted.

## Testing & gates

```bash
uv run pytest          # 110+ tests, 2 skipped (LLM tests gated on RUN_LLM_TESTS=1)
uv run ruff check src tests
uv run mypy src        # strict mode, must be clean
```

LLM-real test (slow, downloads + loads ~18GB):
```bash
RUN_LLM_TESTS=1 uv run pytest tests/test_llm.py -v
```

## When making changes

- Storage layer is the foundation; schema migrations live in `schema/*.sql`
  and apply idempotently on every connection open. Don't break backward
  compat with existing on-disk DBs.
- Hooks must NEVER block the user's session. The handler `main()` functions
  catch all exceptions and exit 0. If you raise from inside, the hook fails
  silently — debug by adding a `printf "$input" >> debug.log` in the bash
  shim and capture the actual envelope.
- Heavy dream's lock semantics: acquire `.heavy-lock` per scope, release in
  `finally`. Same for `.light-lock` for light dream. The
  `cleanup-stale-consolidate-locks.sh` SessionStart hook (in `~/.claude/`)
  sweeps dead-PID and >1h-old locks.
