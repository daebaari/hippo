# LLM Backend Toggle — Design

Date: 2026-04-28
Status: Approved, awaiting implementation plan

## Problem

Hippo's heavy-dream pipeline is hardcoded to Qwen 2.5 32B via mlx-lm. A
prior attempt (`stash@{0}: gemini-backend-wip-from-plan7-overreach`) added
a Gemini backend behind an env var, but the integration was incomplete:

- Stash conflicts with the post-stash bash-shim refactor of
  `bin/dream-heavy` and `bin/dream-bootstrap` — won't apply cleanly.
- launchd's 3am job does not inherit user shell env, so the env-var toggle
  silently fails for the only path that actually matters in production.
- Hardcoded model id, narrow retry exception list, no client timeout,
  process-env mutation, no test coverage for the dispatch logic.

We want an easily toggleable backend selector (Qwen default, Gemini
optional) that works for both manual invocation and the nightly launchd
run, with no surgery on the plist or install script.

## Goals

1. User can switch backends with a single slash command, persisted across
   restarts and reboots.
2. Default is `qwen`; nothing changes for users who don't opt in.
3. Both `bin/dream-heavy` (manual) and the launchd 3am job pick up the
   user's choice without any env-var plumbing.
4. Manual runs fail loudly on misconfiguration; nightly runs degrade
   gracefully so a missed config doesn't mean a missed consolidation.
5. All review findings on the prior stash are resolved as part of this
   work.

## Non-goals

- Real-Gemini integration tests in the suite.
- CLI flag overrides (`--backend=gemini`); slash command is the only
  switch surface.
- Per-project or per-time-of-day backend profiles.
- Automatic key rotation, validation, refresh.
- GUI / TUI for config.
- Fixing the unrelated `KNOWN_ISSUES.md` items (N² edge proposal,
  raw-turn retrieval gap).

## Architecture

Three new persistent surfaces and one shared resolver:

```
~/.claude/hippo-config.toml          # editable, contains backend choice
~/.claude/hippo-secrets              # mode-600, KEY=value lines
~/.claude/commands/hippo-backend.md  # slash command, reads/writes the TOML
```

Resolution flow (every dream run, manual or 3am):

```
bin/dream-heavy (bash shim)
    └─ uv run python -m hippo.cli.dream_heavy --strict
         └─ select_llm(strict=True)
              ├─ load_config()  reads hippo-config.toml; defaults if absent
              ├─ if backend == "qwen":   LocalLLM.load()
              └─ if backend == "gemini":
                   ├─ load_api_key()  env → secrets file
                   ├─ if key present:   GeminiLLM.load(api_key, model_id, ...)
                   └─ else:
                        - strict=True  → raise ConfigError
                        - strict=False → warn + LocalLLM.load()
```

The launchd path (`hippo.cli.dream_heavy` invoked directly via plist) does
not pass `--strict`, so its `select_llm()` call defaults to `strict=False`
and silently falls back to qwen on missing key. The plist template is
untouched.

## Components

### `src/hippo/config.py` (new)

- `Config` dataclass: `backend`, `gemini_model_id`,
  `gemini_default_thinking_level`. All defaults live here as the single
  source of truth.
- `load_config() -> Config` — reads via stdlib `tomllib`. Missing file →
  defaults. Malformed TOML or unknown values → raise `ConfigError`.
- `write_config(cfg) -> None` — atomic (`tmp + os.replace`).
- `load_api_key() -> str | None` — env (`GOOGLE_API_KEY` →
  `GEMINI_API_KEY`) → secrets file. Parses `KEY=value` lines, ignores
  comments and blanks.
- `config_path() -> Path`, `secrets_path() -> Path` — overridable via
  `HIPPO_CONFIG_DIR` env var (test isolation).
- `ConfigError(RuntimeError)` — sentinel exception.

### `src/hippo/models/llm.py` (modified)

- `select_llm(*, strict: bool = False)` — uses `config.load_config()` and
  `config.load_api_key()`. Returns the right backend or raises
  `ConfigError` per the resolution flow.
- `GeminiLLM.load(*, api_key: str, model_id: str,
  default_thinking_level: str)` — explicit args, no `os.environ`
  mutation. Constructs `genai.Client(api_key=api_key)`.
- Retry loop catches `(errors.APIError, OSError, TimeoutError,
  httpx.RequestError)`, configures explicit per-request timeout. Drop the
  unreachable trailing raise.
- `LLMProto` lives here as the single canonical Protocol; the six
  duplicates in `src/hippo/dream/*.py` are deleted and import from here.

### `src/hippo/cli/dream_heavy.py` and `dream_bootstrap.py` (modified)

- Add `--strict` argparse flag (default `False`).
- Replace `LocalLLM.load()` with `select_llm(strict=args.strict)`.
- Wrap in `try/except ConfigError` to print clean error instead of a
  Python traceback.

### `bin/dream-heavy` and `bin/dream-bootstrap` (modified, bash shims)

- Append `--strict` to the args passed to the underlying
  `python -m hippo.cli.*` invocation. Direct user-facing invocation is
  always strict; launchd (which calls `python -m hippo.cli.dream_heavy`
  directly without going through the shim) stays non-strict.

### `src/hippo/cli/backend_toggle.py` (new)

Backs the slash command. Three subcommands via argparse:

- (no args) → status output: current backend, model id, key-detection
  result (env / secrets file / not found), config file path, secrets
  file path, recommended log file to tail.
- `qwen` → write `backend = "qwen"` to config, print confirmation.
- `gemini` → write `backend = "gemini"`, warn if no key detected.

### `~/.claude/commands/hippo-backend.md` (new)

Slash command shim. Runs
`uv run --project ~/code/hippo python -m hippo.cli.backend_toggle "$@"`.
The repo path is detected at install time (mirrors how
`scripts/install-dream.sh` already wires absolute uv paths into the plist).

### `pyproject.toml` (modified)

Move `google-genai>=1.0` from top-level `dependencies` to
`[project.optional-dependencies] gemini = [...]`. Default `uv sync` is
unchanged for users who only want the local backend.

`select_llm()` raises a friendly `ConfigError` ("install with
`uv sync --extra gemini`") if backend=gemini and the import fails.

## Failure modes

| Scenario | Result |
|---|---|
| Config file missing | Defaults (qwen). Silent. |
| Config file malformed TOML | `ConfigError`. Hard fail both paths. |
| Config has unknown backend | `ConfigError`. Hard fail both paths. |
| Gemini selected, no key, `strict=True` | `ConfigError`. Manual run exits 1. |
| Gemini selected, no key, `strict=False` | Warning to stderr/log. Falls back to qwen. Run continues. |
| Gemini selected, key present, network blip | Retried per `GeminiLLM.max_attempts` with exponential backoff. After exhaustion: heavy dream's outer handler logs the failure for that scope. |
| `google-genai` not installed | `ConfigError` advising the install command. Hard fail both paths (misinstall, not runtime). |
| Slash command bad args | argparse error to stderr; exit 2. |
| Slash command can't write config | Raise `PermissionError` with the actual path. |

The launchd-path warning (gemini → qwen fallback) writes to
`~/.claude/debug/dream-heavy.err` (already wired by the existing plist).
The slash command's status output names this file so users know where to
look.

## Testing

### New test files

`tests/test_config.py` — config and key-loading, pure stdlib, no LLM.
Round-trip, defaults, malformed TOML, unknown backend, env-vs-file
precedence, missing-key behaviour.

`tests/test_llm_select.py` — dispatch logic with `LocalLLM.load` and
`GeminiLLM.load` mocked. Asserts the right class is returned and called
with the right args under each branch.

`tests/test_backend_toggle.py` — slash-command CLI module. Status output
shape, write-on-switch, warning-on-missing-key, argparse errors.

### Modified test files

Three existing tests (`test_contradiction.py`, `test_edge_proposal.py`,
one of `test_atomize.py`/`test_multi_head.py`) gain one assertion each
capturing `thinking_level` in the fake's call log, to lock in that the
short-call sites pass `"minimal"` and the long sites don't.

### Skipped

No real-Gemini integration test (would require live key in CI).
`tests/test_llm.py` already gates Qwen tests on `RUN_LLM_TESTS=1`; future
Gemini equivalent can follow that pattern.

### Gates (unchanged)

`uv run pytest`, `uv run ruff check src tests`, `uv run mypy src` (strict
mode must stay clean — the consolidated `LLMProto` will need a single
typing pass).

## Documentation

All updates are modifications or deletions of existing sections. No new
doc files. No new top-level sections.

- `KNOWN_ISSUES.md` — delete the stash-preserved entry once the work
  lands.
- `docs/operations.md` — append a "Switching LLM backend" note inside
  whichever existing section already covers dream operations. Three
  command forms, file paths, where to look on silent fallback. Point at
  `src/hippo/config.py` and `src/hippo/models/llm.py` for current values
  (no model ids, no thinking-level strings, no version numbers).
- `CLAUDE.md` (project) — one-line note in the existing "Running
  services" section pointing at `/hippo-backend` and the config file
  path.
- `README.md` — one bullet in the existing setup section: optional
  `uv sync --extra gemini` for cloud backend.

## Install impact

`scripts/install-dream.sh` and `launchd/dream-heavy.plist.template` are
unchanged. The whole architectural point of file-based config + secrets
is that the install path stays simple. Slash command auto-discovers from
`~/.claude/commands/`.

## Migration

- Existing users on a fresh install: nothing changes. `select_llm()`
  defaults to `local-qwen` when the config file is absent.
- The pre-existing stash (`stash@{0}`) gets dropped once this work lands
  (the Gemini code moves into the new structure). Concretely: after
  merge, `git stash drop` removes the WIP, and the
  `KNOWN_ISSUES.md:117-130` paragraph that warned about it gets deleted.

## Open questions

None outstanding. All five clarifying questions answered:
slash-command-driven (D), TOML config (B), hybrid strict/non-strict (C),
status-with-detail UX (B), secrets file at `~/.claude/hippo-secrets` (A).
