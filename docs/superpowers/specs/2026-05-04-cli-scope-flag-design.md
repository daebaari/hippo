# CLI Scope Flag — Auto-Detect Project From CWD

**Status:** Design approved 2026-05-04. Implementation pending.

## Problem

`hippo` CLI commands inconsistently handle the project scope:

- `hippo dream`, `archive`, `stats` — default to global only; require `--project foo` to touch a project.
- `hippo search`, `get` — auto-detect project from cwd, append to global scope.
- `hippo bootstrap` — `--project` is required.

Result: `cd ~/code/hippo && hippo dream` does not dream the hippo project. The user has to remember which commands auto-detect and which don't, and pass `--project hippo` redundantly when already inside the repo.

## Goal

Make every CLI command auto-detect the current project from cwd (like `git`), with the flag as an explicit override rather than a requirement.

## Decisions

1. **Read/write split for default scopes.**
   - **Scoped writes** (`dream`, `archive`, `bootstrap`) — operate on the detected project only. Writes are project-local; mixing global into a project dream muddies intent.
   - **Cross-scope reads** (`search`, `get`, `stats`, `status`) — operate on global + detected project. Reflects how retrieval already runs at hook time.

2. **Override semantics.** Explicit `--scope` flag *replaces* auto-detect; cwd is ignored once any `--scope` is passed. Repeatable for multi-scope ops (e.g., `hippo stats --scope global --scope hippo`).

3. **Flag rename.** `--project` → `--scope`. The reserved value `global` selects the global scope. The old `--global-only` flag is removed (express it as `--scope global`).

4. **No-detected-project fallback.** Uniform across all commands: if cwd has no detected project AND no `--scope`/`--all-scopes` is passed, error with:
   `"hippo: not in a project (no .git or CLAUDE.md found walking up from <cwd>); pass --scope <name> or --all-scopes"`

5. **`--all-scopes` flag for ops/cron.** Enumerates `global` + every project under `PROJECTS_ROOT`. Replaces the implicit "no flag = global only" current behavior in `dream`. The launchd plist for the 3am dream job must pass `--all-scopes`.

6. **`bootstrap` is single-scope-only.** Multiple `--scope` and `--all-scopes` produce a parser-level error. Bootstrap migrates one project at a time.

7. **Backwards compat.** Hard cutover. `--project` and `--global-only` removed entirely. The launchd plist and any user-side scripts/aliases get updated as part of the change.

8. **Detection rule unchanged.** Continue using `_resolve_project` from `capture/userprompt_hook.py`: walks up from cwd, returns the basename of the first ancestor with `.git` (dir or worktree-pointer file) OR `CLAUDE.md`. This rule is shared with the Stop capture hook; changing it affects capture scoping. Out of scope for this design — kept as-is so CLI scope and capture scope agree.

## Architecture

A single shared helper module owns scope policy. Every CLI imports it.

**New module:** `src/hippo/cli/scope_args.py`

```python
from typing import Literal

CommandKind = Literal["scoped_write", "cross_read", "targeted", "single_scope_write"]

def add_scope_args(parser: argparse.ArgumentParser, *, kind: CommandKind) -> None:
    """Register --scope (repeatable) and --all-scopes on the parser.
    For kind='single_scope_write', --scope is single-valued and --all-scopes is rejected.
    """

def resolve_scopes(args: argparse.Namespace, *, kind: CommandKind, cwd: str) -> list[Scope]:
    """Returns the list[Scope] to operate on. Raises SystemExit on invalid combos
    or unresolvable cwd (per Decision 4).

    Policy:
      - If args.scope is non-empty: return those scopes (cwd ignored).
      - Else if args.all_scopes: return [global, *every project in PROJECTS_ROOT].
      - Else: detect project from cwd via _resolve_project.
        - If detected: return scopes per kind:
          - scoped_write / single_scope_write: [project]
          - cross_read / targeted: [global, project]
        - If not detected: SystemExit with the standard error message.
    """
```

**Helper extraction:** `_resolve_project` graduates from `src/hippo/capture/userprompt_hook.py` to a new `src/hippo/scope_detect.py`. The capture hook re-imports it from there. No behavior change.

Each CLI module's scope-handling shrinks to ~3 lines:

```python
from hippo.cli.scope_args import add_scope_args, resolve_scopes
add_scope_args(p, kind="scoped_write")
# ... after parse_args ...
scopes = resolve_scopes(args, kind="scoped_write", cwd=os.getcwd())
```

## Per-Command Behavior

| Command | Kind | No flag, in project | No flag, no project | `--scope X` | `--scope X --scope Y` | `--all-scopes` |
|---|---|---|---|---|---|---|
| `hippo dream` | scoped_write | project only | error | X only | X + Y | global + every project |
| `hippo archive <id>` | targeted | global + project | error | X | X + Y (search in order) | scan all |
| `hippo bootstrap` | single_scope_write | project only | error | X only | parser error | parser error |
| `hippo search <q>` | cross_read | global + project | error | X | X + Y | global + every project |
| `hippo get <id>` | targeted | global + project | error | X | X + Y (search in order) | scan all |
| `hippo stats` | cross_read | global + project | error | X | X + Y | global + every project |
| `hippo status` | cross_read | global + project | error | X | X + Y | global + every project |

`archive` and `get` use the scope list as an **ordered search path**; first scope containing the head_id wins.

## Files Touched

**Code:**
- `src/hippo/cli/scope_args.py` — new shared helper.
- `src/hippo/scope_detect.py` — new module, contains relocated `_resolve_project`.
- `src/hippo/capture/userprompt_hook.py` — re-import `_resolve_project` from new location.
- `src/hippo/cli/dream_heavy.py` — switch to shared helper. Remove `--global-only`. Remove the `not args.project or args.global_only` branch (replaced by `resolve_scopes` policy).
- `src/hippo/cli/archive.py` — switch to shared helper.
- `src/hippo/cli/get.py` — switch to shared helper. Remove the inlined `PROJECTS_ROOT.iterdir()` fallback (now reached via `--all-scopes`).
- `src/hippo/cli/search.py` — switch to shared helper.
- `src/hippo/cli/stats.py` — switch to shared helper.
- `src/hippo/cli/dream_status.py` — switch to shared helper. (Verify current behavior; add scope args.)
- `src/hippo/cli/dream_bootstrap.py` — switch to shared helper with `single_scope_write`. Remove `required=True` on the project arg.

**Launchd:**
- The 3am `com.<user>.dream-heavy` plist passes `--all-scopes` to `bin/dream-heavy`. (This fixes a real existing gap — see Decisions §5. Today's plist doesn't pass `--project`, so the nightly job dreams `global` only.)

**Tests:**
- `tests/test_scope_args.py` — new. Matrix: `kind` × explicit `--scope` × `--all-scopes` × cwd-in-project / cwd-elsewhere → expected scope list. Error cases. `single_scope_write` rejection of multi-scope/all-scopes.
- `tests/test_get.py`, `tests/test_search.py`, `tests/test_archive.py`, `tests/test_dream_heavy.py`, `tests/test_dream_bootstrap.py`, `tests/test_dream_status.py`, `tests/test_stats.py` — update existing flag-related tests to use `--scope`. Add cwd-detection cases (using a tmp git repo fixture). Add error-when-cwd-not-in-project cases.
- Existing `_resolve_project` tests follow the helper to its new location (or stay in the capture-hook test file, importing from the new path).

**Claude-facing surface:**
- `~/.claude/commands/hippo-dream.md` — replace `--project` examples with `--scope`. Document `--all-scopes`. Note that `cd <repo> && hippo dream` Just Works.
- `~/.claude/commands/hippo-backend.md` — scan and update if any flag mentions exist.
- `CLAUDE.md` (this repo) — scan and update flag references.
- `README.md` — update example invocations.
- `docs/operations.md` — update operational examples (most likely heavy hitter).
- `KNOWN_ISSUES.md` — update any debug recipes that reference old flags.
- `~/.claude/docs/specs/2026-04-28-atomic-memory-system-design.md` — canonical spec, update any flag references in place (per the global memory rule: edit-or-delete, don't append a "what changed" section).
- Argparse `--help` strings — updated as part of code changes.

## Memory Pruning

Old-flag references in the project memory store will misguide future Claude sessions. Pruning runs **after the code change is merged** (otherwise users see docs pointing to flags the binary already rejects).

**Discovery:**
```bash
hippo search "memory-stats --project" --all-scopes
hippo search "memory-archive --project" --all-scopes
hippo search "global-only flag" --all-scopes
hippo search "dream-heavy --project" --all-scopes
```
For each hit, `hippo get <head_id>` to confirm the body's *primary value* is flag-instruction (vs. an incidental mention).

**Confirmed targets** (found during brainstorming):
- `e8e6ca4e72aa4382a6ef995c89e5d2c2` — "Using memory-stats and SQL queries to inspect archived Hippo atoms"
- `eeb7412592244c79a3f0c35bc7b732b1` — "Procedures for testing and verifying Hippo memory pruning"

**Action:**
```bash
hippo archive <head_id> --reason "stale: superseded by --scope flag rename"
```

**Decision rule for incidental mentions:** if a body's primary topic is something else and the old-flag mention is one line in an example, leave it — the next dream cycle catches it via the review phase. Only archive bodies whose value *is* the flag instruction.

## Testing & Verification

**Automated gates** (existing): `uv run pytest`, `uv run ruff check`, `uv run mypy src` — all green before merge.

**Migration verification (manual checklist):**
- `grep -rn "\-\-project\|\-\-global-only" src/ tests/ docs/ ~/.claude/commands/hippo-*.md ~/.claude/docs/specs/2026-04-28-atomic-memory-system-design.md` returns nothing.
- From `~/code/hippo`: `hippo dream` processes hippo scope only.
- From `~`: `hippo dream` errors with the standard message.
- From `~`: `hippo dream --all-scopes` processes global + every project.
- Trigger nightly path: `launchctl kickstart -k gui/$(id -u)/com.<user>.dream-heavy` after plist update; verify it picks up `--all-scopes` and processes all scopes.
- `hippo bootstrap --scope foo --scope bar` returns a parser error.

## Risks

- **Behavioral change to nightly cron.** Today's `dream-heavy` plist runs without scope args, dreaming `global` only. After this change with `--all-scopes`, the 3am job will process every project for the first time, which could produce a longer-than-usual run on first execution as it catches up on accumulated captures across all projects. Acceptable: the job is AC-only and the user is asleep.
- **Capture/CLI scope-detection drift.** Mitigated by sharing one `_resolve_project` implementation across both. Tests in both locations exercise it.
- **Memory archive timing.** Archiving before code merge would leave docs that reference flags the binary still accepts. Archiving after merge is the correct order; documented in the Memory Pruning section.

## Out of Scope

- Changing the project-boundary detection rule (e.g., dropping `CLAUDE.md` or adding a `.hippo` sentinel). Considered and rejected — would break the documented `~/.claude/` infrastructure-scope workflow and require capture-side migration.
- A central project registry (`~/.claude/memory/projects.toml`). Implicit basename-based identification stays.
- Changes to `bin/*` shim semantics beyond passing through new flag names.
