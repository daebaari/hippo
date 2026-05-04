# CLI Scope Flag — Auto-Detect Project From CWD — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `--project` → `--scope` across all `hippo` CLI commands, make project auto-detected from cwd, add `--all-scopes` for ops/cron, and update all Claude-facing docs and stale memory bodies.

**Architecture:** A single shared helper module (`hippo.cli.scope_args`) owns scope policy. Project-detection logic (`resolve_project`) lives in a new `hippo.scope_detect` module shared by the capture hook and CLI. Each CLI command shrinks to a few lines of scope plumbing.

**Tech Stack:** Python 3.12+, argparse, pytest, sqlite (existing). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-04-cli-scope-flag-design.md`

---

## File Structure

**New files:**
- `src/hippo/scope_detect.py` — relocated `resolve_project` (was `_resolve_project` in capture/userprompt_hook.py).
- `src/hippo/cli/scope_args.py` — shared `add_scope_args` and `resolve_scopes` helpers used by every CLI.
- `tests/test_scope_args.py` — unit tests for the helpers.

**Modified files:**
- `src/hippo/capture/userprompt_hook.py` — re-import `resolve_project` from new location, delete the local copy.
- `src/hippo/cli/dream_heavy.py` — switch to shared helper. Drop `--global-only` and `--project`.
- `src/hippo/cli/archive.py` — switch to shared helper.
- `src/hippo/cli/get.py` — switch to shared helper. Drop the inline PROJECTS_ROOT scan (now reachable via `--all-scopes`).
- `src/hippo/cli/search.py` — switch to shared helper.
- `src/hippo/cli/stats.py` — switch to shared helper.
- `src/hippo/cli/dream_status.py` — convert `--scope` from single-string to the shared (repeatable) form; align "no flag = all scopes" → "no flag = global+detected, --all-scopes for everything."
- `src/hippo/cli/dream_bootstrap.py` — switch to shared helper with `single_scope_write` kind. Drop `required=True` on the project arg.
- `tests/test_resolve_project.py` — update import path.
- `tests/test_dream_heavy_orchestrator.py`, `tests/test_dream_status_cli.py`, `tests/test_stats_cli.py`, `tests/test_cli_tools.py`, `tests/test_bootstrap.py`, `tests/test_capture.py`, `tests/test_userprompt_hook.py` — update any `--project`/`--global-only` references to the new flag names.
- `launchd/dream-heavy.plist.template` — add `--all-scopes` to the program arguments.
- `~/.claude/commands/hippo-dream.md`, `~/.claude/commands/hippo-backend.md` — update flag examples.
- `CLAUDE.md`, `README.md`, `docs/operations.md`, `KNOWN_ISSUES.md` — update flag references.
- `~/.claude/docs/specs/2026-04-28-atomic-memory-system-design.md` — update flag references.

---

## Task 1: Relocate `resolve_project` to `hippo.scope_detect`

This is a pure refactor — no behavior change. Done first so Task 3 can import the new path cleanly.

**Files:**
- Create: `src/hippo/scope_detect.py`
- Modify: `src/hippo/capture/userprompt_hook.py:35-89` (delete `_resolve_project` and `_read_worktree_pointer`, add import from new location)
- Modify: `tests/test_resolve_project.py:6` (update import)

- [ ] **Step 1: Create `src/hippo/scope_detect.py` with the relocated functions**

Copy `_resolve_project` and `_read_worktree_pointer` from `src/hippo/capture/userprompt_hook.py` into the new file, drop the leading underscore on `resolve_project` (it is now a public helper). Keep `_read_worktree_pointer` private — only `resolve_project` needs to be exported.

```python
"""Detect the current project from a working directory.

Walks up from cwd looking for a project boundary marker (`.git` directory,
`.git` worktree-pointer file, or `CLAUDE.md`). The basename of that
directory is the project name used by Hippo's scope system.

This module is shared between the Stop/UserPromptSubmit capture hooks and
the CLI commands so capture-side and CLI-side scope detection always agree.
"""
from __future__ import annotations

from pathlib import Path


def resolve_project(cwd: str) -> str | None:
    """Walk up from cwd; return the basename of the first dir that looks like
    a project root, or None if none found.

    Git worktrees: when ``.git`` is a file (a worktree pointer like
    ``gitdir: /path/to/main/.git/worktrees/<name>``), resolve to the MAIN
    repo's basename so captures from a worktree land in the same scope as
    the main checkout. Falls back to the worktree directory's own name if
    the pointer can't be parsed.
    """
    p = Path(cwd).resolve()
    for candidate in [p, *p.parents]:
        git_entry = candidate / ".git"
        if git_entry.is_dir() or (candidate / "CLAUDE.md").exists():
            return candidate.name
        if git_entry.is_file():
            main_repo = _read_worktree_pointer(git_entry)
            return main_repo.name if main_repo else candidate.name
    return None


def _read_worktree_pointer(git_file: Path) -> Path | None:
    """Parse a worktree's `.git` file. Returns the MAIN repo path, or None.

    Format (per `man gitrepository-layout`):
        gitdir: <absolute-or-relative-path-to>/.git/worktrees/<name>
    The main repo is the parent of the `/.git/worktrees/<name>` segment.
    """
    try:
        first_line = git_file.read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    prefix = "gitdir:"
    if not first_line.startswith(prefix):
        return None
    pointer = first_line[len(prefix):].strip()
    if not pointer:
        return None
    pointer_path = Path(pointer)
    if not pointer_path.is_absolute():
        pointer_path = (git_file.parent / pointer_path).resolve()
    parts = pointer_path.parts
    try:
        idx = len(parts) - 1 - list(reversed(parts)).index(".git")
    except ValueError:
        return None
    if idx < 1:
        return None
    return Path(*parts[:idx])
```

- [ ] **Step 2: Update `src/hippo/capture/userprompt_hook.py` to re-import**

Delete the bodies of `_resolve_project` and `_read_worktree_pointer` from `userprompt_hook.py` (lines 35-89). Add at the top of the imports section (after the stdlib imports):

```python
from hippo.scope_detect import resolve_project as _resolve_project
```

The local alias `_resolve_project` keeps existing call sites in this file working without further edits. Other modules that imported `_resolve_project` from `capture.userprompt_hook` (notably `cli/search.py` and `cli/get.py`) will be migrated in their own tasks; their existing import keeps working through the re-export until then.

- [ ] **Step 3: Update test import**

In `tests/test_resolve_project.py:6`, replace:

```python
from hippo.capture.userprompt_hook import _resolve_project
```

with:

```python
from hippo.scope_detect import resolve_project as _resolve_project
```

The alias preserves the rest of the test file unchanged.

- [ ] **Step 4: Run the test suite to confirm no regression**

```bash
uv run pytest tests/test_resolve_project.py tests/test_userprompt_hook.py tests/test_capture.py tests/test_stop_hook.py -v
```

Expected: all pass. If any fail, the import re-export is wrong — verify Step 2.

- [ ] **Step 5: Run full gates**

```bash
uv run ruff check src tests
uv run mypy src
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/hippo/scope_detect.py src/hippo/capture/userprompt_hook.py tests/test_resolve_project.py
git commit -m "refactor: extract resolve_project into hippo.scope_detect

Pure relocation, no behavior change. The helper is shared by the capture
hook and (in a follow-up task) the CLI scope-args module."
```

---

## Task 2: Write failing tests for `hippo.cli.scope_args`

Test-driven: define the helper's contract first.

**Files:**
- Create: `tests/test_scope_args.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for hippo.cli.scope_args — shared scope-resolution helpers."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hippo.cli.scope_args import (
    CommandKind,
    add_scope_args,
    resolve_scopes,
)
from hippo.storage.multi_store import Scope


def _parse(argv: list[str], *, kind: CommandKind) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_scope_args(p, kind=kind)
    return p.parse_args(argv)


def test_explicit_scope_replaces_cwd_detection(tmp_path: Path) -> None:
    repo = tmp_path / "myrepo"
    (repo / ".git").mkdir(parents=True)
    args = _parse(["--scope", "explicit"], kind="cross_read")
    scopes = resolve_scopes(args, kind="cross_read", cwd=str(repo))
    assert scopes == [Scope.project("explicit")]


def test_explicit_scope_repeatable(tmp_path: Path) -> None:
    args = _parse(["--scope", "a", "--scope", "b"], kind="cross_read")
    scopes = resolve_scopes(args, kind="cross_read", cwd=str(tmp_path))
    assert scopes == [Scope.project("a"), Scope.project("b")]


def test_explicit_scope_global_value(tmp_path: Path) -> None:
    args = _parse(["--scope", "global"], kind="scoped_write")
    scopes = resolve_scopes(args, kind="scoped_write", cwd=str(tmp_path))
    assert scopes == [Scope.global_()]


def test_scoped_write_in_project_returns_project_only(tmp_path: Path) -> None:
    repo = tmp_path / "hippo"
    (repo / ".git").mkdir(parents=True)
    args = _parse([], kind="scoped_write")
    scopes = resolve_scopes(args, kind="scoped_write", cwd=str(repo))
    assert scopes == [Scope.project("hippo")]


def test_cross_read_in_project_returns_global_plus_project(tmp_path: Path) -> None:
    repo = tmp_path / "hippo"
    (repo / ".git").mkdir(parents=True)
    args = _parse([], kind="cross_read")
    scopes = resolve_scopes(args, kind="cross_read", cwd=str(repo))
    assert scopes == [Scope.global_(), Scope.project("hippo")]


def test_targeted_in_project_returns_global_plus_project(tmp_path: Path) -> None:
    repo = tmp_path / "hippo"
    (repo / ".git").mkdir(parents=True)
    args = _parse([], kind="targeted")
    scopes = resolve_scopes(args, kind="targeted", cwd=str(repo))
    assert scopes == [Scope.global_(), Scope.project("hippo")]


def test_no_project_no_flag_errors(tmp_path: Path) -> None:
    bare = tmp_path / "nothing"
    bare.mkdir()
    args = _parse([], kind="scoped_write")
    with pytest.raises(SystemExit):
        resolve_scopes(args, kind="scoped_write", cwd=str(bare))


def test_no_project_no_flag_errors_for_cross_read(tmp_path: Path) -> None:
    bare = tmp_path / "nothing"
    bare.mkdir()
    args = _parse([], kind="cross_read")
    with pytest.raises(SystemExit):
        resolve_scopes(args, kind="cross_read", cwd=str(bare))


def test_all_scopes_enumerates_global_and_every_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects_root = tmp_path / "projects"
    for name in ["alpha", "beta"]:
        memdir = projects_root / name / "memory"
        memdir.mkdir(parents=True)
        (memdir / "memory.db").touch()
    monkeypatch.setattr("hippo.cli.scope_args.PROJECTS_ROOT", projects_root)

    args = _parse(["--all-scopes"], kind="cross_read")
    scopes = resolve_scopes(args, kind="cross_read", cwd=str(tmp_path))
    assert scopes == [
        Scope.global_(),
        Scope.project("alpha"),
        Scope.project("beta"),
    ]


def test_all_scopes_works_outside_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects_root = tmp_path / "projects"
    projects_root.mkdir()
    monkeypatch.setattr("hippo.cli.scope_args.PROJECTS_ROOT", projects_root)
    bare = tmp_path / "nothing"
    bare.mkdir()
    args = _parse(["--all-scopes"], kind="scoped_write")
    # Should NOT error even though cwd has no project.
    scopes = resolve_scopes(args, kind="scoped_write", cwd=str(bare))
    assert scopes == [Scope.global_()]


def test_single_scope_write_rejects_multi_scope() -> None:
    p = argparse.ArgumentParser()
    add_scope_args(p, kind="single_scope_write")
    with pytest.raises(SystemExit):
        p.parse_args(["--scope", "a", "--scope", "b"])


def test_single_scope_write_rejects_all_scopes() -> None:
    p = argparse.ArgumentParser()
    add_scope_args(p, kind="single_scope_write")
    # add_scope_args must NOT register --all-scopes for single_scope_write.
    with pytest.raises(SystemExit):
        p.parse_args(["--all-scopes"])


def test_single_scope_write_accepts_one_scope(tmp_path: Path) -> None:
    args = _parse(["--scope", "foo"], kind="single_scope_write")
    scopes = resolve_scopes(args, kind="single_scope_write", cwd=str(tmp_path))
    assert scopes == [Scope.project("foo")]


def test_single_scope_write_auto_detects_in_project(tmp_path: Path) -> None:
    repo = tmp_path / "myproj"
    (repo / ".git").mkdir(parents=True)
    args = _parse([], kind="single_scope_write")
    scopes = resolve_scopes(args, kind="single_scope_write", cwd=str(repo))
    assert scopes == [Scope.project("myproj")]
```

- [ ] **Step 2: Run the test file to confirm it fails (module doesn't exist yet)**

```bash
uv run pytest tests/test_scope_args.py -v
```

Expected: collection error or `ImportError: cannot import name ... from hippo.cli.scope_args`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_scope_args.py
git commit -m "test: add failing tests for cli.scope_args module"
```

---

## Task 3: Implement `hippo.cli.scope_args`

**Files:**
- Create: `src/hippo/cli/scope_args.py`

- [ ] **Step 1: Implement the module**

```python
"""Shared --scope / --all-scopes argparse helpers and resolution policy.

Every Hippo CLI command uses these helpers so scope semantics stay consistent.
See ``docs/superpowers/specs/2026-05-04-cli-scope-flag-design.md``.

CommandKind:
  - scoped_write       — operates on the detected project only.
                         Examples: ``hippo dream``.
  - cross_read         — global + detected project (or whatever --scope says).
                         Examples: ``hippo search``, ``hippo stats``.
  - targeted           — same default as cross_read; --all-scopes implies a
                         broad search-by-id across every store on disk.
                         Examples: ``hippo get``, ``hippo archive``.
  - single_scope_write — exactly one scope; multi-scope and --all-scopes are
                         parser-level errors. Example: ``hippo bootstrap``.
"""
from __future__ import annotations

import argparse
import sys
from typing import Literal

from hippo.config import DB_FILENAME, PROJECTS_ROOT
from hippo.scope_detect import resolve_project
from hippo.storage.multi_store import Scope

CommandKind = Literal[
    "scoped_write", "cross_read", "targeted", "single_scope_write"
]

_NOT_IN_PROJECT_HINT = (
    "hippo: not in a project (no .git or CLAUDE.md found walking up from {cwd}); "
    "pass --scope <name> or --all-scopes"
)


class _SingleScopeAction(argparse.Action):
    """--scope action for single_scope_write: rejects a second occurrence."""

    def __call__(  # type: ignore[override]
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, None):
            parser.error(
                f"{option_string}: only one --scope value allowed for this command"
            )
        setattr(namespace, self.dest, [values])


def add_scope_args(
    parser: argparse.ArgumentParser, *, kind: CommandKind
) -> None:
    """Register --scope (and, where applicable, --all-scopes) on the parser."""
    if kind == "single_scope_write":
        parser.add_argument(
            "--scope",
            action=_SingleScopeAction,
            default=[],
            help=(
                "Scope to operate on (project name, or 'global'). "
                "Auto-detected from cwd if omitted. Single-valued for this command."
            ),
        )
        return

    parser.add_argument(
        "--scope",
        action="append",
        default=[],
        help=(
            "Scope to operate on (project name, or 'global'). Repeatable. "
            "Auto-detected from cwd if omitted."
        ),
    )
    parser.add_argument(
        "--all-scopes",
        action="store_true",
        default=False,
        dest="all_scopes",
        help="Operate on global + every project under PROJECTS_ROOT.",
    )


def _enumerate_all_scopes() -> list[Scope]:
    scopes: list[Scope] = [Scope.global_()]
    if PROJECTS_ROOT.exists():
        for entry in sorted(PROJECTS_ROOT.iterdir()):
            if (entry / "memory" / DB_FILENAME).exists():
                scopes.append(Scope.project(entry.name))
    return scopes


def _scope_from_value(value: str) -> Scope:
    return Scope.global_() if value == "global" else Scope.project(value)


def resolve_scopes(
    args: argparse.Namespace,
    *,
    kind: CommandKind,
    cwd: str,
) -> list[Scope]:
    """Apply policy to produce the list of scopes the command should target.

    Raises SystemExit (via sys.exit) when cwd has no detectable project and
    no flag is provided.
    """
    explicit: list[str] = list(args.scope or [])
    if explicit:
        return [_scope_from_value(v) for v in explicit]

    if getattr(args, "all_scopes", False):
        return _enumerate_all_scopes()

    project = resolve_project(cwd)
    if project is None:
        sys.stderr.write(_NOT_IN_PROJECT_HINT.format(cwd=cwd) + "\n")
        raise SystemExit(2)

    if kind in ("scoped_write", "single_scope_write"):
        return [Scope.project(project)]
    # cross_read, targeted
    return [Scope.global_(), Scope.project(project)]
```

- [ ] **Step 2: Run the new tests; expect all pass**

```bash
uv run pytest tests/test_scope_args.py -v
```

Expected: every test passes.

- [ ] **Step 3: Run lint and mypy**

```bash
uv run ruff check src tests
uv run mypy src
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add src/hippo/cli/scope_args.py
git commit -m "feat(cli): add scope_args helper for --scope and --all-scopes"
```

---

## Task 4: Migrate `dream_heavy.py` to scope_args

**Files:**
- Modify: `src/hippo/cli/dream_heavy.py`
- Modify: `tests/test_dream_heavy_orchestrator.py` (update any --project/--global-only references)

- [ ] **Step 1: Find existing test references to old flags**

```bash
grep -n "\-\-project\|\-\-global-only\|args\.project\|args\.global_only" tests/test_dream_heavy_orchestrator.py
```

Note any matches; they will need editing to use `--scope` (or `args.scope`/`args.all_scopes` if the test reaches into argparse namespaces).

- [ ] **Step 2: Update those test sites**

For any test that previously asserted behavior with `--project`, replace with `--scope`. For tests that called `main()` or constructed argv, swap the strings. Keep the test intent.

If a test exercised "no --project = global only", the new equivalent is "--scope global" (or "--all-scopes" if the test wanted everything). Pick the variant that matches the test's intent.

If a test exercises "no flag, run from a project repo," it must now run inside a `tmp_path` git repo and assert the project scope is detected. Add such a test if absent — see Task 9 for a pattern.

- [ ] **Step 3: Replace `src/hippo/cli/dream_heavy.py` with the migrated form**

```python
"""Heavy dream entry point. Run by launchd at 3am, or manually via /hippo-dream."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from hippo.cli.scope_args import add_scope_args, resolve_scopes
from hippo.config import ConfigError
from hippo.daemon.client import DaemonClient
from hippo.dream.heavy import run_heavy_dream_all_scopes
from hippo.models.llm import select_llm


def _is_on_ac() -> bool:
    """macOS-only: returns True if on AC power."""
    try:
        out = subprocess.check_output(["pmset", "-g", "ps"], text=True)
        return "AC Power" in out
    except Exception:
        return True


def main() -> int:
    p = argparse.ArgumentParser(prog="dream-heavy")
    p.add_argument("--force", action="store_true", help="bypass AC check")
    p.add_argument(
        "--strict",
        action="store_true",
        help="hard-fail on backend misconfiguration",
    )
    add_scope_args(p, kind="scoped_write")
    args = p.parse_args()

    if not args.force and not _is_on_ac():
        sys.stderr.write(
            "Not on AC power; skipping heavy dream. Use --force to override.\n"
        )
        return 0

    scopes = resolve_scopes(args, kind="scoped_write", cwd=os.getcwd())

    daemon = DaemonClient(socket_path=Path.home() / ".claude" / "memory-daemon.sock")
    try:
        llm = select_llm(strict=args.strict)
    except ConfigError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1
    stats = run_heavy_dream_all_scopes(scopes=scopes, llm=llm, daemon=daemon)
    print("heavy dream complete:")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_dream_heavy_orchestrator.py tests/test_scope_args.py -v
```

Expected: all pass. Common failures:
- A test that constructs argv with `--project foo` → update to `--scope foo`.
- A test that asserted `args.global_only` → drop it; replace with checking `scopes == [Scope.global_()]`.

- [ ] **Step 5: Lint + mypy**

```bash
uv run ruff check src tests
uv run mypy src
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/hippo/cli/dream_heavy.py tests/test_dream_heavy_orchestrator.py
git commit -m "feat(cli): migrate dream_heavy to scope_args (auto-detect + --all-scopes)"
```

---

## Task 5: Migrate `archive.py` to scope_args (targeted kind)

**Files:**
- Modify: `src/hippo/cli/archive.py`
- Modify: `tests/test_cli_tools.py` (or wherever archive is exercised — search first)

- [ ] **Step 1: Locate archive tests**

```bash
grep -rn "archive_head_cli\|memory-archive\|--reason" tests/
```

- [ ] **Step 2: Replace `src/hippo/cli/archive.py`**

```python
"""memory-archive <head_id> --reason '...': soft-delete a head."""
from __future__ import annotations

import argparse
import os
import sys

from hippo.cli.scope_args import add_scope_args, resolve_scopes
from hippo.storage.heads import archive_head, get_head
from hippo.storage.multi_store import open_store


def archive_head_cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="memory-archive")
    p.add_argument("head_id")
    p.add_argument("--reason", required=True)
    add_scope_args(p, kind="targeted")
    args = p.parse_args(argv)

    scopes = resolve_scopes(args, kind="targeted", cwd=os.getcwd())
    for scope in scopes:
        store = open_store(scope)
        try:
            head = get_head(store.conn, args.head_id)
            if head is None:
                continue
            archive_head(store.conn, args.head_id, reason=args.reason)
            print(f"archived {args.head_id} ({scope.as_string()}): {args.reason}")
            return 0
        finally:
            store.conn.close()
    print(f"head_id {args.head_id} not found", file=sys.stderr)
    return 1


def main() -> int:
    return archive_head_cli(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Update any archive tests that used `--project`**

For each test file matched in Step 1, replace `--project foo` with `--scope foo`. If a test asserted "no flag = global only," update its expectation to either pass `--scope global` explicitly or run inside a tmp git repo.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_cli_tools.py tests/test_scope_args.py -v
```

Expected: all pass.

- [ ] **Step 5: Lint + mypy**

```bash
uv run ruff check src tests
uv run mypy src
```

- [ ] **Step 6: Commit**

```bash
git add src/hippo/cli/archive.py tests/test_cli_tools.py
git commit -m "feat(cli): migrate archive to scope_args"
```

---

## Task 6: Migrate `get.py` to scope_args (targeted kind)

The current `get.py` has an extra fallback: when cwd has no project, it scans every PROJECTS_ROOT entry. With the new design that fallback is reachable explicitly via `--all-scopes` instead of being implicit.

**Files:**
- Modify: `src/hippo/cli/get.py`
- Modify: tests for get (search `tests/` for `memory-get` or `get_body_cli`).

- [ ] **Step 1: Find get tests**

```bash
grep -rn "get_body_cli\|memory-get" tests/
```

- [ ] **Step 2: Replace `src/hippo/cli/get.py`**

```python
"""memory-get <head_id>: print the body markdown for the given head."""
from __future__ import annotations

import argparse
import os
import sys

from hippo.cli.scope_args import add_scope_args, resolve_scopes
from hippo.config import BODIES_SUBDIR
from hippo.storage.body_files import read_body_file
from hippo.storage.heads import get_head
from hippo.storage.multi_store import open_store


def get_body_cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="memory-get")
    p.add_argument("head_id")
    add_scope_args(p, kind="targeted")
    args = p.parse_args(argv)

    scopes = resolve_scopes(args, kind="targeted", cwd=os.getcwd())
    for scope in scopes:
        store = open_store(scope)
        try:
            head = get_head(store.conn, args.head_id)
            if head is None or head.archived:
                continue
            body_path = store.memory_dir / BODIES_SUBDIR / f"{head.body_id}.md"
            body = read_body_file(body_path)
            print(f"# {body.title}")
            print(f"scope: {body.scope}  body_id: {body.body_id}")
            print()
            print(body.content)
            return 0
        finally:
            store.conn.close()
    print(f"head_id {args.head_id} not found in any scope", file=sys.stderr)
    return 1


def main() -> int:
    return get_body_cli(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Update get tests**

Any test that relied on the implicit "scan all projects when cwd has no project" should now pass `--all-scopes` explicitly, or run inside a tmp git repo. Update accordingly.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_cli_tools.py tests/test_scope_args.py -v
```

Expected: all pass.

- [ ] **Step 5: Lint + mypy + commit**

```bash
uv run ruff check src tests
uv run mypy src
git add src/hippo/cli/get.py tests/test_cli_tools.py
git commit -m "feat(cli): migrate get to scope_args; remove implicit scan-all"
```

---

## Task 7: Migrate `search.py` to scope_args (cross_read kind)

**Files:**
- Modify: `src/hippo/cli/search.py`
- Modify: any search tests (`grep -rn "memory_search_cli\|memory-search" tests/`)

- [ ] **Step 1: Replace `src/hippo/cli/search.py`**

```python
"""memory-search '<query>': run full retrieval pipeline; print the <memory> block."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from hippo.cli.scope_args import add_scope_args, resolve_scopes
from hippo.config import (
    RETRIEVAL_HOP_LIMIT_PER_SEED,
    RETRIEVAL_RERANK_TOP_K,
    RETRIEVAL_TOTAL_CAP,
    RETRIEVAL_VECTOR_TOP_K_PER_SCOPE,
)
from hippo.daemon.client import DaemonClient
from hippo.retrieval.inject import format_memory_block, load_body_preview
from hippo.retrieval.pipeline import RetrievalPipeline
from hippo.storage.multi_store import resolve_memory_dir

DEFAULTS = dict(
    vector_top_k_per_scope=RETRIEVAL_VECTOR_TOP_K_PER_SCOPE,
    hop_limit_per_seed=RETRIEVAL_HOP_LIMIT_PER_SEED,
    total_cap=RETRIEVAL_TOTAL_CAP,
    rerank_top_k=RETRIEVAL_RERANK_TOP_K,
)


def memory_search_cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="memory-search")
    p.add_argument("query")
    p.add_argument(
        "--socket", default=str(Path.home() / ".claude" / "memory-daemon.sock")
    )
    add_scope_args(p, kind="cross_read")
    args = p.parse_args(argv)

    scopes = resolve_scopes(args, kind="cross_read", cwd=os.getcwd())
    daemon = DaemonClient(socket_path=Path(args.socket))
    pipeline = RetrievalPipeline(daemon=daemon, scopes=scopes, **DEFAULTS)
    result = pipeline.run(args.query)
    scope_to_dir = {scope.as_string(): resolve_memory_dir(scope) for scope in scopes}
    block = format_memory_block(
        result,
        body_resolver=lambda hit: load_body_preview(
            scope_to_dir[hit.scope], hit.head.body_id
        ),
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
```

- [ ] **Step 2: Update search tests**

```bash
grep -rn "memory_search_cli\|memory-search" tests/
```

For each match: swap `--project` → `--scope`. If a test ran from a non-project cwd and expected global-only, update it to pass `--scope global` or run inside a tmp git repo.

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_scope_args.py -v
uv run pytest tests/ -k "search" -v
```

Expected: all pass.

- [ ] **Step 4: Lint + mypy + commit**

```bash
uv run ruff check src tests
uv run mypy src
git add src/hippo/cli/search.py tests/
git commit -m "feat(cli): migrate search to scope_args"
```

---

## Task 8: Migrate `stats.py` to scope_args (cross_read kind)

**Files:**
- Modify: `src/hippo/cli/stats.py`
- Modify: `tests/test_stats_cli.py`

- [ ] **Step 1: Replace `src/hippo/cli/stats.py`**

```python
"""memory-stats: print body/head/edge counts + recent dream runs per scope."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from hippo.cli.scope_args import add_scope_args, resolve_scopes
from hippo.storage.dream_runs import get_recent_runs
from hippo.storage.multi_store import Scope, open_store


def collect_stats(scopes: list[Scope]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for scope in scopes:
        store = open_store(scope)
        try:
            body_count = int(
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM bodies WHERE archived = 0"
                ).fetchone()["c"]
            )
            head_count = int(
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM heads WHERE archived = 0"
                ).fetchone()["c"]
            )
            edge_count = int(
                store.conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"]
            )
            recent = get_recent_runs(store.conn, limit=5)
            out[scope.as_string()] = {
                "body_count": body_count,
                "head_count": head_count,
                "edge_count": edge_count,
                "recent_runs": [
                    {
                        "run_id": r.run_id,
                        "type": r.type,
                        "status": r.status,
                        "started_at": r.started_at,
                        "atoms_created": r.atoms_created,
                        "heads_created": r.heads_created,
                        "edges_created": r.edges_created,
                        "contradictions_resolved": r.contradictions_resolved,
                    }
                    for r in recent
                ],
            }
        finally:
            store.conn.close()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memory-stats")
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of human format"
    )
    add_scope_args(parser, kind="cross_read")
    args = parser.parse_args(argv)

    scopes = resolve_scopes(args, kind="cross_read", cwd=os.getcwd())
    result = collect_stats(scopes)

    if args.json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    for scope_name, info in result.items():
        print(f"=== {scope_name} ===")
        print(f"  bodies (active):  {info['body_count']}")
        print(f"  heads  (active):  {info['head_count']}")
        print(f"  edges:            {info['edge_count']}")
        print("  recent dream runs:")
        for r in info["recent_runs"]:
            print(
                f"    [{r['type']:5}] run #{r['run_id']:>4} status={r['status']:9} "
                f"atoms={r['atoms_created']} heads={r['heads_created']} "
                f"edges={r['edges_created']} contradictions={r['contradictions_resolved']}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Update `tests/test_stats_cli.py`**

The current test only exercises `collect_stats` with an explicit list — that test stays as-is (no flag interaction). If new tests for the CLI argv layer are wanted, add them following the pattern in Task 9 below.

- [ ] **Step 3: Run tests + gates + commit**

```bash
uv run pytest tests/test_stats_cli.py tests/test_scope_args.py -v
uv run ruff check src tests
uv run mypy src
git add src/hippo/cli/stats.py
git commit -m "feat(cli): migrate stats to scope_args"
```

---

## Task 9: Migrate `dream_status.py` to repeatable `--scope` and `--all-scopes`

`dream_status.py` already accepts `--scope` but with different semantics: single-string, with no flag meaning "all scopes." After migration, `--scope` becomes repeatable, "no flag = global+detected project" (cross_read), and "all scopes" requires `--all-scopes`.

**Files:**
- Modify: `src/hippo/cli/dream_status.py`
- Modify: `tests/test_dream_status_cli.py`

- [ ] **Step 1: Inspect existing tests**

```bash
grep -n "\-\-scope\|args\.scope" tests/test_dream_status_cli.py
```

Note any test that calls `dream_status_cli([])` and expects all-scope behavior; that test must change its expectation (now: error if cwd has no project, or scopes [global, detected_project] if cwd is in a project).

- [ ] **Step 2: Replace `src/hippo/cli/dream_status.py`**

```python
"""dream-status: print the most recent dream run across selected scope DBs."""
from __future__ import annotations

import argparse
import os
import sys
import time

from hippo.cli.scope_args import add_scope_args, resolve_scopes
from hippo.storage.dream_runs import (
    DreamRunRecord,
    get_recent_runs,
    get_running_run,
)
from hippo.storage.multi_store import Scope, open_store


def render_run_line(rec: DreamRunRecord, *, scope_name: str, now_unix: int) -> str:
    state = rec.status
    elapsed = (rec.completed_at or now_unix) - rec.started_at
    elapsed_str = f"{elapsed // 60}m" if elapsed >= 60 else f"{elapsed}s"
    phase = rec.current_phase or "?"
    if rec.phase_done is not None and rec.phase_total:
        pct = 100 * rec.phase_done / rec.phase_total
        phase_part = (
            f"phase={phase} {rec.phase_done}/{rec.phase_total} ({pct:.1f}%)"
        )
    else:
        phase_part = f"phase={phase}"
    return (
        f"{state}: {scope_name} run_id={rec.run_id} {phase_part} elapsed={elapsed_str}"
    )


def _scope_display_name(scope: Scope) -> str:
    s = scope.as_string()
    return "global" if s == "global" else s.removeprefix("project:")


def dream_status_cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dream-status")
    add_scope_args(p, kind="cross_read")
    args = p.parse_args(argv)
    now = int(time.time())

    scopes = resolve_scopes(args, kind="cross_read", cwd=os.getcwd())
    scope_pairs = [(s, _scope_display_name(s)) for s in scopes]

    # First pass: any running run in any selected scope?
    for scope, name in scope_pairs:
        store = open_store(scope)
        try:
            running = get_running_run(store.conn)
            if running is not None:
                print(render_run_line(running, scope_name=name, now_unix=now))
                return 0
        finally:
            store.conn.close()

    # Fallback: most recent completed/failed run across selected scopes.
    best: tuple[DreamRunRecord, str] | None = None
    for scope, name in scope_pairs:
        store = open_store(scope)
        try:
            recents = get_recent_runs(store.conn, limit=1)
            if recents and (best is None or recents[0].started_at > best[0].started_at):
                best = (recents[0], name)
        finally:
            store.conn.close()

    if best is None:
        print("no dream runs found")
        return 1
    print(render_run_line(best[0], scope_name=best[1], now_unix=now))
    return 0


def main() -> int:
    return dream_status_cli(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Update `tests/test_dream_status_cli.py`**

Any test that previously called `dream_status_cli([])` expecting cross-scope behavior must now either:
- Pass `["--all-scopes"]` to keep the old "all scopes" intent.
- Run inside a tmp git repo (use `monkeypatch.chdir(tmp_path / "repo_with_dotgit")`) for the `cross_read` default.

Pattern for tests that need a project cwd:

```python
def test_status_in_project_uses_global_plus_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    monkeypatch.setattr("hippo.config.PROJECTS_ROOT", tmp_path / "projects")
    repo = tmp_path / "myproj"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.chdir(repo)
    # ...exercise dream_status_cli([])...
```

- [ ] **Step 4: Run tests + gates + commit**

```bash
uv run pytest tests/test_dream_status_cli.py tests/test_scope_args.py -v
uv run ruff check src tests
uv run mypy src
git add src/hippo/cli/dream_status.py tests/test_dream_status_cli.py
git commit -m "feat(cli): migrate dream_status to scope_args (repeatable --scope)"
```

---

## Task 10: Migrate `dream_bootstrap.py` to scope_args (single_scope_write kind)

**Files:**
- Modify: `src/hippo/cli/dream_bootstrap.py`
- Modify: `tests/test_bootstrap.py` (search for `--project`)

- [ ] **Step 1: Find bootstrap tests touching the project flag**

```bash
grep -n "\-\-project\|args\.project" tests/test_bootstrap.py
```

- [ ] **Step 2: Replace the argument-parsing section of `dream_bootstrap.py`**

Replace the parser construction (currently lines 25-35 in the file) with:

```python
def main() -> int:
    p = argparse.ArgumentParser(prog="dream-bootstrap")
    p.add_argument(
        "--memory-dir",
        required=True,
        help="legacy memory dir, e.g. ~/.claude/projects/-Users-keon-kaleon-kaleon/memory",
    )
    p.add_argument(
        "--no-archive", action="store_true", help="don't move files to .legacy/"
    )
    p.add_argument(
        "--strict", action="store_true", help="hard-fail on backend misconfiguration"
    )
    add_scope_args(p, kind="single_scope_write")
    args = p.parse_args()
```

Then below that, replace the existing `g_scope = Scope.global_()` / `p_scope = Scope.project(args.project)` block with:

```python
    scopes = resolve_scopes(args, kind="single_scope_write", cwd=os.getcwd())
    # single_scope_write guarantees exactly one scope, and bootstrap requires it
    # to be a project scope (global has no legacy markdown to atomize).
    target_scope = scopes[0]
    if target_scope.kind != "project" or target_scope.project_name is None:
        sys.stderr.write(
            "ERROR: dream-bootstrap requires a project scope, got 'global'\n"
        )
        return 1
    project_name: str = target_scope.project_name

    g_scope = Scope.global_()
    p_scope = target_scope
```

Then update line 86's `project=args.project` to `project=project_name`.

Add the new imports at the top of the file (alongside the existing imports):

```python
import os

from hippo.cli.scope_args import add_scope_args, resolve_scopes
```

(`Scope.kind` is `"global"` or `"project"`, and `Scope.project_name` is `str | None` — see `src/hippo/storage/multi_store.py:24-40`.)

- [ ] **Step 3: Update `tests/test_bootstrap.py`**

Replace any `--project foo` argv with `--scope foo`. If a test previously expected a parser error when `--project` was missing, the equivalent now is "test passes when run inside a tmp git repo with no `--scope`" — add such a test if one doesn't exist.

- [ ] **Step 4: Run tests + gates + commit**

```bash
uv run pytest tests/test_bootstrap.py tests/test_scope_args.py -v
uv run ruff check src tests
uv run mypy src
git add src/hippo/cli/dream_bootstrap.py tests/test_bootstrap.py
git commit -m "feat(cli): migrate dream_bootstrap to scope_args (single_scope_write)"
```

---

## Task 11: Update launchd plist template + reinstall

**Files:**
- Modify: `launchd/dream-heavy.plist.template`
- Run: `scripts/install-dream.sh` to re-render and reload

- [ ] **Step 1: Add `--all-scopes` to the template**

Edit `launchd/dream-heavy.plist.template` and add a new `<string>` after `hippo.cli.dream_heavy`:

```xml
        <string>python</string>
        <string>-m</string>
        <string>hippo.cli.dream_heavy</string>
        <string>--all-scopes</string>
```

- [ ] **Step 2: Verify the change**

```bash
grep -n "all-scopes" launchd/dream-heavy.plist.template
```

Expected: one match.

- [ ] **Step 3: Re-render and reload (manual; user-machine-side)**

```bash
bash scripts/install-dream.sh
```

This unloads the old plist, renders the new template, and reloads.

- [ ] **Step 4: Verify the rendered plist**

```bash
grep "all-scopes" "$HOME/Library/LaunchAgents/com.$(whoami).dream-heavy.plist"
```

Expected: one match.

- [ ] **Step 5: Commit**

```bash
git add launchd/dream-heavy.plist.template
git commit -m "chore(launchd): pass --all-scopes to nightly dream-heavy

Today's plist runs dream-heavy without scope args; under the new defaults
that would error (cwd is /). --all-scopes processes global + every
project, which is what the nightly job is supposed to do."
```

---

## Task 12: Update Claude-facing documentation

Run a single grep to find every old-flag reference, then edit in place. Per the user's global memory rule: edit-or-delete, don't append.

**Files (probable, finalized by grep results):**
- `~/.claude/commands/hippo-dream.md`
- `~/.claude/commands/hippo-backend.md`
- `CLAUDE.md`
- `README.md`
- `docs/operations.md`
- `KNOWN_ISSUES.md`
- `~/.claude/docs/specs/2026-04-28-atomic-memory-system-design.md`

- [ ] **Step 1: Locate every old-flag reference**

```bash
grep -rn "\-\-project\|\-\-global-only" \
  CLAUDE.md README.md docs/ KNOWN_ISSUES.md \
  $HOME/.claude/commands/hippo-dream.md \
  $HOME/.claude/commands/hippo-backend.md \
  $HOME/.claude/docs/specs/2026-04-28-atomic-memory-system-design.md \
  2>/dev/null
```

Note every match. Skip matches inside `docs/superpowers/specs/2026-04-28-llm-backend-toggle-design.md` and `docs/superpowers/plans/*.md` — those are historical records of past work and should not be edited.

- [ ] **Step 2: Edit each match**

For each match, the rewrite rule is mechanical:
- `--project foo` → `--scope foo`
- `--project hippo` → `--scope hippo`
- `--global-only` → `--scope global`
- `memory-stats --project hippo` → `cd ~/code/hippo && hippo stats` (or `hippo stats --scope hippo` from elsewhere)
- Any "you must pass --project" / "the --project flag" prose → describe `--scope` as an override of cwd auto-detect.

If a doc has an `## Examples` section with old-flag commands, update the commands; do not add a "what changed" subsection.

- [ ] **Step 3: Verify nothing's left**

```bash
grep -rn "\-\-project\|\-\-global-only" \
  CLAUDE.md README.md docs/ KNOWN_ISSUES.md \
  $HOME/.claude/commands/hippo-dream.md \
  $HOME/.claude/commands/hippo-backend.md \
  $HOME/.claude/docs/specs/2026-04-28-atomic-memory-system-design.md \
  2>/dev/null \
  | grep -v "docs/superpowers/"
```

Expected: no output (the historical-record exclusion stays).

- [ ] **Step 4: Commit (in-repo edits only)**

```bash
git add CLAUDE.md README.md docs/ KNOWN_ISSUES.md
git commit -m "docs: update CLI flag references for --scope rename"
```

For files outside the repo (`~/.claude/commands/*.md` and the canonical spec), they are user-machine state. Leave a note for the user to commit them separately if they version-control `~/.claude/`.

- [ ] **Step 5: Verify CLI help text matches docs**

```bash
uv run hippo dream -h
uv run hippo search -h
uv run hippo stats -h
```

Expected: every command's `--help` shows `--scope` (and where applicable `--all-scopes`); no mention of `--project` or `--global-only`.

---

## Task 13: Manual end-to-end verification

These are user-side checks — run after Task 12 lands.

- [ ] **Step 1: Auto-detect from project repo**

```bash
cd ~/code/hippo
uv run hippo stats
```

Expected: shows stats for `global` and `project:hippo`.

- [ ] **Step 2: Auto-detect from non-project dir errors**

```bash
cd /tmp
uv run hippo dream
```

Expected: exits with the standard "not in a project" error message and a non-zero return code.

- [ ] **Step 3: Explicit override**

```bash
cd /tmp
uv run hippo stats --scope global
```

Expected: stats for `global` only.

- [ ] **Step 4: All-scopes**

```bash
cd /tmp
uv run hippo stats --all-scopes
```

Expected: stats for `global` and every project under PROJECTS_ROOT.

- [ ] **Step 5: Trigger the nightly path manually**

```bash
launchctl kickstart -k "gui/$(id -u)/com.$(whoami).dream-heavy"
tail -f ~/.claude/debug/dream-heavy.log
```

Expected: heavy dream runs across global + every project with markdown captures (likely the longest run yet — see Risks section in spec).

- [ ] **Step 6: Bootstrap rejects multi-scope**

```bash
uv run hippo bootstrap --memory-dir /tmp/x --scope a --scope b 2>&1 | head -3
```

Expected: parser error about "only one --scope value allowed".

---

## Task 14: Memory pruning (post-merge)

Run **after** Tasks 1-13 are merged. Pruning before merge would orphan memory that still references valid CLI behavior.

- [ ] **Step 1: Discover stale memory bodies**

```bash
hippo search "memory-stats --project" --all-scopes
hippo search "memory-archive --project" --all-scopes
hippo search "global-only flag" --all-scopes
hippo search "dream-heavy --project" --all-scopes
```

Capture every `head_id` returned. For each, run:

```bash
hippo get <head_id>
```

and decide: is the body's *primary value* the old-flag instruction, or is it an incidental mention?

- [ ] **Step 2: Archive bodies whose primary value is old-flag instruction**

For each confirmed target:

```bash
hippo archive <head_id> --reason "stale: superseded by --scope flag rename"
```

The two known confirmed targets from the spec:

```bash
hippo archive e8e6ca4e72aa4382a6ef995c89e5d2c2 \
  --reason "stale: superseded by --scope flag rename"
hippo archive eeb7412592244c79a3f0c35bc7b732b1 \
  --reason "stale: superseded by --scope flag rename"
```

(These IDs were discovered during brainstorming on 2026-05-04. If they no longer resolve when this task runs, they were already archived or rotated — that's fine; skip.)

- [ ] **Step 3: Verify the next dream cycle catches incidental mentions**

```bash
cd ~/code/hippo
uv run hippo dream
hippo search "memory-stats --project" --all-scopes
```

Expected: zero or only-incidental hits remain. Anything still surfacing should be a body that uses old flags only as one example among many — leave it; the review phase will eventually consolidate.

- [ ] **Step 4: No commit needed**

Memory archives are in-store soft deletes, not git-tracked. Done.

---

## Final Gates

- [ ] **All tests green**

```bash
uv run pytest
```

Expected: 110+ tests pass, 2 skipped (LLM-gated). No new failures.

- [ ] **Lint and types**

```bash
uv run ruff check src tests
uv run mypy src
```

Expected: clean.

- [ ] **No old-flag references remain**

```bash
grep -rn "\-\-project\|\-\-global-only" src/ tests/ \
  CLAUDE.md README.md docs/ KNOWN_ISSUES.md \
  | grep -v "docs/superpowers/specs/2026-04-28\|docs/superpowers/plans/2026-04-30\|docs/superpowers/plans/2026-04-28"
```

Expected: empty (the excluded paths are pre-existing historical records).
