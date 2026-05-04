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

    def __call__(
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
