"""dream-status: print the most recent dream run across all scope DBs."""
from __future__ import annotations

import argparse
import sys
import time

from hippo import config
from hippo.config import DB_FILENAME
from hippo.storage.dream_runs import (
    DreamRunRecord,
    get_recent_runs,
    get_running_run,
)
from hippo.storage.multi_store import Scope, open_store


def _all_scopes() -> list[tuple[Scope, str]]:
    """Return (scope, display_name) pairs for every scope DB on disk."""
    scopes: list[tuple[Scope, str]] = [(Scope.global_(), "global")]
    projects_root = config.PROJECTS_ROOT
    if projects_root.exists():
        for entry in sorted(projects_root.iterdir()):
            if (entry / "memory" / DB_FILENAME).exists():
                scopes.append((Scope.project(entry.name), entry.name))
    return scopes


def _filtered_scopes(scope_name: str | None) -> list[tuple[Scope, str]]:
    if scope_name is None:
        return _all_scopes()
    if scope_name == "global":
        return [(Scope.global_(), "global")]
    return [(Scope.project(scope_name), scope_name)]


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


def dream_status_cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dream-status")
    p.add_argument("--scope", default=None,
                   help="Restrict to one scope (e.g. 'global' or a project name).")
    args = p.parse_args(argv)
    now = int(time.time())

    # First pass: any running run anywhere?
    for scope, name in _filtered_scopes(args.scope):
        store = open_store(scope)
        try:
            running = get_running_run(store.conn)
            if running is not None:
                print(render_run_line(running, scope_name=name, now_unix=now))
                return 0
        finally:
            store.conn.close()

    # Fallback: most recent completed/failed run across scopes.
    best: tuple[DreamRunRecord, str] | None = None
    for scope, name in _filtered_scopes(args.scope):
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
