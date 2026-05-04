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
