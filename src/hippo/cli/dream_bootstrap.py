"""One-time bootstrap: atomize legacy markdown memory into the new Hippo schema.

Idempotent — re-running atomizes any files left in the legacy directory and
moves them to ``.legacy/<timestamp>/`` after consolidation phases complete.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from hippo.cli.scope_args import add_scope_args, resolve_scopes
from hippo.config import HEAVY_LOCK_FILENAME, ConfigError
from hippo.daemon.client import DaemonClient
from hippo.dream.bootstrap import atomize_legacy_files
from hippo.dream.contradiction import resolve_contradictions
from hippo.dream.edge_proposal import propose_edges
from hippo.dream.multi_head import expand_heads_for_eligible_bodies
from hippo.lock import LockHeldError, acquire_lock, release_lock
from hippo.models.llm import select_llm
from hippo.storage.multi_store import Scope, open_store


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

    legacy_dir = Path(args.memory_dir).expanduser()
    if not legacy_dir.exists():
        print(f"ERROR: {legacy_dir} not found", file=sys.stderr)
        return 1

    scopes = resolve_scopes(args, kind="single_scope_write", cwd=os.getcwd())
    # single_scope_write returns exactly one scope; bootstrap requires it to be a
    # project scope (global has no legacy markdown to atomize).
    target_scope = scopes[0]
    if target_scope.kind != "project" or target_scope.project_name is None:
        sys.stderr.write(
            "ERROR: dream-bootstrap requires a project scope, got 'global'\n"
        )
        return 1
    project_name: str = target_scope.project_name

    daemon = DaemonClient(socket_path=Path.home() / ".claude" / "memory-daemon.sock")

    # Open both stores and acquire heavy-dream locks before any LLM/atomize work.
    g_scope = Scope.global_()
    p_scope = target_scope
    g_store = open_store(g_scope)
    p_store = open_store(p_scope)

    try:
        g_handle = acquire_lock(g_store.memory_dir / HEAVY_LOCK_FILENAME)
    except LockHeldError:
        print(
            f"ABORT: {g_scope.as_string()} heavy-lock held by another process",
            file=sys.stderr,
        )
        g_store.conn.close()
        p_store.conn.close()
        return 1

    try:
        p_handle = acquire_lock(p_store.memory_dir / HEAVY_LOCK_FILENAME)
    except LockHeldError:
        print(
            f"ABORT: {p_scope.as_string()} heavy-lock held by another process",
            file=sys.stderr,
        )
        release_lock(g_handle)
        g_store.conn.close()
        p_store.conn.close()
        return 1

    try:
        try:
            llm = select_llm(strict=args.strict)
        except ConfigError as exc:
            sys.stderr.write(f"ERROR: {exc}\n")
            release_lock(g_handle)
            release_lock(p_handle)
            g_store.conn.close()
            p_store.conn.close()
            return 1

        # Atomize files
        n = atomize_legacy_files(
            legacy_dir=legacy_dir, project=project_name, llm=llm, daemon=daemon
        )
        print(f"Atomized {n} bodies from {legacy_dir}")

        # Run consolidation phases on the bootstrapped corpus, reusing already-opened stores.
        for scope, store in [(g_scope, g_store), (p_scope, p_store)]:
            heads_added = expand_heads_for_eligible_bodies(store=store, llm=llm, daemon=daemon)
            edges_added = propose_edges(store=store, llm=llm)
            contradictions = resolve_contradictions(store=store, llm=llm)
            print(
                f"{scope.as_string()}: +{heads_added} heads, +{edges_added} edges, "
                f"{contradictions} contradictions resolved"
            )
    finally:
        release_lock(g_handle)
        release_lock(p_handle)
        g_store.conn.close()
        p_store.conn.close()

    # Archive legacy files (no SQLite writes, safe outside lock).
    if not args.no_archive:
        ts = time.strftime("%Y%m%d-%H%M%S")
        archive_dir = legacy_dir / ".legacy" / ts
        archive_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        failures: list[tuple[Path, str]] = []
        for md in legacy_dir.glob("*.md"):
            if md.parent == archive_dir or ".legacy" in md.parts:
                continue
            try:
                shutil.move(str(md), str(archive_dir / md.name))
            except OSError as exc:
                failures.append((md, str(exc)))
            else:
                moved += 1
        print(f"Moved {moved} legacy files \u2192 {archive_dir}")
        if failures:
            print(f"FAILED to move {len(failures)} files:", file=sys.stderr)
            for path, reason in failures:
                print(f"  {path}: {reason}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
