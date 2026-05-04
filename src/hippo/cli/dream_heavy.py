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
