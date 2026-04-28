"""PreCompact hook: trigger light dream for current scope."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from hippo.capture.userprompt_hook import _resolve_project
from hippo.daemon.client import DaemonClient
from hippo.dream.light import run_light_dream
from hippo.storage.multi_store import Scope


def main() -> int:
    try:
        text = sys.stdin.read()
        if text.strip():
            payload = json.loads(text)
        else:
            payload = {}
        cwd = payload.get("cwd", os.getcwd())
        project = _resolve_project(cwd)
        scope = Scope.project(project) if project else Scope.global_()

        daemon = DaemonClient(socket_path=Path.home() / ".claude" / "memory-daemon.sock")
        # Run light dream on current scope; also on global so cross-session work is consolidated
        run_light_dream(scope=Scope.global_(), daemon=daemon)
        if project:
            run_light_dream(scope=scope, daemon=daemon)
        return 0
    except Exception as e:
        sys.stderr.write(f"hippo precompact-hook error: {e}\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
