"""Stop hook handler: capture turn + embed for immediate retrievability."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Protocol

from hippo.capture.userprompt_hook import _resolve_project  # reuse
from hippo.daemon.client import DaemonClient
from hippo.storage.capture import CaptureRecord, enqueue_capture
from hippo.storage.multi_store import Scope, open_store
from hippo.storage.turn_embeddings import insert_turn_embedding


class DaemonClientProto(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _mechanical_summary(user_msg: str | None, asst_msg: str | None) -> str:
    parts = []
    if user_msg:
        parts.append(user_msg.strip().replace("\n", " ")[:80])
    if asst_msg:
        parts.append(asst_msg.strip().replace("\n", " ")[:80])
    return " | ".join(parts)


def handle_stop(*, stdin_text: str, daemon: DaemonClientProto | None = None) -> str:
    payload = json.loads(stdin_text)
    user_message = payload.get("user_message")
    assistant_message = payload.get("assistant_message")
    if not (user_message or assistant_message):
        return ""

    session_id = payload.get("session_id", "unknown")
    transcript_path = payload.get("transcript_path")
    cwd = payload.get("cwd", os.getcwd())
    project = _resolve_project(cwd)

    scope = Scope.project(project) if project else Scope.global_()
    if daemon is None:
        daemon = DaemonClient(socket_path=Path.home() / ".claude" / "memory-daemon.sock")

    store = open_store(scope)
    try:
        cap_id = enqueue_capture(store.conn, CaptureRecord(
            session_id=session_id, project=project,
            user_message=user_message, assistant_message=assistant_message,
            transcript_path=transcript_path,
        ))
        concat = ((user_message or "") + "\n\n" + (assistant_message or "")).strip()
        if concat:
            vec = daemon.embed([concat])[0]
            insert_turn_embedding(
                store.conn,
                capture_id=cap_id,
                summary=_mechanical_summary(user_message, assistant_message),
                embedding=vec,
            )
    finally:
        store.conn.close()
    return ""


def main() -> int:
    try:
        handle_stop(stdin_text=sys.stdin.read())
        return 0
    except Exception as e:
        sys.stderr.write(f"hippo stop-hook error: {e}\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
