"""Stop hook handler: capture turn + embed for immediate retrievability.

Claude Code's Stop hook envelope provides:
- ``session_id``, ``cwd``, ``transcript_path``, ``hook_event_name="Stop"``
- ``last_assistant_message``: extracted text of the most recent assistant turn
- ``stop_hook_active``, ``permission_mode``: control fields

The user message is NOT in the envelope. We read it from the JSONL
transcript at ``transcript_path`` (the last ``type=user`` line). We also
accept legacy ``user_message`` / ``assistant_message`` fields so unit
tests can drive the handler without a real transcript file.
"""
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


def _read_last_user_message(transcript_path: str | None) -> str | None:
    """Walk the JSONL transcript backwards; return the most recent user message text."""
    if not transcript_path:
        return None
    p = Path(transcript_path)
    if not p.exists():
        return None
    try:
        lines = p.read_text().splitlines()
    except OSError:
        return None
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "user":
            continue
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        # User messages with attachments arrive as list-of-blocks
        if isinstance(content, list):
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            joined = "\n".join(t for t in text_parts if t)
            if joined:
                return joined
    return None


def handle_stop(*, stdin_text: str, daemon: DaemonClientProto | None = None) -> str:
    payload = json.loads(stdin_text)

    # Real Claude Code envelope uses last_assistant_message + transcript_path.
    # Legacy fields (user_message / assistant_message) are still accepted so
    # unit tests can drive this without writing a transcript file.
    assistant_message = (
        payload.get("assistant_message") or payload.get("last_assistant_message")
    )
    user_message = payload.get("user_message")
    if user_message is None:
        user_message = _read_last_user_message(payload.get("transcript_path"))

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
