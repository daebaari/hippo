"""Atomize phase: read session captures, prompt LLM, write bodies+heads."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from hippo.dream.prompts import render
from hippo.models.llm import LLMProto
from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.body_files import BodyFile, write_body_file
from hippo.storage.heads import HeadRecord, insert_head
from hippo.storage.multi_store import Store
from hippo.storage.vec import insert_head_embedding


class DaemonProto(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # remove first fence line
        s = s.split("\n", 1)[1] if "\n" in s else s
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    return s.strip()


def atomize_session(
    *, store: Store, session_id: str, project: str | None, run_id: int,
    llm: LLMProto, daemon: DaemonProto,
) -> int:
    """Atomize one session's captures. Returns count of bodies created."""
    cap_rows = store.conn.execute(
        "SELECT * FROM capture_queue"
        " WHERE session_id = ? AND processed_at IS NULL ORDER BY created_at",
        (session_id,),
    ).fetchall()
    if not cap_rows:
        return 0

    transcript_lines = []
    for r in cap_rows:
        if r["user_message"]:
            transcript_lines.append(f"USER: {r['user_message']}")
        if r["assistant_message"]:
            transcript_lines.append(f"ASSISTANT: {r['assistant_message']}")
    transcript = "\n\n".join(transcript_lines)

    prompt = render(
        "atomize",
        project=project or "",
        session_id=session_id,
        transcript=transcript,
    )
    raw = llm.generate_chat(
        [{"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=4096,
    )
    try:
        atoms = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return 0  # LLM didn't return valid JSON; skip this session for now

    n_bodies = 0
    for atom in atoms:
        title = atom.get("title", "")[:120]
        body_content = atom.get("body", "")
        heads = atom.get("heads", [])
        if not title or not body_content or not heads:
            continue

        # If LLM said scope='global' but we're processing a project store, still write to project
        # store (user can rebalance later via promote/demote — Plan 6 doesn't include those).
        body_id = uuid4().hex
        now = datetime.now(UTC)
        write_body_file(store.memory_dir, BodyFile(
            body_id=body_id, title=title, scope=store.scope.as_string(),
            created=now, updated=now, content=body_content,
        ))
        insert_body(store.conn, BodyRecord(
            body_id=body_id, file_path=f"bodies/{body_id}.md",
            title=title, scope=store.scope.as_string(),
            source=f"heavy-dream-run:{run_id}",
        ))
        # Embed all heads in one batched call
        head_summaries = [h for h in heads if isinstance(h, str) and h.strip()][:5]
        if head_summaries:
            vecs = daemon.embed(head_summaries)
            for summary, vec in zip(head_summaries, vecs, strict=True):
                head_id = uuid4().hex
                insert_head(
                    store.conn, HeadRecord(head_id=head_id, body_id=body_id, summary=summary)
                )
                insert_head_embedding(store.conn, head_id, vec)
        n_bodies += 1

    return n_bodies
