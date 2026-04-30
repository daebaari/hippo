"""Multi-head phase: bodies that get retrieved often but have few heads
deserve more search-affordances. LLM proposes diverse new heads."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol
from uuid import uuid4

from hippo.dream.atomize import _strip_fences  # reuse
from hippo.dream.prompts import render
from hippo.models.llm import LLMProto
from hippo.storage.body_files import read_body_file
from hippo.storage.heads import HeadRecord, insert_head, list_heads_for_body
from hippo.storage.multi_store import Store
from hippo.storage.vec import insert_head_embedding


class DaemonProto(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def expand_heads_for_eligible_bodies(
    *,
    store: Store,
    llm: LLMProto,
    daemon: DaemonProto,
    target_total_heads: int = 3,
    progress_cb: "Callable[[int, int], None] | None" = None,
) -> int:
    """For bodies with retrieval_count > 0 and < target_total_heads heads,
    generate additional diverse heads. Returns count of new heads inserted."""
    rows = store.conn.execute(
        """
        SELECT b.body_id, b.file_path, b.title, COUNT(h.head_id) AS head_count
        FROM bodies b
        LEFT JOIN heads h ON h.body_id = b.body_id AND h.archived = 0
        WHERE b.archived = 0
        GROUP BY b.body_id
        HAVING head_count < ?
          AND (
              SELECT MAX(retrieval_count) FROM heads WHERE body_id = b.body_id
          ) > 0
        """,
        (target_total_heads,),
    ).fetchall()

    total = len(rows)
    n_new = 0
    for idx, r in enumerate(rows, start=1):
        existing = list_heads_for_body(store.conn, r["body_id"])
        try:
            body = read_body_file(store.memory_dir / r["file_path"])
        except FileNotFoundError:
            if progress_cb is not None:
                progress_cb(idx, total)
            continue
        n_to_add = target_total_heads - len(existing)
        prompt = render(
            "multi_head",
            existing_heads="\n".join(f"- {h.summary}" for h in existing),
            title=r["title"],
            body=body.content,
            n=n_to_add,
        )
        raw = llm.generate_chat(
            [{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=512,
        )
        try:
            new_summaries = json.loads(_strip_fences(raw))
        except json.JSONDecodeError:
            if progress_cb is not None:
                progress_cb(idx, total)
            continue
        new_summaries = [
            s for s in new_summaries if isinstance(s, str) and s.strip()
        ][:n_to_add]
        if not new_summaries:
            if progress_cb is not None:
                progress_cb(idx, total)
            continue
        vecs = daemon.embed(new_summaries)
        for summary, vec in zip(new_summaries, vecs, strict=True):
            head_id = uuid4().hex
            insert_head(
                store.conn, HeadRecord(head_id=head_id, body_id=r["body_id"], summary=summary)
            )
            insert_head_embedding(store.conn, head_id, vec)
            n_new += 1
        if progress_cb is not None:
            progress_cb(idx, total)
    return n_new
