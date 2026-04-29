"""Contradiction resolution: confirm and archive losers."""
from __future__ import annotations

import json

from hippo.dream.atomize import _strip_fences
from hippo.dream.prompts import render
from hippo.models.llm import LLMProto
from hippo.storage.bodies import archive_body, get_body
from hippo.storage.body_files import read_body_file
from hippo.storage.heads import archive_head, get_head, list_heads_for_body
from hippo.storage.multi_store import Store


def resolve_contradictions(*, store: Store, llm: LLMProto) -> int:
    """For each contradicts-edge pair, ask the LLM to pick the current one and archive the loser.

    Returns count archived.
    """
    pair_rows = store.conn.execute(
        """
        SELECT DISTINCT e1.from_head AS a, e1.to_head AS b
        FROM edges e1
        WHERE e1.relation = 'contradicts'
          AND e1.from_head < e1.to_head  -- avoid double-processing reciprocal pairs
        """
    ).fetchall()

    n_archived = 0
    for r in pair_rows:
        a_head = get_head(store.conn, r["a"])
        b_head = get_head(store.conn, r["b"])
        if a_head is None or b_head is None or a_head.archived or b_head.archived:
            continue
        a_body = get_body(store.conn, a_head.body_id)
        b_body = get_body(store.conn, b_head.body_id)
        if a_body is None or b_body is None or a_body.archived or b_body.archived:
            continue

        try:
            a_md = read_body_file(store.memory_dir / a_body.file_path).content
            b_md = read_body_file(store.memory_dir / b_body.file_path).content
        except FileNotFoundError:
            continue

        prompt = render(
            "contradiction",
            a_body_id=a_body.body_id,
            a_updated=a_body.updated_at,
            a_body=a_md,
            b_body_id=b_body.body_id,
            b_updated=b_body.updated_at,
            b_body=b_md,
        )
        raw = llm.generate_chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
            thinking_level="minimal",
        )
        try:
            obj = json.loads(_strip_fences(raw))
        except json.JSONDecodeError:
            continue
        if not obj.get("contradicts"):
            continue
        winner = obj.get("current_body_id")
        loser = b_body.body_id if winner == a_body.body_id else a_body.body_id
        if winner not in (a_body.body_id, b_body.body_id):
            continue

        # Archive loser body + all its active heads
        archive_body(store.conn, loser, reason=f"contradicted_by:{winner}", in_favor_of=winner)
        loser_heads = list_heads_for_body(store.conn, loser)
        for h in loser_heads:
            archive_head(store.conn, h.head_id, reason=f"body_archived:contradicted_by:{winner}")
        n_archived += 1
    return n_archived
