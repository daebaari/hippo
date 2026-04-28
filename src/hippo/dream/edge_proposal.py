"""Edge proposal phase: within each cluster, prompt LLM on each pair to
propose a typed edge. Skip pairs the LLM marks 'none'."""
from __future__ import annotations

import json
import sqlite3
from typing import Protocol

from hippo.dream.atomize import _strip_fences
from hippo.dream.cluster import cluster_active_heads
from hippo.dream.prompts import render
from hippo.storage.edges import EdgeRecord, insert_edge_with_reciprocal
from hippo.storage.heads import get_head
from hippo.storage.multi_store import Store


class LLMProto(Protocol):
    def generate_chat(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
    ) -> str: ...


VALID_RELATIONS = {"causes", "supersedes", "contradicts", "applies_when", "related"}


def propose_edges(*, store: Store, llm: LLMProto) -> int:
    clusters = cluster_active_heads(store.conn)
    n_inserted = 0
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                a_id, b_id = cluster[i], cluster[j]
                a = get_head(store.conn, a_id)
                b = get_head(store.conn, b_id)
                if a is None or b is None:
                    continue
                # Skip if any directed edge between them already exists
                existing = store.conn.execute(
                    "SELECT 1 FROM edges "
                    "WHERE (from_head=? AND to_head=?) OR (from_head=? AND to_head=?) "
                    "LIMIT 1",
                    (a_id, b_id, b_id, a_id),
                ).fetchone()
                if existing is not None:
                    continue

                prompt = render("edge_typing", head_a=a.summary, head_b=b.summary)
                raw = llm.generate_chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200,
                )
                try:
                    obj = json.loads(_strip_fences(raw))
                except json.JSONDecodeError:
                    continue
                relation = obj.get("relation")
                if relation not in VALID_RELATIONS:
                    continue
                weight = float(obj.get("weight", 1.0))
                try:
                    insert_edge_with_reciprocal(
                        store.conn,
                        EdgeRecord(from_head=a_id, to_head=b_id, relation=relation, weight=weight),
                    )
                    n_inserted += 1
                except (sqlite3.IntegrityError, ValueError):
                    continue
    return n_inserted
