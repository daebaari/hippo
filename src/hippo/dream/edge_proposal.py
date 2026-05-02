"""Edge proposal phase: within each cluster, prompt LLM on each pair to
propose a typed edge. Skip pairs the LLM marks 'none'.

Uses LLMProto.generate_chat_batch so backends with batched primitives
(LocalLLM via mlx_lm.batch_generate + shared-prefix KV cache) can amortize
prefill across many pairs. Sequential backends (Gemini) fall back to a loop
without changing semantics.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from hippo.dream.atomize import _strip_fences
from hippo.dream.cluster import cluster_active_heads
from hippo.dream.prompts import render
from hippo.models.llm import LLMProto
from hippo.storage.edges import EdgeRecord, insert_edge_with_reciprocal
from hippo.storage.heads import get_head
from hippo.storage.multi_store import Store

VALID_RELATIONS = {"causes", "supersedes", "contradicts", "applies_when", "related"}

# Default chunk size for batched LLM calls. Conservative for ~15GB MoE on
# Apple Silicon unified memory; raise if you have headroom.
DEFAULT_BATCH_SIZE = 8


@dataclass
class _PendingPair:
    a_id: str
    b_id: str
    prompt: str


def _collect_pending_pairs(
    store: Store,
    clusters: list[list[str]],
) -> list[_PendingPair]:
    """First pass: enumerate all within-cluster pairs, drop ones already edged
    or with missing heads, render the LLM prompt for survivors."""
    pending: list[_PendingPair] = []
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                a_id, b_id = cluster[i], cluster[j]
                a = get_head(store.conn, a_id)
                b = get_head(store.conn, b_id)
                if a is None or b is None:
                    continue
                existing = store.conn.execute(
                    "SELECT 1 FROM edges "
                    "WHERE (from_head=? AND to_head=?) OR (from_head=? AND to_head=?) "
                    "LIMIT 1",
                    (a_id, b_id, b_id, a_id),
                ).fetchone()
                if existing is not None:
                    continue
                pending.append(_PendingPair(
                    a_id=a_id,
                    b_id=b_id,
                    prompt=render("edge_typing", head_a=a.summary, head_b=b.summary),
                ))
    return pending


def propose_edges(
    *,
    store: Store,
    llm: LLMProto,
    progress_cb: Callable[[int, int], None] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    clusters = cluster_active_heads(store.conn)
    pending = _collect_pending_pairs(store, clusters)
    total_pairs = len(pending)
    n_inserted = 0

    if total_pairs == 0:
        if progress_cb is not None:
            progress_cb(0, 0)
        return 0

    seen = 0
    for start in range(0, total_pairs, batch_size):
        chunk = pending[start:start + batch_size]
        message_lists = [
            [{"role": "user", "content": p.prompt}] for p in chunk
        ]
        responses = llm.generate_chat_batch(
            message_lists,
            temperature=0.1,
            max_tokens=200,
            thinking_level="minimal",
            batch_size=batch_size,
        )
        for pair, raw in zip(chunk, responses, strict=True):
            seen += 1
            try:
                obj = json.loads(_strip_fences(raw))
            except json.JSONDecodeError:
                if progress_cb is not None:
                    progress_cb(seen, total_pairs)
                continue
            relation = obj.get("relation")
            if relation not in VALID_RELATIONS:
                if progress_cb is not None:
                    progress_cb(seen, total_pairs)
                continue
            weight = float(obj.get("weight", 1.0))
            try:
                insert_edge_with_reciprocal(
                    store.conn,
                    EdgeRecord(
                        from_head=pair.a_id,
                        to_head=pair.b_id,
                        relation=relation,
                        weight=weight,
                    ),
                )
                n_inserted += 1
            except (sqlite3.IntegrityError, ValueError):
                pass
            if progress_cb is not None:
                progress_cb(seen, total_pairs)
    return n_inserted
