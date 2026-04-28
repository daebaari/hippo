"""Step 1 of retrieval: vector search across one or more scoped stores.

Each scope's DB is queried independently; results are merged and returned
sorted by ascending distance. Scope is attached as metadata so downstream
display can show '[global]' vs '[project:X]' tags.
"""
from __future__ import annotations

from dataclasses import dataclass

from hippo.storage.heads import HeadRecord, get_head
from hippo.storage.multi_store import Scope, open_store
from hippo.storage.vec import vector_search_heads


@dataclass
class VectorHit:
    head_id: str
    distance: float
    scope: str
    head: HeadRecord


def vector_search_all_scopes(
    *, scopes: list[Scope], query: list[float], top_k_per_scope: int,
) -> list[VectorHit]:
    hits: list[VectorHit] = []
    for scope in scopes:
        store = open_store(scope)
        try:
            for r in vector_search_heads(store.conn, query, top_k=top_k_per_scope):
                head = get_head(store.conn, r.head_id)
                if head is None or head.archived:
                    continue
                hits.append(VectorHit(
                    head_id=r.head_id,
                    distance=r.distance,
                    scope=scope.as_string(),
                    head=head,
                ))
        finally:
            store.conn.close()
    hits.sort(key=lambda h: h.distance)
    return hits
