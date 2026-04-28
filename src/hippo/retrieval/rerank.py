"""Step 3 of retrieval: cross-encoder rerank with edge-type boosts.

Takes the merged (vector + graph) candidate set and a query string;
calls the daemon's rerank API to score (query, head.summary) pairs;
applies an EDGE_BOOST multiplier when the candidate came from a graph hop;
returns top_k by final score (descending).
"""
from __future__ import annotations

from typing import Protocol

from hippo.config import EDGE_BOOST
from hippo.retrieval.graph_expand import GraphHit


class RerankClient(Protocol):
    def rerank(self, pairs: list[tuple[str, str]]) -> list[float]: ...


def rerank_candidates(
    *, query: str, candidates: list[GraphHit], client: RerankClient, top_k: int,
) -> list[GraphHit]:
    if not candidates:
        return []
    pairs = [(query, c.head.summary) for c in candidates]
    raw_scores = client.rerank(pairs)

    boosted: list[tuple[float, GraphHit]] = []
    for c, s in zip(candidates, raw_scores, strict=True):
        boost = EDGE_BOOST.get(c.edge_relation, 1.0) if c.edge_relation else 1.0
        boosted.append((s * boost, c))

    boosted.sort(key=lambda pair: -pair[0])
    return [hit for _, hit in boosted[:top_k]]
