"""Tests for the rerank step. Uses a mock daemon client."""
from __future__ import annotations

from dataclasses import dataclass

from hippo.retrieval.graph_expand import GraphHit
from hippo.retrieval.rerank import rerank_candidates
from hippo.storage.heads import HeadRecord


@dataclass
class FakeRerankClient:
    scores_by_pair: dict[tuple[str, str], float]

    def rerank(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self.scores_by_pair[p] for p in pairs]


def _make_hit(
    head_id: str,
    summary: str,
    edge_relation: str | None = None,
    distance: float = 0.0,
) -> GraphHit:
    head = HeadRecord(head_id=head_id, body_id="b", summary=summary)
    return GraphHit(
        head_id=head_id,
        distance=distance,
        scope="global",
        head=head,
        edge_relation=edge_relation,
    )


def test_rerank_returns_top_k_by_boosted_score() -> None:
    a = _make_hit("h1", "apples")
    b = _make_hit("h2", "oranges")
    c = _make_hit("h3", "carbon", edge_relation="contradicts")
    client = FakeRerankClient({
        ("query", "apples"): 0.5,
        ("query", "oranges"): 0.3,
        ("query", "carbon"): 0.4,  # before boost: 0.4; after contradicts boost (1.3x): 0.52
    })
    out = rerank_candidates(query="query", candidates=[a, b, c], client=client, top_k=2)
    assert [h.head_id for h in out] == ["h3", "h1"]


def test_rerank_handles_empty() -> None:
    client = FakeRerankClient({})
    assert rerank_candidates(query="q", candidates=[], client=client, top_k=10) == []
