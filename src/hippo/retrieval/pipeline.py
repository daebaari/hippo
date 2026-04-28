"""End-to-end retrieval pipeline.

Embed user message -> vector top-K per scope -> graph 1-hop -> rerank -> top-K.
Returns a structured result that includes head metadata, relation tags,
and scope info so downstream injection can format nicely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hippo.retrieval.graph_expand import GraphHit, expand_via_graph
from hippo.retrieval.rerank import rerank_candidates
from hippo.retrieval.vector_search import vector_search_all_scopes
from hippo.storage.multi_store import Scope


class DaemonClientProto(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def rerank(self, pairs: list[tuple[str, str]]) -> list[float]: ...


@dataclass
class RetrievalResult:
    heads: list[GraphHit]
    user_message: str


@dataclass
class RetrievalPipeline:
    daemon: DaemonClientProto
    scopes: list[Scope]
    vector_top_k_per_scope: int
    hop_limit_per_seed: int
    total_cap: int
    rerank_top_k: int

    def run(self, user_message: str) -> RetrievalResult:
        if not user_message.strip():
            return RetrievalResult(heads=[], user_message=user_message)
        query_vec = self.daemon.embed([user_message])[0]
        seeds = vector_search_all_scopes(
            scopes=self.scopes,
            query=query_vec,
            top_k_per_scope=self.vector_top_k_per_scope,
        )
        expanded = expand_via_graph(
            seeds, scopes=self.scopes,
            hop_limit_per_seed=self.hop_limit_per_seed,
            total_cap=self.total_cap,
        )
        final = rerank_candidates(
            query=user_message, candidates=expanded, client=self.daemon,
            top_k=self.rerank_top_k,
        )
        return RetrievalResult(heads=final, user_message=user_message)
