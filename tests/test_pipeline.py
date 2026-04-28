"""End-to-end pipeline test with mock daemon."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from hippo.config import EMBEDDING_DIM
from hippo.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.heads import HeadRecord, insert_head
from hippo.storage.multi_store import Scope, open_store
from hippo.storage.vec import insert_head_embedding


@dataclass
class FakeDaemon:
    """Produces a deterministic embedding (one-hot of len modulo dim) and rerank scores
    proportional to text length match."""
    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * EMBEDDING_DIM
            v[len(t) % EMBEDDING_DIM] = 1.0
            out.append(v)
        return out

    def rerank(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [1.0 / (1 + abs(len(q) - len(d))) for q, d in pairs]


def test_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    insert_body(s.conn, BodyRecord(body_id="b1", file_path="bodies/b1.md", title="t", scope="global", source="manual"))
    for hid, summary in [("h1", "apple"), ("h2", "orange"), ("h3", "banana ")]:
        insert_head(s.conn, HeadRecord(head_id=hid, body_id="b1", summary=summary))
        v = [0.0] * EMBEDDING_DIM
        v[len(summary) % EMBEDDING_DIM] = 1.0
        insert_head_embedding(s.conn, hid, v)
    s.conn.close()

    daemon = FakeDaemon()
    pipeline = RetrievalPipeline(
        daemon=daemon,
        scopes=[Scope.global_()],
        vector_top_k_per_scope=10,
        hop_limit_per_seed=5,
        total_cap=20,
        rerank_top_k=2,
    )
    result = pipeline.run(user_message="apple")
    assert len(result.heads) <= 2
    assert isinstance(result, RetrievalResult)
