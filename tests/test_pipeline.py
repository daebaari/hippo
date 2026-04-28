"""End-to-end pipeline test with mock daemon."""
from __future__ import annotations

from dataclasses import dataclass, field

from hippo.config import EMBEDDING_DIM
from hippo.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.heads import HeadRecord, insert_head
from hippo.storage.multi_store import Scope, open_store
from hippo.storage.vec import insert_head_embedding


@dataclass
class FakeDaemon:
    """Produces a deterministic embedding (one-hot of len modulo dim) and rerank scores
    proportional to text length match. Records calls for assertions."""
    embed_calls: list[list[str]] = field(default_factory=list)
    rerank_calls: list[list[tuple[str, str]]] = field(default_factory=list)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        out = []
        for t in texts:
            v = [0.0] * EMBEDDING_DIM
            v[len(t) % EMBEDDING_DIM] = 1.0
            out.append(v)
        return out

    def rerank(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.rerank_calls.append(list(pairs))
        return [1.0 / (1 + abs(len(q) - len(d))) for q, d in pairs]


def test_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    insert_body(
        s.conn,
        BodyRecord(
            body_id="b1",
            file_path="bodies/b1.md",
            title="t",
            scope="global",
            source="manual",
        ),
    )
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

    assert isinstance(result, RetrievalResult)
    assert len(result.heads) <= 2

    # Daemon must have been called for embedding the query.
    assert daemon.embed_calls == [["apple"]]

    # Daemon must have been called at least once for rerank, and the pairs must
    # include the apple-matching head (h1's summary is "apple").
    assert len(daemon.rerank_calls) >= 1
    rerank_pairs = daemon.rerank_calls[0]
    assert ("apple", "apple") in rerank_pairs

    # Ordering: query "apple" (len=5) one-hot at index 5; h1 summary "apple" same vector
    # -> distance 0 from sqlite-vec. Rerank prefers length-match -> h1 ranks first.
    assert len(result.heads) >= 1
    assert result.heads[0].head_id == "h1"


def test_pipeline_empty_message_short_circuits(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")

    daemon = FakeDaemon()
    pipeline = RetrievalPipeline(
        daemon=daemon,
        scopes=[Scope.global_()],
        vector_top_k_per_scope=10,
        hop_limit_per_seed=5,
        total_cap=20,
        rerank_top_k=2,
    )

    for message in ("", "   "):
        result = pipeline.run(user_message=message)
        assert isinstance(result, RetrievalResult)
        assert result.heads == []
        assert result.user_message == message

    # Daemon must never have been called when the message is empty/whitespace.
    assert daemon.embed_calls == []
    assert daemon.rerank_calls == []
