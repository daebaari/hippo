"""Tests for the edge proposal phase (cluster + LLM edge typing)."""
from __future__ import annotations

from pathlib import Path

import pytest

from hippo.config import EMBEDDING_DIM
from hippo.dream.edge_proposal import propose_edges
from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.heads import HeadRecord, insert_head
from hippo.storage.multi_store import Scope, open_store
from hippo.storage.vec import insert_head_embedding


class StubLLM:
    """Always returns a 'causes' edge."""

    def generate_chat(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
    ) -> str:
        return '{"relation":"causes","weight":0.8}'


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    return open_store(Scope.global_())


def _make_embedding(first: float, second: float = 0.0) -> list[float]:
    """Return a unit-ish vector with given first two components."""
    vec = [0.0] * EMBEDDING_DIM
    vec[0] = first
    vec[1] = second
    return vec


def test_propose_edges_creates_edge_for_similar_heads(store: object) -> None:
    """Two heads with cosine >= 0.7 should get one edge proposed."""
    # Insert a body
    insert_body(
        store.conn,  # type: ignore[attr-defined]
        BodyRecord(
            body_id="body-1",
            file_path="bodies/body-1.md",
            title="Test body",
            scope="global",
            source="test",
        ),
    )

    # Insert two heads
    insert_head(
        store.conn,  # type: ignore[attr-defined]
        HeadRecord(head_id="head-a", body_id="body-1", summary="Python is fast"),
    )
    insert_head(
        store.conn,  # type: ignore[attr-defined]
        HeadRecord(head_id="head-b", body_id="body-1", summary="Python runs quickly"),
    )

    # [1.0, 0, ...] and [0.99, 0.14, 0, ...] — cosine ≈ 0.99, well above threshold
    emb_a = _make_embedding(1.0, 0.0)
    emb_b = _make_embedding(0.99, 0.14)

    insert_head_embedding(store.conn, "head-a", emb_a)  # type: ignore[attr-defined]
    insert_head_embedding(store.conn, "head-b", emb_b)  # type: ignore[attr-defined]

    count = propose_edges(store=store, llm=StubLLM())  # type: ignore[arg-type]

    assert count == 1

    row = store.conn.execute(  # type: ignore[attr-defined]
        "SELECT * FROM edges WHERE from_head='head-a' AND to_head='head-b'"
    ).fetchone()
    assert row is not None
    assert row["relation"] == "causes"
    assert abs(float(row["weight"]) - 0.8) < 1e-6


def test_propose_edges_skips_dissimilar_heads(store: object) -> None:
    """Two heads with cosine < 0.7 should produce no edges."""
    insert_body(
        store.conn,  # type: ignore[attr-defined]
        BodyRecord(
            body_id="body-2",
            file_path="bodies/body-2.md",
            title="Dissimilar body",
            scope="global",
            source="test",
        ),
    )

    insert_head(
        store.conn,  # type: ignore[attr-defined]
        HeadRecord(head_id="head-c", body_id="body-2", summary="Cats are mammals"),
    )
    insert_head(
        store.conn,  # type: ignore[attr-defined]
        HeadRecord(head_id="head-d", body_id="body-2", summary="Quantum entanglement"),
    )

    # Orthogonal embeddings — cosine = 0.0
    emb_c = _make_embedding(1.0, 0.0)
    emb_d = _make_embedding(0.0, 1.0)

    insert_head_embedding(store.conn, "head-c", emb_c)  # type: ignore[attr-defined]
    insert_head_embedding(store.conn, "head-d", emb_d)  # type: ignore[attr-defined]

    count = propose_edges(store=store, llm=StubLLM())  # type: ignore[arg-type]

    assert count == 0


def test_propose_edges_skips_existing_edges(store: object) -> None:
    """If an edge already exists between a pair, don't call LLM or insert again."""
    insert_body(
        store.conn,  # type: ignore[attr-defined]
        BodyRecord(
            body_id="body-3",
            file_path="bodies/body-3.md",
            title="Pre-existing edge body",
            scope="global",
            source="test",
        ),
    )

    insert_head(
        store.conn,  # type: ignore[attr-defined]
        HeadRecord(head_id="head-e", body_id="body-3", summary="Head E"),
    )
    insert_head(
        store.conn,  # type: ignore[attr-defined]
        HeadRecord(head_id="head-f", body_id="body-3", summary="Head F"),
    )

    emb = _make_embedding(1.0, 0.0)
    insert_head_embedding(store.conn, "head-e", emb)  # type: ignore[attr-defined]
    insert_head_embedding(store.conn, "head-f", emb)  # type: ignore[attr-defined]

    # Pre-insert an edge
    store.conn.execute(  # type: ignore[attr-defined]
        "INSERT INTO edges(from_head, to_head, relation, weight, created_at) "
        "VALUES ('head-e', 'head-f', 'related', 1.0, 1000)"
    )
    store.conn.commit()  # type: ignore[attr-defined]

    count = propose_edges(store=store, llm=StubLLM())  # type: ignore[arg-type]

    assert count == 0
