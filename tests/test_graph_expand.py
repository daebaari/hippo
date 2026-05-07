"""Tests for graph 1-hop candidate expansion."""
from __future__ import annotations

from hippo.retrieval.graph_expand import expand_via_graph
from hippo.retrieval.vector_search import VectorHit
from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.edges import EdgeRecord, insert_edge
from hippo.storage.heads import HeadRecord, insert_head
from hippo.storage.multi_store import Scope, open_store


def test_expand_adds_neighbors_via_edges(tmp_path, monkeypatch):
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
    for hid in ["h1", "h2", "h3"]:
        insert_head(s.conn, HeadRecord(head_id=hid, body_id="b1", summary=hid))
    insert_edge(s.conn, EdgeRecord(from_head="h1", to_head="h2", relation="causes"))
    insert_edge(s.conn, EdgeRecord(from_head="h1", to_head="h3", relation="related"))
    s.conn.close()

    seed = [
        VectorHit(
            head_id="h1",
            distance=0.1,
            scope="global",
            head=HeadRecord(head_id="h1", body_id="b1", summary="h1"),
        )
    ]
    expanded = expand_via_graph(seed, scopes=[Scope.global_()], hop_limit_per_seed=5, total_cap=20)
    head_ids = {hit.head_id for hit in expanded}
    assert "h1" in head_ids and "h2" in head_ids and "h3" in head_ids


def test_expand_prefers_higher_weight_within_same_relation(tmp_path, monkeypatch):
    """Within a relation class, higher-weight edges must outrank lower ones.

    Regression: graph_expand sorted only by EDGE_BOOST[relation]; Python's
    stable sort then preserved DB insertion order within each relation, so
    a weight-0.4 'related' edge inserted earlier would beat a weight-0.9
    one inserted later.
    """
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
    insert_head(s.conn, HeadRecord(head_id="seed", body_id="b1", summary="seed"))
    insert_head(s.conn, HeadRecord(head_id="weak", body_id="b1", summary="weak"))
    insert_head(s.conn, HeadRecord(head_id="strong", body_id="b1", summary="strong"))
    # Insert weak edge FIRST so insertion order favors it under the bug.
    insert_edge(
        s.conn,
        EdgeRecord(from_head="seed", to_head="weak", relation="related", weight=0.4),
    )
    insert_edge(
        s.conn,
        EdgeRecord(from_head="seed", to_head="strong", relation="related", weight=0.9),
    )
    s.conn.close()

    seed = [
        VectorHit(
            head_id="seed",
            distance=0.1,
            scope="global",
            head=HeadRecord(head_id="seed", body_id="b1", summary="seed"),
        )
    ]
    expanded = expand_via_graph(
        seed, scopes=[Scope.global_()], hop_limit_per_seed=1, total_cap=10
    )
    neighbor_ids = [hit.head_id for hit in expanded if hit.head_id != "seed"]
    assert neighbor_ids == ["strong"], (
        f"expected the higher-weight neighbor first, got {neighbor_ids}"
    )


def test_expand_typed_edge_beats_higher_weight_related(tmp_path, monkeypatch):
    """Typed relations (causes/applies_when/...) still outrank 'related'.

    Even if a 'related' edge has higher raw weight, a typed edge should
    win because EDGE_BOOST is the primary sort key.
    """
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
    insert_head(s.conn, HeadRecord(head_id="seed", body_id="b1", summary="seed"))
    insert_head(s.conn, HeadRecord(head_id="rel_high", body_id="b1", summary="rel"))
    insert_head(s.conn, HeadRecord(head_id="typed_low", body_id="b1", summary="typed"))
    insert_edge(
        s.conn,
        EdgeRecord(from_head="seed", to_head="rel_high", relation="related", weight=0.95),
    )
    insert_edge(
        s.conn,
        EdgeRecord(from_head="seed", to_head="typed_low", relation="causes", weight=0.5),
    )
    s.conn.close()

    seed = [
        VectorHit(
            head_id="seed",
            distance=0.1,
            scope="global",
            head=HeadRecord(head_id="seed", body_id="b1", summary="seed"),
        )
    ]
    expanded = expand_via_graph(
        seed, scopes=[Scope.global_()], hop_limit_per_seed=1, total_cap=10
    )
    neighbor_ids = [hit.head_id for hit in expanded if hit.head_id != "seed"]
    assert neighbor_ids == ["typed_low"], (
        f"typed edge should win on EDGE_BOOST, got {neighbor_ids}"
    )


def test_expand_caps_total_added(tmp_path, monkeypatch):
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
    insert_head(s.conn, HeadRecord(head_id="h0", body_id="b1", summary="seed"))
    for i in range(50):
        insert_head(s.conn, HeadRecord(head_id=f"n{i}", body_id="b1", summary=f"n{i}"))
        insert_edge(s.conn, EdgeRecord(from_head="h0", to_head=f"n{i}", relation="related"))
    s.conn.close()

    seed = [
        VectorHit(
            head_id="h0",
            distance=0.1,
            scope="global",
            head=HeadRecord(head_id="h0", body_id="b1", summary="seed"),
        )
    ]
    expanded = expand_via_graph(seed, scopes=[Scope.global_()], hop_limit_per_seed=10, total_cap=15)
    # Max should be seed (1) + 14 neighbors = 15
    assert len(expanded) <= 15
