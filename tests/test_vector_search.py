"""Tests for the dual-DB vector search step."""
from __future__ import annotations

from hippo.config import EMBEDDING_DIM
from hippo.retrieval.vector_search import vector_search_all_scopes
from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.heads import HeadRecord, insert_head
from hippo.storage.multi_store import Scope, open_store
from hippo.storage.vec import insert_head_embedding


def _populate(store, body_id: str, head_id: str, summary: str, vec: list[float]) -> None:
    insert_body(store.conn, BodyRecord(
        body_id=body_id, file_path=f"bodies/{body_id}.md",
        title=summary, scope=store.scope.as_string(), source="manual",
    ))
    insert_head(store.conn, HeadRecord(head_id=head_id, body_id=body_id, summary=summary))
    insert_head_embedding(store.conn, head_id, vec)


def test_search_merges_global_and_project(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    monkeypatch.setattr("hippo.config.PROJECTS_ROOT", tmp_path / "projects")

    g = open_store(Scope.global_())
    p = open_store(Scope.project("kaleon"))

    v_close = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    v_far   = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)

    _populate(g, "g1", "gh1", "global head", v_close)
    _populate(p, "p1", "ph1", "project head", v_far)

    results = vector_search_all_scopes(
        scopes=[Scope.global_(), Scope.project("kaleon")],
        query=v_close, top_k_per_scope=5,
    )
    assert len(results) == 2
    # Closest first
    assert results[0].head_id == "gh1"
    assert results[0].scope == "global"
    assert results[1].head_id == "ph1"
    assert results[1].scope == "project:kaleon"


def test_top_k_per_scope_caps_results(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    g = open_store(Scope.global_())
    for i in range(10):
        v = [float(i == j) for j in range(EMBEDDING_DIM)]
        _populate(g, f"b{i}", f"h{i}", f"summary {i}", v)
    q = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
    results = vector_search_all_scopes(
        scopes=[Scope.global_()], query=q, top_k_per_scope=3,
    )
    assert len(results) == 3
