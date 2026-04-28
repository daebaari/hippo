"""Tests for context injection format."""
from __future__ import annotations

from hippo.retrieval.graph_expand import GraphHit
from hippo.retrieval.inject import format_memory_block, load_body_preview
from hippo.retrieval.pipeline import RetrievalResult
from hippo.storage.body_files import BodyFile, write_body_file
from hippo.storage.heads import HeadRecord


def _hit(head_id: str, body_id: str, summary: str, scope: str) -> GraphHit:
    return GraphHit(
        head_id=head_id, distance=0.1, scope=scope,
        head=HeadRecord(head_id=head_id, body_id=body_id, summary=summary),
        edge_relation=None,
    )


def test_format_memory_block_dedupes_by_body(tmp_path):
    from datetime import UTC, datetime
    body = BodyFile(
        body_id="b1", title="Kalshi fees", scope="project:kaleon",
        created=datetime.now(UTC), updated=datetime.now(UTC),
        content="Fee is 2c at typical prices and grows to ...",
    )
    bodies_dir = tmp_path / "global"
    write_body_file(bodies_dir, body)

    hits = [
        _hit("h1", "b1", "Fee is 2c at typical prices", "project:kaleon"),
        _hit("h2", "b1", "Break-even accuracy is 67% at $0.65", "project:kaleon"),
    ]
    result = RetrievalResult(heads=hits, user_message="kalshi fees?")

    block = format_memory_block(
        result,
        body_resolver=lambda hit: load_body_preview(bodies_dir, hit.head.body_id, max_chars=80),
    )
    assert "<memory>" in block and "</memory>" in block
    # Both heads listed
    assert "h1" in block and "h2" in block
    # Body preview appears once (deduped by body_id)
    assert block.count("body preview") == 1


def test_format_memory_block_empty():
    result = RetrievalResult(heads=[], user_message="hi")
    block = format_memory_block(result, body_resolver=lambda h: None)
    # No empty <memory> noise injected
    assert block == ""


def test_format_memory_block_includes_edge_relation_and_scope_tag(tmp_path):
    hit = GraphHit(
        head_id="h1", distance=0.2, scope="project:kaleon",
        head=HeadRecord(head_id="h1", body_id="b1", summary="Fee is 2c"),
        edge_relation="contradicts",
    )
    result = RetrievalResult(heads=[hit], user_message="?")
    block = format_memory_block(result, body_resolver=lambda h: None)
    assert "[project:kaleon] h1 — Fee is 2c (contradicts)" in block


def test_format_memory_block_omits_edge_relation_for_vector_seeds(tmp_path):
    hit = GraphHit(
        head_id="h1", distance=0.0, scope="global",
        head=HeadRecord(head_id="h1", body_id="b1", summary="just a summary"),
        edge_relation=None,
    )
    result = RetrievalResult(heads=[hit], user_message="?")
    block = format_memory_block(result, body_resolver=lambda h: None)
    assert "[global] h1 — just a summary" in block
    # No trailing parens for vector seeds
    assert "just a summary (" not in block


def test_load_body_preview_truncates_to_max_chars(tmp_path):
    from datetime import UTC, datetime
    body = BodyFile(
        body_id="b1", title="t", scope="global",
        created=datetime.now(UTC), updated=datetime.now(UTC),
        content="A" * 200,
    )
    write_body_file(tmp_path, body)
    preview = load_body_preview(tmp_path, "b1", max_chars=50)
    assert preview is not None
    assert len(preview) == 50
    assert preview.endswith("…")


def test_load_body_preview_returns_none_for_missing_body(tmp_path):
    assert load_body_preview(tmp_path, "nonexistent", max_chars=80) is None
