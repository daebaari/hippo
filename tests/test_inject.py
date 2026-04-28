"""Tests for context injection format."""
from __future__ import annotations

from pathlib import Path

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
    from datetime import datetime, timezone
    body = BodyFile(
        body_id="b1", title="Kalshi fees", scope="project:kaleon",
        created=datetime.now(timezone.utc), updated=datetime.now(timezone.utc),
        content="Fee is 2c at typical prices and grows to ...",
    )
    bodies_dir = tmp_path / "global"
    write_body_file(bodies_dir, body)

    hits = [
        _hit("h1", "b1", "Fee is 2c at typical prices", "project:kaleon"),
        _hit("h2", "b1", "Break-even accuracy is 67% at $0.65", "project:kaleon"),
    ]
    result = RetrievalResult(heads=hits, user_message="kalshi fees?")

    block = format_memory_block(result, body_resolver=lambda hit: load_body_preview(bodies_dir, hit.head.body_id, max_chars=80))
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
