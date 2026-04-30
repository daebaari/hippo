"""Tests for the contradiction resolution phase."""
from __future__ import annotations

import json
from datetime import UTC, datetime

from hippo.dream.contradiction import resolve_contradictions
from hippo.storage.bodies import BodyRecord, get_body, insert_body
from hippo.storage.body_files import BodyFile, write_body_file
from hippo.storage.edges import EdgeRecord, insert_edge_with_reciprocal
from hippo.storage.heads import HeadRecord, get_head, insert_head
from hippo.storage.multi_store import Scope, open_store


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[str] = []
        self.thinking_levels: list[str | None] = []

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        thinking_level: str | None = None,
    ) -> str:
        self.calls.append(messages[-1]["content"])
        self.thinking_levels.append(thinking_level)
        return self.response


def _setup_bodies_and_heads(store, *, a_body_id: str, b_body_id: str) -> tuple[str, str]:
    """Insert two bodies with heads and body files. Returns (a_head_id, b_head_id)."""
    now = datetime.now(UTC)

    for body_id, content in [
        (a_body_id, "The deployment target is AWS us-east-1."),
        (b_body_id, "The deployment target is GCP europe-west1."),
    ]:
        write_body_file(
            store.memory_dir,
            BodyFile(
                body_id=body_id,
                title=f"Deployment info {body_id}",
                scope="global",
                created=now,
                updated=now,
                content=content,
            ),
        )
        insert_body(
            store.conn,
            BodyRecord(
                body_id=body_id,
                file_path=f"bodies/{body_id}.md",
                title=f"Deployment info {body_id}",
                scope="global",
                source="test",
            ),
        )

    a_head_id = f"head-{a_body_id}"
    b_head_id = f"head-{b_body_id}"

    insert_head(
        store.conn, HeadRecord(head_id=a_head_id, body_id=a_body_id, summary="AWS deployment")
    )
    insert_head(
        store.conn, HeadRecord(head_id=b_head_id, body_id=b_body_id, summary="GCP deployment")
    )

    return a_head_id, b_head_id


def test_resolve_contradictions_archives_loser(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())

    a_body_id = "body-aaa"
    b_body_id = "body-bbb"
    a_head_id, b_head_id = _setup_bodies_and_heads(store, a_body_id=a_body_id, b_body_id=b_body_id)

    # Insert a contradicts edge (symmetric — reciprocal auto-inserted)
    # Use a_head < b_head alphabetically so the query picks it up
    insert_edge_with_reciprocal(
        store.conn,
        EdgeRecord(from_head=a_head_id, to_head=b_head_id, relation="contradicts"),
    )

    llm = FakeLLM(
        json.dumps({"contradicts": True, "current_body_id": a_body_id, "reason": "newer"})
    )

    n = resolve_contradictions(store=store, llm=llm)

    assert n == 1

    # Winner (a) is still active
    winner = get_body(store.conn, a_body_id)
    assert winner is not None
    assert not winner.archived

    # Loser (b) is archived
    loser = get_body(store.conn, b_body_id)
    assert loser is not None
    assert loser.archived
    assert loser.archived_in_favor_of == a_body_id

    # Loser's head is also archived
    loser_head = get_head(store.conn, b_head_id)
    assert loser_head is not None
    assert loser_head.archived

    # Regression: assert thinking_level="medium" was passed
    assert all(level == "medium" for level in llm.thinking_levels)
    assert llm.thinking_levels  # at least one call

    store.conn.close()


def test_resolve_contradictions_skips_when_llm_says_no_contradiction(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())

    a_body_id = "body-ccc"
    b_body_id = "body-ddd"
    a_head_id, b_head_id = _setup_bodies_and_heads(store, a_body_id=a_body_id, b_body_id=b_body_id)

    insert_edge_with_reciprocal(
        store.conn,
        EdgeRecord(from_head=a_head_id, to_head=b_head_id, relation="contradicts"),
    )

    llm = FakeLLM(json.dumps({"contradicts": False, "reason": "no conflict"}))

    n = resolve_contradictions(store=store, llm=llm)

    assert n == 0

    # Both bodies remain active
    assert not get_body(store.conn, a_body_id).archived  # type: ignore[union-attr]
    assert not get_body(store.conn, b_body_id).archived  # type: ignore[union-attr]

    store.conn.close()


def test_resolve_contradictions_skips_already_archived(tmp_path, monkeypatch):
    """If one side is already archived before we run, skip the pair."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())

    a_body_id = "body-eee"
    b_body_id = "body-fff"
    a_head_id, b_head_id = _setup_bodies_and_heads(store, a_body_id=a_body_id, b_body_id=b_body_id)

    insert_edge_with_reciprocal(
        store.conn,
        EdgeRecord(from_head=a_head_id, to_head=b_head_id, relation="contradicts"),
    )

    # Pre-archive head a
    store.conn.execute("UPDATE heads SET archived = 1 WHERE head_id = ?", (a_head_id,))
    store.conn.commit()

    llm = FakeLLM(json.dumps({"contradicts": True, "current_body_id": b_body_id, "reason": "ok"}))

    n = resolve_contradictions(store=store, llm=llm)

    assert n == 0
    # b_body should still be active since we skipped
    assert not get_body(store.conn, b_body_id).archived  # type: ignore[union-attr]

    store.conn.close()


def test_resolve_contradictions_handles_invalid_json(tmp_path, monkeypatch):
    """Gracefully skip pairs where LLM returns non-JSON."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())

    a_body_id = "body-ggg"
    b_body_id = "body-hhh"
    a_head_id, b_head_id = _setup_bodies_and_heads(store, a_body_id=a_body_id, b_body_id=b_body_id)

    insert_edge_with_reciprocal(
        store.conn,
        EdgeRecord(from_head=a_head_id, to_head=b_head_id, relation="contradicts"),
    )

    llm = FakeLLM("This is not JSON at all!")

    n = resolve_contradictions(store=store, llm=llm)

    assert n == 0

    store.conn.close()
