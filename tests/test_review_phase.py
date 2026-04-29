"""Tests for the prune-phase review module."""
from __future__ import annotations

import json
from datetime import UTC, datetime

from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.body_files import BodyFile, write_body_file
from hippo.storage.multi_store import Scope, open_store


class FakeLLM:
    """LLM stub. By default returns the configured response; can also be a callable
    of (messages) -> response for per-call control."""

    def __init__(self, response):
        self.response = response
        self.calls: list[str] = []
        self.thinking_levels: list[str | None] = []

    def generate_chat(self, messages, *, temperature, max_tokens, thinking_level=None):
        content = messages[-1]["content"]
        self.calls.append(content)
        self.thinking_levels.append(thinking_level)
        if callable(self.response):
            return self.response(content)
        return self.response


class FakeDaemon:
    def embed(self, texts):
        from hippo.config import EMBEDDING_DIM
        return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]


def _setup_two_bodies(store, *, a_id="bid-a", b_id="bid-b"):
    now = datetime.now(UTC)
    for bid, content in [(a_id, "Body A content"), (b_id, "Body B content")]:
        write_body_file(
            store.memory_dir,
            BodyFile(
                body_id=bid, title=bid, scope="global",
                created=now, updated=now, content=content,
            ),
        )
        insert_body(
            store.conn,
            BodyRecord(
                body_id=bid, file_path=f"bodies/{bid}.md",
                title=bid, scope="global", source="test",
            ),
        )


def test_judge_pair_returns_merge_with_keeper(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_two_bodies(store)
    a = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-a'").fetchone()
    b = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-b'").fetchone()

    from hippo.dream.review import _judge_pair
    from hippo.storage.bodies import _row_to_record

    llm = FakeLLM(json.dumps({"decision": "merge", "keeper": "bid-a", "reason": "x"}))
    decision, keeper = _judge_pair(
        store=store, llm=llm,
        a=_row_to_record(a), b=_row_to_record(b),
    )
    assert decision == "merge"
    assert keeper == "bid-a"
    assert llm.thinking_levels == ["minimal"]
    store.conn.close()


def test_judge_pair_invalid_json_returns_keep_both(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_two_bodies(store)
    a = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-a'").fetchone()
    b = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-b'").fetchone()

    from hippo.dream.review import _judge_pair
    from hippo.storage.bodies import _row_to_record

    llm = FakeLLM("not JSON")
    decision, keeper = _judge_pair(
        store=store, llm=llm,
        a=_row_to_record(a), b=_row_to_record(b),
    )
    assert decision == "keep_both"
    assert keeper is None
    store.conn.close()


def test_judge_pair_unknown_keeper_returns_keep_both(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_two_bodies(store)
    a = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-a'").fetchone()
    b = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-b'").fetchone()

    from hippo.dream.review import _judge_pair
    from hippo.storage.bodies import _row_to_record

    # decision=merge but keeper points at a body not in the pair
    llm = FakeLLM(json.dumps(
        {"decision": "merge", "keeper": "totally-unknown", "reason": "x"}
    ))
    decision, keeper = _judge_pair(
        store=store, llm=llm,
        a=_row_to_record(a), b=_row_to_record(b),
    )
    assert decision == "keep_both"
    assert keeper is None
    store.conn.close()


def test_judge_pair_keep_both(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_two_bodies(store)
    a = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-a'").fetchone()
    b = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-b'").fetchone()

    from hippo.dream.review import _judge_pair
    from hippo.storage.bodies import _row_to_record

    llm = FakeLLM(json.dumps({"decision": "keep_both", "keeper": None, "reason": "distinct"}))
    decision, keeper = _judge_pair(
        store=store, llm=llm,
        a=_row_to_record(a), b=_row_to_record(b),
    )
    assert decision == "keep_both"
    assert keeper is None
    store.conn.close()


def test_judge_pair_missing_body_file_returns_keep_both(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_two_bodies(store)
    # Delete one of the body files
    (store.memory_dir / "bodies" / "bid-a.md").unlink()

    a = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-a'").fetchone()
    b = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-b'").fetchone()

    from hippo.dream.review import _judge_pair
    from hippo.storage.bodies import _row_to_record

    llm = FakeLLM(json.dumps({"decision": "merge", "keeper": "bid-a", "reason": "x"}))
    decision, keeper = _judge_pair(
        store=store, llm=llm,
        a=_row_to_record(a), b=_row_to_record(b),
    )
    assert decision == "keep_both"
    assert keeper is None
    store.conn.close()
