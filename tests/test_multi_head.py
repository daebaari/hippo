"""Test multi-head expansion phase with a mock LLM."""
from __future__ import annotations

import json
from datetime import UTC, datetime

from hippo.dream.multi_head import expand_heads_for_eligible_bodies
from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.body_files import BodyFile, write_body_file
from hippo.storage.heads import HeadRecord, increment_retrieval, insert_head, list_heads_for_body
from hippo.storage.multi_store import Scope, open_store


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[str] = []

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        thinking_level: str | None = None,
    ) -> str:
        self.calls.append(messages[-1]["content"])
        return self.response


class FakeDaemon:
    def embed(self, texts: list[str]) -> list[list[float]]:
        from hippo.config import EMBEDDING_DIM
        return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]


def test_expand_heads_inserts_new_heads(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())

    # Insert a body with one head that has retrieval_count > 0
    body_id = "body0001"
    now = datetime.now(UTC)
    write_body_file(s.memory_dir, BodyFile(
        body_id=body_id,
        title="Project uses postgres",
        scope="global",
        created=now,
        updated=now,
        content="We use postgres as the main DB.",
    ))
    insert_body(s.conn, BodyRecord(
        body_id=body_id,
        file_path=f"bodies/{body_id}.md",
        title="Project uses postgres",
        scope="global",
        source="test",
    ))
    head_id = "head0001"
    insert_head(s.conn, HeadRecord(head_id=head_id, body_id=body_id, summary="uses postgres"))
    increment_retrieval(s.conn, head_id)

    llm = FakeLLM(json.dumps(["new head one", "new head two"]))
    daemon = FakeDaemon()

    n_new = expand_heads_for_eligible_bodies(store=s, llm=llm, daemon=daemon)

    assert n_new == 2
    heads = list_heads_for_body(s.conn, body_id)
    assert len(heads) == 3  # 1 original + 2 new
    summaries = {h.summary for h in heads}
    assert "new head one" in summaries
    assert "new head two" in summaries

    s.conn.close()


def test_expand_heads_skips_body_without_retrieval(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())

    body_id = "body0002"
    now = datetime.now(UTC)
    write_body_file(s.memory_dir, BodyFile(
        body_id=body_id,
        title="Never retrieved body",
        scope="global",
        created=now,
        updated=now,
        content="This body was never retrieved.",
    ))
    insert_body(s.conn, BodyRecord(
        body_id=body_id,
        file_path=f"bodies/{body_id}.md",
        title="Never retrieved body",
        scope="global",
        source="test",
    ))
    # One head, but retrieval_count stays at 0
    insert_head(s.conn, HeadRecord(head_id="head0002", body_id=body_id, summary="unretrieved head"))

    llm = FakeLLM(json.dumps(["extra head"]))
    daemon = FakeDaemon()

    n_new = expand_heads_for_eligible_bodies(store=s, llm=llm, daemon=daemon)

    assert n_new == 0
    heads = list_heads_for_body(s.conn, body_id)
    assert len(heads) == 1  # unchanged

    s.conn.close()


def test_expand_heads_skips_already_full_body(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())

    body_id = "body0003"
    now = datetime.now(UTC)
    write_body_file(s.memory_dir, BodyFile(
        body_id=body_id,
        title="Full head body",
        scope="global",
        created=now,
        updated=now,
        content="Already has enough heads.",
    ))
    insert_body(s.conn, BodyRecord(
        body_id=body_id,
        file_path=f"bodies/{body_id}.md",
        title="Full head body",
        scope="global",
        source="test",
    ))
    for i in range(3):
        hid = f"head000{i + 10}"
        insert_head(s.conn, HeadRecord(head_id=hid, body_id=body_id, summary=f"head {i}"))
        increment_retrieval(s.conn, hid)

    llm = FakeLLM(json.dumps(["should not appear"]))
    daemon = FakeDaemon()

    n_new = expand_heads_for_eligible_bodies(store=s, llm=llm, daemon=daemon)

    assert n_new == 0
    heads = list_heads_for_body(s.conn, body_id)
    assert len(heads) == 3  # unchanged

    s.conn.close()
