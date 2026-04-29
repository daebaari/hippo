"""Tests for the heavy dream orchestrator."""
from __future__ import annotations

import json

from hippo.config import EMBEDDING_DIM, HEAVY_LOCK_FILENAME
from hippo.dream.heavy import run_heavy_dream_all_scopes, run_heavy_dream_for_scope
from hippo.lock import acquire_lock
from hippo.storage.capture import CaptureRecord, enqueue_capture
from hippo.storage.multi_store import Scope, open_store


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        thinking_level: str | None = None,
    ) -> str:
        content = messages[-1]["content"]
        self.calls.append(content)
        if "extracting durable memory atoms" in content:
            return json.dumps([{
                "title": "Test atom",
                "body": "Some content",
                "scope": "global",
                "heads": ["one head"],
                "noise": False,
            }])
        if "generating diverse keyword summaries" in content:
            return json.dumps(["another head"])
        if "deciding whether two memory heads are related" in content:
            return json.dumps({"relation": "related", "weight": 0.5})
        if "deciding whether two memory atoms genuinely contradict" in content:
            return json.dumps({"contradicts": False})
        if "deciding whether two memory atoms are redundant" in content:
            return json.dumps({"decision": "keep_both", "keeper": None, "reason": "x"})
        return "[]"


class FakeDaemon:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]


def test_heavy_dream_full_orchestration(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")

    s = open_store(Scope.global_())
    enqueue_capture(
        s.conn,
        CaptureRecord(session_id="sess-A", user_message="we use postgres", assistant_message="ok"),
    )
    s.conn.close()

    llm = FakeLLM()
    daemon = FakeDaemon()

    results = run_heavy_dream_all_scopes(scopes=[Scope.global_()], llm=llm, daemon=daemon)

    assert "global" in results
    result = results["global"]
    assert isinstance(result["run_id"], int)
    assert result["atoms_created"] == 1
    assert isinstance(result["heads_created"], int)
    assert result["heads_created"] >= 0
    assert isinstance(result["edges_created"], int)
    assert result["edges_created"] >= 0
    assert isinstance(result["contradictions_resolved"], int)
    assert result["contradictions_resolved"] >= 0

    # Verify the capture is now marked as processed
    s2 = open_store(Scope.global_())
    unprocessed = s2.conn.execute(
        "SELECT COUNT(*) FROM capture_queue WHERE processed_at IS NULL"
    ).fetchone()[0]
    assert unprocessed == 0
    s2.conn.close()


def test_heavy_dream_lock_held_returns_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")

    # Pre-create the store so the memory dir and lock path exist
    s = open_store(Scope.global_())
    lock_path = s.memory_dir / HEAVY_LOCK_FILENAME
    s.conn.close()

    # Acquire the lock ourselves first
    handle = acquire_lock(lock_path)
    try:
        result = run_heavy_dream_for_scope(
            scope=Scope.global_(), llm=FakeLLM(), daemon=FakeDaemon()
        )
        assert result == {"skipped_locked": True}
    finally:
        from hippo.lock import release_lock
        release_lock(handle)


def test_heavy_dream_runs_review_phase_and_records_counter(tmp_path, monkeypatch):
    """End-to-end: heavy dream runs the review phase between atomize and multi_head,
    populates bodies_archived_review in dream_runs, and exposes it in the result."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")

    s = open_store(Scope.global_())
    enqueue_capture(
        s.conn,
        CaptureRecord(
            session_id="sess-A", user_message="we use postgres", assistant_message="ok",
        ),
    )
    s.conn.close()

    llm = FakeLLM()
    daemon = FakeDaemon()
    results = run_heavy_dream_all_scopes(scopes=[Scope.global_()], llm=llm, daemon=daemon)

    result = results["global"]
    assert "bodies_archived_review" in result
    assert isinstance(result["bodies_archived_review"], int)

    # The dream_runs row reflects the same counter
    s = open_store(Scope.global_())
    row = s.conn.execute(
        "SELECT bodies_archived_review FROM dream_runs WHERE run_id = ?",
        (result["run_id"],),
    ).fetchone()
    assert row is not None
    assert row["bodies_archived_review"] == result["bodies_archived_review"]
    s.conn.close()
