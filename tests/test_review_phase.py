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


def _setup_b_and_c_with_high_similarity(store):
    """Insert body B and body C such that find_merge_candidates(B) returns C."""
    from hippo.config import EMBEDDING_DIM
    from hippo.storage.heads import HeadRecord, insert_head
    from hippo.storage.vec import insert_head_embedding

    now = datetime.now(UTC)
    for bid in ("bid-B", "bid-C"):
        write_body_file(
            store.memory_dir,
            BodyFile(
                body_id=bid, title=bid, scope="global",
                created=now, updated=now, content=f"{bid} content",
            ),
        )
        insert_body(
            store.conn,
            BodyRecord(
                body_id=bid, file_path=f"bodies/{bid}.md",
                title=bid, scope="global", source="test",
            ),
        )
        head_id = f"h-{bid}"
        insert_head(
            store.conn, HeadRecord(head_id=head_id, body_id=bid, summary=bid)
        )
        # Identical embeddings → cosine 1.0
        vec = [1.0, 0.0] + [0.0] * (EMBEDDING_DIM - 2)
        insert_head_embedding(store.conn, head_id, vec)


def test_review_body_archives_loser_on_merge(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_b_and_c_with_high_similarity(store)

    from hippo.dream.review import _review_body_against_neighbors
    from hippo.storage.bodies import get_body

    # Keeper = bid-B → bid-C must be archived
    llm = FakeLLM(json.dumps({"decision": "merge", "keeper": "bid-B", "reason": "x"}))
    archived = _review_body_against_neighbors(
        store=store, llm=llm, body_id="bid-B",
    )
    assert archived == 1

    loser = get_body(store.conn, "bid-C")
    assert loser is not None
    assert loser.archived
    assert loser.archived_in_favor_of == "bid-B"
    assert (loser.archive_reason or "").startswith("merged_into:")

    # B was stamped
    b = get_body(store.conn, "bid-B")
    assert b is not None
    assert b.last_reviewed_at is not None

    # B's heads remain active; C's heads are archived
    rows = store.conn.execute(
        "SELECT body_id, archived FROM heads ORDER BY body_id"
    ).fetchall()
    by_bid = {r["body_id"]: r["archived"] for r in rows}
    assert by_bid == {"bid-B": 0, "bid-C": 1}

    store.conn.close()


def test_review_body_archives_loser_on_supersede(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_b_and_c_with_high_similarity(store)

    from hippo.dream.review import _review_body_against_neighbors
    from hippo.storage.bodies import get_body

    llm = FakeLLM(json.dumps({"decision": "supersede", "keeper": "bid-C", "reason": "x"}))
    archived = _review_body_against_neighbors(
        store=store, llm=llm, body_id="bid-B",
    )
    assert archived == 1

    loser = get_body(store.conn, "bid-B")
    assert loser is not None
    assert loser.archived
    assert (loser.archive_reason or "").startswith("superseded_by:")
    assert loser.archived_in_favor_of == "bid-C"

    store.conn.close()


def test_review_body_keep_both_archives_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_b_and_c_with_high_similarity(store)

    from hippo.dream.review import _review_body_against_neighbors
    from hippo.storage.bodies import get_body

    llm = FakeLLM(json.dumps({"decision": "keep_both", "keeper": None, "reason": "x"}))
    archived = _review_body_against_neighbors(
        store=store, llm=llm, body_id="bid-B",
    )
    assert archived == 0

    assert not get_body(store.conn, "bid-B").archived  # type: ignore[union-attr]
    assert not get_body(store.conn, "bid-C").archived  # type: ignore[union-attr]
    # B was still stamped
    assert get_body(store.conn, "bid-B").last_reviewed_at is not None  # type: ignore[union-attr]
    store.conn.close()


def test_review_body_with_no_candidates_stamps_and_returns_zero(tmp_path, monkeypatch):
    """A body whose heads are far from everything else: no LLM call, just stamp."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    # Only one body — no neighbors
    now = datetime.now(UTC)
    write_body_file(
        store.memory_dir,
        BodyFile(
            body_id="bid-solo", title="solo", scope="global",
            created=now, updated=now, content="content",
        ),
    )
    insert_body(
        store.conn,
        BodyRecord(
            body_id="bid-solo", file_path="bodies/bid-solo.md",
            title="solo", scope="global", source="test",
        ),
    )

    from hippo.dream.review import _review_body_against_neighbors
    from hippo.storage.bodies import get_body

    llm = FakeLLM(json.dumps({"decision": "merge", "keeper": "bid-solo", "reason": "x"}))
    archived = _review_body_against_neighbors(
        store=store, llm=llm, body_id="bid-solo",
    )
    assert archived == 0
    # No LLM call should have happened
    assert llm.calls == []
    # Stamped anyway
    assert get_body(store.conn, "bid-solo").last_reviewed_at is not None  # type: ignore[union-attr]
    store.conn.close()


def test_review_new_atoms_processes_only_run_bodies(tmp_path, monkeypatch):
    """Bodies with source='heavy-dream-run:7' are reviewed; others are skipped."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())

    from hippo.config import EMBEDDING_DIM
    from hippo.storage.heads import HeadRecord, insert_head
    from hippo.storage.vec import insert_head_embedding

    now = datetime.now(UTC)
    # Body from this run
    write_body_file(
        store.memory_dir,
        BodyFile(
            body_id="this-run", title="t", scope="global",
            created=now, updated=now, content="content",
        ),
    )
    insert_body(
        store.conn,
        BodyRecord(
            body_id="this-run", file_path="bodies/this-run.md",
            title="t", scope="global", source="heavy-dream-run:7",
        ),
    )
    insert_head(
        store.conn, HeadRecord(head_id="h-this", body_id="this-run", summary="x")
    )
    insert_head_embedding(store.conn, "h-this", [1.0, 0.0] + [0.0] * (EMBEDDING_DIM - 2))

    # Body from a previous run (must NOT be touched by review_new_atoms)
    write_body_file(
        store.memory_dir,
        BodyFile(
            body_id="prev", title="p", scope="global",
            created=now, updated=now, content="prev content",
        ),
    )
    insert_body(
        store.conn,
        BodyRecord(
            body_id="prev", file_path="bodies/prev.md",
            title="p", scope="global", source="heavy-dream-run:6",
        ),
    )
    insert_head(store.conn, HeadRecord(head_id="h-prev", body_id="prev", summary="y"))
    # Identical embedding → would-be merge candidate
    insert_head_embedding(store.conn, "h-prev", [1.0, 0.0] + [0.0] * (EMBEDDING_DIM - 2))

    from hippo.dream.review import review_new_atoms
    from hippo.storage.bodies import get_body

    llm = FakeLLM(json.dumps({"decision": "merge", "keeper": "this-run", "reason": "x"}))
    n_archived = review_new_atoms(store=store, llm=llm, run_id=7)
    assert n_archived == 1

    # The previous-run body got archived (loser)
    prev = get_body(store.conn, "prev")
    assert prev is not None
    assert prev.archived

    # The new body has last_reviewed_at stamped
    new = get_body(store.conn, "this-run")
    assert new is not None
    assert new.last_reviewed_at is not None

    store.conn.close()


def test_review_new_atoms_returns_zero_when_no_run_bodies(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())

    from hippo.dream.review import review_new_atoms

    llm = FakeLLM("")
    assert review_new_atoms(store=store, llm=llm, run_id=99) == 0
    assert llm.calls == []
    store.conn.close()
