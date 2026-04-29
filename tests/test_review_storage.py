"""Storage helpers used by the review phase."""
from __future__ import annotations

import time

from hippo.storage.bodies import (
    BodyRecord,
    get_body,
    insert_body,
    update_last_reviewed_at,
)
from hippo.storage.migrations import run_migrations


def test_update_last_reviewed_at_stamps_now(sqlite_conn):
    run_migrations(sqlite_conn)
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="b1", file_path="bodies/b1.md", title="t",
            scope="global", source="test",
        ),
    )

    before = int(time.time())
    update_last_reviewed_at(sqlite_conn, "b1")
    after = int(time.time())

    rec = get_body(sqlite_conn, "b1")
    assert rec is not None
    assert rec.last_reviewed_at is not None
    assert before <= rec.last_reviewed_at <= after


def test_find_oldest_unreviewed_active_orders_null_first_then_oldest(sqlite_conn):
    run_migrations(sqlite_conn)
    # b-null: never reviewed (NULL)
    # b-old:  reviewed long ago
    # b-new:  reviewed recently
    # b-arch: archived (must be excluded)
    for bid in ("b-null", "b-old", "b-new", "b-arch"):
        insert_body(
            sqlite_conn,
            BodyRecord(
                body_id=bid, file_path=f"bodies/{bid}.md",
                title=bid, scope="global", source="test",
            ),
        )
    sqlite_conn.execute("UPDATE bodies SET last_reviewed_at = 100 WHERE body_id = 'b-old'")
    sqlite_conn.execute("UPDATE bodies SET last_reviewed_at = 999 WHERE body_id = 'b-new'")
    sqlite_conn.execute("UPDATE bodies SET archived = 1 WHERE body_id = 'b-arch'")
    sqlite_conn.commit()

    from hippo.storage.bodies import find_oldest_unreviewed_active

    out = find_oldest_unreviewed_active(sqlite_conn, scope="global", limit=10)
    ids = [b.body_id for b in out]
    assert ids == ["b-null", "b-old", "b-new"]


def test_find_oldest_unreviewed_active_respects_limit(sqlite_conn):
    run_migrations(sqlite_conn)
    for bid in ("b1", "b2", "b3"):
        insert_body(
            sqlite_conn,
            BodyRecord(
                body_id=bid, file_path=f"bodies/{bid}.md",
                title=bid, scope="global", source="test",
            ),
        )

    from hippo.storage.bodies import find_oldest_unreviewed_active

    out = find_oldest_unreviewed_active(sqlite_conn, scope="global", limit=2)
    assert len(out) == 2


def test_find_oldest_unreviewed_active_filters_by_scope(sqlite_conn):
    run_migrations(sqlite_conn)
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="g1", file_path="bodies/g1.md", title="g1",
            scope="global", source="test",
        ),
    )
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="p1", file_path="bodies/p1.md", title="p1",
            scope="project:foo", source="test",
        ),
    )

    from hippo.storage.bodies import find_oldest_unreviewed_active

    out = find_oldest_unreviewed_active(sqlite_conn, scope="global", limit=10)
    assert [b.body_id for b in out] == ["g1"]


def test_find_active_bodies_by_run_source_returns_only_run_bodies(sqlite_conn):
    run_migrations(sqlite_conn)
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="b-this", file_path="bodies/b-this.md", title="t",
            scope="global", source="heavy-dream-run:42",
        ),
    )
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="b-other", file_path="bodies/b-other.md", title="t",
            scope="global", source="heavy-dream-run:7",
        ),
    )
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="b-arch", file_path="bodies/b-arch.md", title="t",
            scope="global", source="heavy-dream-run:42",
        ),
    )
    sqlite_conn.execute("UPDATE bodies SET archived = 1 WHERE body_id = 'b-arch'")
    sqlite_conn.commit()

    from hippo.storage.bodies import find_active_bodies_by_run_source

    out = find_active_bodies_by_run_source(sqlite_conn, run_id=42)
    assert [b.body_id for b in out] == ["b-this"]


def _setup_two_bodies_with_heads(conn, *, sim_pairs):
    """Helper: insert body B and a candidate C with controlled head embeddings.

    sim_pairs is a list of (b_vec, c_vec) tuples; each pair becomes one head per side.
    """
    from hippo.config import EMBEDDING_DIM
    from hippo.storage.heads import HeadRecord, insert_head
    from hippo.storage.vec import insert_head_embedding

    run_migrations(conn)
    for bid in ("B", "C"):
        insert_body(
            conn,
            BodyRecord(
                body_id=bid, file_path=f"bodies/{bid}.md",
                title=bid, scope="global", source="test",
            ),
        )

    for i, (bvec, cvec) in enumerate(sim_pairs):
        b_head = f"hB-{i}"
        c_head = f"hC-{i}"
        # pad each input to EMBEDDING_DIM
        b_full = list(bvec) + [0.0] * (EMBEDDING_DIM - len(bvec))
        c_full = list(cvec) + [0.0] * (EMBEDDING_DIM - len(cvec))
        insert_head(conn, HeadRecord(head_id=b_head, body_id="B", summary=f"b{i}"))
        insert_head(conn, HeadRecord(head_id=c_head, body_id="C", summary=f"c{i}"))
        insert_head_embedding(conn, b_head, b_full)
        insert_head_embedding(conn, c_head, c_full)


def test_find_merge_candidates_returns_high_similarity_body(sqlite_conn):
    # B head: [1, 0]; C head: [1, 0] → cosine = 1.0
    _setup_two_bodies_with_heads(sqlite_conn, sim_pairs=[([1.0, 0.0], [1.0, 0.0])])

    from hippo.storage.bodies import find_merge_candidates

    out = find_merge_candidates(sqlite_conn, body_id="B", threshold=0.85, k=5)
    assert len(out) == 1
    rec, sim = out[0]
    assert rec.body_id == "C"
    assert sim >= 0.99


def test_find_merge_candidates_skips_below_threshold(sqlite_conn):
    # B head [1, 0]; C head [0, 1] → cosine = 0.0
    _setup_two_bodies_with_heads(sqlite_conn, sim_pairs=[([1.0, 0.0], [0.0, 1.0])])

    from hippo.storage.bodies import find_merge_candidates

    out = find_merge_candidates(sqlite_conn, body_id="B", threshold=0.85, k=5)
    assert out == []


def test_find_merge_candidates_excludes_self(sqlite_conn):
    # Only one body, identical to itself
    _setup_two_bodies_with_heads(sqlite_conn, sim_pairs=[([1.0, 0.0], [1.0, 0.0])])
    # Drop body C so only B remains
    sqlite_conn.execute("UPDATE bodies SET archived = 1 WHERE body_id = 'C'")
    sqlite_conn.execute("UPDATE heads SET archived = 1 WHERE body_id = 'C'")
    sqlite_conn.commit()

    from hippo.storage.bodies import find_merge_candidates

    out = find_merge_candidates(sqlite_conn, body_id="B", threshold=0.85, k=5)
    assert out == []


def test_find_merge_candidates_respects_k(sqlite_conn):
    """When multiple candidate bodies exist, only top-K (by max similarity) are returned."""
    from hippo.config import EMBEDDING_DIM
    from hippo.storage.heads import HeadRecord, insert_head
    from hippo.storage.vec import insert_head_embedding

    run_migrations(sqlite_conn)
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="B", file_path="bodies/B.md", title="B",
            scope="global", source="test",
        ),
    )
    insert_head(sqlite_conn, HeadRecord(head_id="hB", body_id="B", summary="b"))
    insert_head_embedding(sqlite_conn, "hB", [1.0, 0.0] + [0.0] * (EMBEDDING_DIM - 2))

    # Three candidates, one perfectly similar (C1), two below threshold (C2, C3)
    candidates = [
        ("C1", [1.0, 0.0]),     # cosine 1.0
        ("C2", [0.0, 1.0]),     # cosine 0.0
        ("C3", [-1.0, 0.0]),    # cosine -1.0
    ]
    for cid, vec in candidates:
        insert_body(
            sqlite_conn,
            BodyRecord(
                body_id=cid, file_path=f"bodies/{cid}.md", title=cid,
                scope="global", source="test",
            ),
        )
        insert_head(sqlite_conn, HeadRecord(head_id=f"h{cid}", body_id=cid, summary=cid))
        insert_head_embedding(
            sqlite_conn, f"h{cid}", vec + [0.0] * (EMBEDDING_DIM - 2)
        )

    from hippo.storage.bodies import find_merge_candidates

    out = find_merge_candidates(sqlite_conn, body_id="B", threshold=0.85, k=5)
    assert [r.body_id for r, _ in out] == ["C1"]
