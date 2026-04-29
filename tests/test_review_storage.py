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
