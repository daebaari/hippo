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
