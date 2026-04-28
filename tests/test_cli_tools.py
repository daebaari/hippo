"""Tests for memory-get, memory-search, memory-archive CLI helpers."""
from __future__ import annotations

from datetime import UTC, datetime

from hippo.cli.archive import archive_head_cli
from hippo.cli.get import get_body_cli
from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.body_files import BodyFile, write_body_file
from hippo.storage.heads import HeadRecord, get_head, insert_head
from hippo.storage.multi_store import Scope, open_store


def test_get_body_cli_returns_content(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    body = BodyFile(
        body_id="b1", title="t", scope="global",
        created=datetime.now(UTC), updated=datetime.now(UTC),
        content="hello world",
    )
    write_body_file(s.memory_dir, body)
    insert_body(
        s.conn,
        BodyRecord(
            body_id="b1", file_path="bodies/b1.md", title="t",
            scope="global", source="manual",
        ),
    )
    insert_head(s.conn, HeadRecord(head_id="h1", body_id="b1", summary="x"))
    s.conn.close()

    rc = get_body_cli(["h1"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "hello world" in captured.out


def test_archive_head_cli_marks_archived(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    insert_body(
        s.conn,
        BodyRecord(
            body_id="b1", file_path="bodies/b1.md", title="t",
            scope="global", source="manual",
        ),
    )
    insert_head(s.conn, HeadRecord(head_id="h1", body_id="b1", summary="x"))
    s.conn.close()

    rc = archive_head_cli(["h1", "--reason", "wrong"])
    assert rc == 0
    s2 = open_store(Scope.global_())
    head = get_head(s2.conn, "h1")
    assert head is not None
    assert head.archived is True
    assert head.archive_reason == "wrong"
    s2.conn.close()
