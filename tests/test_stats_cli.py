"""End-to-end smoke test for the memory-stats CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hippo.cli.stats import collect_stats
from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.dream_runs import complete_run, start_run
from hippo.storage.heads import HeadRecord, insert_head
from hippo.storage.multi_store import Scope, open_store


def test_collect_stats_returns_counts_per_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    monkeypatch.setattr("hippo.config.PROJECTS_ROOT", tmp_path / "projects")

    g = open_store(Scope.global_())
    insert_body(g.conn, BodyRecord(body_id="b1", file_path="bodies/b1.md", title="t1", scope="global", source="manual"))
    insert_head(g.conn, HeadRecord(head_id="h1", body_id="b1", summary="x"))
    rid = start_run(g.conn, "heavy")
    complete_run(g.conn, rid, atoms_created=1, heads_created=1)
    g.conn.close()

    p = open_store(Scope.project("kaleon"))
    insert_body(p.conn, BodyRecord(body_id="b2", file_path="bodies/b2.md", title="t2", scope="project:kaleon", source="manual"))
    p.conn.close()

    result = collect_stats(scopes=[Scope.global_(), Scope.project("kaleon")])
    assert result["global"]["body_count"] == 1
    assert result["global"]["head_count"] == 1
    assert result["global"]["recent_runs"][0]["status"] == "completed"
    assert result["project:kaleon"]["body_count"] == 1
    assert result["project:kaleon"]["head_count"] == 0
