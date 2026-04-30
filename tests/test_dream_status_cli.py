"""Tests for the dream-status CLI."""
from __future__ import annotations

import json

import pytest

from hippo.cli.dream_status import dream_status_cli, render_run_line
from hippo.storage.dream_runs import (
    DreamRunRecord,
    complete_run,
    start_phase,
    start_run,
    update_progress,
)
from hippo.storage.migrations import run_migrations
from hippo.storage.multi_store import Scope, open_store


def _make_record(
    *,
    run_id: int = 1,
    status: str = "running",
    current_phase: str | None = "edge_proposal",
    phase_done: int | None = 100,
    phase_total: int | None = 500,
    started_at: int = 1_700_000_000,
    last_progress_at: int | None = 1_700_000_120,
) -> DreamRunRecord:
    return DreamRunRecord(
        run_id=run_id,
        type="heavy",
        started_at=started_at,
        completed_at=None,
        status=status,
        atoms_created=0,
        heads_created=0,
        edges_created=0,
        contradictions_resolved=0,
        bodies_archived_review=0,
        error_message=None,
        current_phase=current_phase,
        phase_done=phase_done,
        phase_total=phase_total,
        phase_started_at=started_at,
        last_progress_at=last_progress_at,
    )


def test_render_run_line_for_running_run():
    rec = _make_record()
    line = render_run_line(rec, scope_name="kaleon", now_unix=1_700_000_180)
    assert "running" in line
    assert "kaleon" in line
    assert "phase=edge_proposal" in line
    assert "100/500" in line
    assert "(20.0%)" in line


def test_render_run_line_for_completed_run():
    rec = _make_record(status="completed", current_phase="cleanup")
    line = render_run_line(rec, scope_name="kaleon", now_unix=1_700_000_180)
    assert "completed" in line


def test_dream_status_cli_no_runs_returns_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    monkeypatch.setattr("hippo.config.PROJECTS_ROOT", tmp_path / "projects")
    rc = dream_status_cli(["--scope", "global"])
    assert rc == 1
    assert "no dream" in capsys.readouterr().out.lower()


def test_dream_status_cli_finds_running_dream(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    rid = start_run(s.conn, "heavy")
    start_phase(s.conn, rid, phase="edge_proposal", total=500)
    update_progress(s.conn, rid, done=100)
    s.conn.close()

    rc = dream_status_cli(["--scope", "global"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "running" in out
    assert "phase=edge_proposal" in out


def test_dream_status_cli_falls_back_to_completed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    rid = start_run(s.conn, "heavy")
    complete_run(s.conn, rid)
    s.conn.close()

    rc = dream_status_cli(["--scope", "global"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "completed" in out
