"""Tests for body markdown file I/O with YAML frontmatter."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from hippo.storage.body_files import BodyFile, read_body_file, write_body_file


def test_write_then_read_roundtrip(temp_memory_dir: Path) -> None:
    body = BodyFile(
        body_id="01HZK1234567890ABCDEFGHIJK",
        title="Kalshi taker fee max",
        scope="project:kaleon",
        created=datetime(2026, 4, 15, 10, 30, tzinfo=timezone.utc),
        updated=datetime(2026, 4, 27, 15, 0, tzinfo=timezone.utc),
        content="Taker fee is ceil(0.07 × P × (1-P)) cents.\nMax is 2c at 20-80c.",
    )
    write_body_file(temp_memory_dir, body)
    written_path = temp_memory_dir / "bodies" / f"{body.body_id}.md"
    assert written_path.exists()
    loaded = read_body_file(written_path)
    assert loaded.body_id == body.body_id
    assert loaded.title == body.title
    assert loaded.scope == body.scope
    assert loaded.content == body.content


def test_write_creates_bodies_subdir(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    # bodies/ does NOT exist yet
    body = BodyFile(
        body_id="01HZK1234567890ABCDEFGHIJK",
        title="t",
        scope="global",
        created=datetime.now(timezone.utc),
        updated=datetime.now(timezone.utc),
        content="x",
    )
    write_body_file(memory_dir, body)
    assert (memory_dir / "bodies" / f"{body.body_id}.md").exists()


def test_read_missing_file_raises(temp_memory_dir: Path) -> None:
    import pytest
    with pytest.raises(FileNotFoundError):
        read_body_file(temp_memory_dir / "bodies" / "missing.md")
