"""Tests for config constants and path helpers."""
from __future__ import annotations

from pathlib import Path

from hippo import config


def test_global_memory_dir_under_claude_home() -> None:
    expected = Path.home() / ".claude" / "memory"
    assert config.GLOBAL_MEMORY_DIR == expected


def test_project_memory_dir_template_uses_project_name() -> None:
    project = "kaleon"
    expected = Path.home() / ".claude" / "projects" / project / "memory"
    assert config.project_memory_dir(project) == expected


def test_embedding_dim_is_1024() -> None:
    assert config.EMBEDDING_DIM == 1024


def test_edge_relations_list_is_complete() -> None:
    assert "related" in config.EDGE_RELATIONS
    assert "causes" in config.EDGE_RELATIONS
    assert "supersedes" in config.EDGE_RELATIONS
    assert "contradicts" in config.EDGE_RELATIONS
    assert "applies_when" in config.EDGE_RELATIONS
