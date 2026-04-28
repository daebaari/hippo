"""Tests for bootstrap migration: legacy md → atoms."""
from __future__ import annotations

import json

from hippo.dream.bootstrap import atomize_legacy_files
from hippo.storage.bodies import list_bodies_by_scope
from hippo.storage.multi_store import Scope, open_store


class FakeLLM:
    def __init__(self, response_per_call: list[str]) -> None:
        self.responses = list(response_per_call)
        self.calls: list[str] = []
    def generate_chat(self, messages, *, temperature, max_tokens):
        self.calls.append(messages[-1]["content"])
        return self.responses.pop(0)


class FakeDaemon:
    def embed(self, texts):
        from hippo.config import EMBEDDING_DIM
        return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]


def test_atomize_legacy_files_creates_atoms_per_file(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    monkeypatch.setattr("hippo.config.PROJECTS_ROOT", tmp_path / "projects")

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "feedback_docs_style.md").write_text("Keep docs concise.")
    (legacy_dir / "reference_kalshi_api.md").write_text("Kalshi taker fee max is 2c.")

    fake_atoms_global = json.dumps([{
        "title": "Concise docs",
        "body": "Keep docs concise.",
        "scope": "global",
        "heads": ["docs concise"],
    }])
    fake_atoms_project = json.dumps([{
        "title": "Kalshi fees",
        "body": "Max is 2c.",
        "scope": "project:kaleon",
        "heads": ["kalshi fee max"],
    }])
    llm = FakeLLM([fake_atoms_global, fake_atoms_project])

    n = atomize_legacy_files(
        legacy_dir=legacy_dir, project="kaleon",
        llm=llm, daemon=FakeDaemon(),
    )
    assert n >= 2

    g = open_store(Scope.global_())
    p = open_store(Scope.project("kaleon"))
    g_bodies = list_bodies_by_scope(g.conn, "global")
    p_bodies = list_bodies_by_scope(p.conn, "project:kaleon")
    g.conn.close()
    p.conn.close()

    assert any(b.title == "Concise docs" for b in g_bodies)
    assert any(b.title == "Kalshi fees" for b in p_bodies)
