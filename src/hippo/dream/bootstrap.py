"""Bootstrap migration: atomize legacy memory files into new Hippo schema.

For each .md file (skip MEMORY.md):
- Read filename + content + (optionally) any MEMORY.md description.
- Prompt LLM to atomize, with the filename as a hint to scope.
- Write atoms to global or project store per LLM's scope decision.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from hippo.dream.atomize import _strip_fences
from hippo.dream.prompts import render
from hippo.models.llm import LLMProto
from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.body_files import BodyFile, write_body_file
from hippo.storage.heads import HeadRecord, insert_head
from hippo.storage.multi_store import Scope, open_store
from hippo.storage.vec import insert_head_embedding


class DaemonProto(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _scope_hint_from_filename(filename: str) -> str:
    """Return 'global' or 'project' as a hint based on filename prefix."""
    if filename.startswith(("user_", "feedback_")):
        return "global"
    if filename.startswith(("project_",)):
        return "project"
    # 'reference_' is split: most are project-specific
    return "project"


def atomize_legacy_files(
    *, legacy_dir: Path, project: str,
    llm: LLMProto, daemon: DaemonProto,
) -> int:
    """Run atomize on each legacy .md file. Returns total atoms created."""
    md_files = [
        p for p in sorted(legacy_dir.glob("*.md")) if p.name.upper() != "MEMORY.MD"
    ]
    total = len(md_files)
    n_atoms = 0
    for idx, md_path in enumerate(md_files, start=1):
        print(f"[{idx}/{total}] atomizing {md_path.name}...", flush=True)
        file_atoms = 0
        content = md_path.read_text()
        scope_hint = _scope_hint_from_filename(md_path.name)

        prompt = render(
            "atomize",
            project=project,
            session_id=f"legacy:{md_path.name}",
            transcript=(
                f"FILENAME: {md_path.name}\n"
                f"SCOPE_HINT: {scope_hint}\n\n"
                f"FILE CONTENT:\n{content}"
            ),
        )
        raw = llm.generate_chat(
            [{"role": "user", "content": prompt}], temperature=0.2, max_tokens=4096
        )
        try:
            atoms = json.loads(_strip_fences(raw))
        except json.JSONDecodeError:
            print(f"[{idx}/{total}] {md_path.name}: {file_atoms} atoms", flush=True)
            continue

        for atom in atoms:
            title = (atom.get("title") or "")[:120]
            body_content = atom.get("body") or ""
            scope_str = atom.get("scope") or (
                "global" if scope_hint == "global" else f"project:{project}"
            )
            heads = [
                h for h in (atom.get("heads") or []) if isinstance(h, str) and h.strip()
            ][:5]
            if not title or not body_content or not heads:
                continue

            scope_obj = Scope.global_() if scope_str == "global" else Scope.project(project)
            store = open_store(scope_obj)
            try:
                body_id = uuid4().hex
                now = datetime.now(UTC)
                write_body_file(store.memory_dir, BodyFile(
                    body_id=body_id, title=title, scope=scope_obj.as_string(),
                    created=now, updated=now, content=body_content,
                ))
                insert_body(store.conn, BodyRecord(
                    body_id=body_id, file_path=f"bodies/{body_id}.md",
                    title=title, scope=scope_obj.as_string(),
                    source=f"migration:{md_path.name}",
                ))
                vecs = daemon.embed(heads)
                for summary, vec in zip(heads, vecs, strict=True):
                    head_id = uuid4().hex
                    insert_head(
                        store.conn,
                        HeadRecord(head_id=head_id, body_id=body_id, summary=summary),
                    )
                    insert_head_embedding(store.conn, head_id, vec)
                n_atoms += 1
                file_atoms += 1
            finally:
                store.conn.close()
        print(f"[{idx}/{total}] {md_path.name}: {file_atoms} atoms", flush=True)
    return n_atoms
