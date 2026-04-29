"""Prune-phase review: gate-at-entry + rolling-slice sweep.

Detects redundant or superseded atoms inside the heavy dream and
soft-archives the loser. Modes A (factually superseded) and D
(redundant / mergeable) from the prune-phase design spec.
"""
from __future__ import annotations

import json
from typing import Protocol

from hippo.dream.atomize import _strip_fences
from hippo.dream.prompts import render
from hippo.models.llm import LLMProto
from hippo.storage.bodies import BodyRecord
from hippo.storage.body_files import read_body_file
from hippo.storage.multi_store import Store

_VALID_DECISIONS: frozenset[str] = frozenset({"merge", "supersede", "keep_both"})


class DaemonProto(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _judge_pair(
    *, store: Store, llm: LLMProto, a: BodyRecord, b: BodyRecord
) -> tuple[str, str | None]:
    """Ask the LLM whether to merge, supersede, or keep both.

    Returns (decision, keeper_body_id). `keeper_body_id` is None for keep_both
    and for any failure mode (invalid JSON, unknown decision, unknown keeper,
    missing body file).
    """
    try:
        a_md = read_body_file(store.memory_dir / a.file_path).content
        b_md = read_body_file(store.memory_dir / b.file_path).content
    except FileNotFoundError:
        return ("keep_both", None)

    prompt = render(
        "review",
        a_body_id=a.body_id, a_updated=a.updated_at, a_body=a_md,
        b_body_id=b.body_id, b_updated=b.updated_at, b_body=b_md,
    )
    raw = llm.generate_chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
        thinking_level="minimal",
    )
    try:
        obj = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, TypeError):
        return ("keep_both", None)

    decision = obj.get("decision") if isinstance(obj, dict) else None
    if decision not in _VALID_DECISIONS:
        return ("keep_both", None)
    if decision == "keep_both":
        return ("keep_both", None)

    keeper = obj.get("keeper")
    if keeper not in (a.body_id, b.body_id):
        return ("keep_both", None)
    return (decision, keeper)
