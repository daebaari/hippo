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


def _review_body_against_neighbors(
    *, store: Store, llm: LLMProto, body_id: str
) -> int:
    """Review a single body against its merge candidates. Returns 1 if archived, else 0.

    Always stamps last_reviewed_at on the body so the rolling slice advances.
    """
    from hippo.config import PRUNE_NEAREST_K, PRUNE_SIMILARITY_THRESHOLD
    from hippo.storage.bodies import archive_body as archive_body_fn
    from hippo.storage.bodies import (
        find_merge_candidates,
        get_body,
        update_last_reviewed_at,
    )
    from hippo.storage.heads import archive_head, list_heads_for_body

    self_rec = get_body(store.conn, body_id)
    if self_rec is None or self_rec.archived:
        return 0

    candidates = find_merge_candidates(
        store.conn,
        body_id=body_id,
        threshold=PRUNE_SIMILARITY_THRESHOLD,
        k=PRUNE_NEAREST_K,
    )

    archived = 0
    if not candidates:
        update_last_reviewed_at(store.conn, body_id)
        return 0

    for cand_rec, _sim in candidates:
        decision, keeper = _judge_pair(store=store, llm=llm, a=self_rec, b=cand_rec)
        if decision in ("merge", "supersede") and keeper is not None:
            loser_id = body_id if keeper == cand_rec.body_id else cand_rec.body_id
            winner_id = keeper
            reason_prefix = "merged_into" if decision == "merge" else "superseded_by"
            archive_reason = f"{reason_prefix}:{winner_id}"
            archive_body_fn(
                store.conn, loser_id, reason=archive_reason, in_favor_of=winner_id
            )
            for h in list_heads_for_body(store.conn, loser_id):
                archive_head(
                    store.conn, h.head_id, reason=f"body_archived:{archive_reason}"
                )
            archived = 1
            break

    update_last_reviewed_at(store.conn, body_id)
    return archived
