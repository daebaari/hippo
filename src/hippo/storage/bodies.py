"""CRUD for the bodies table.

bodies stores metadata for atom bodies; the actual content lives in
markdown files under <memory_dir>/bodies/<body_id>.md (managed by
body_files.py). Callers should keep the two in sync.
"""
from __future__ import annotations

import math
import sqlite3
import struct
import time
from dataclasses import dataclass


@dataclass
class BodyRecord:
    body_id: str
    file_path: str
    title: str
    scope: str
    source: str
    archived: bool = False
    archive_reason: str | None = None
    archived_in_favor_of: str | None = None
    created_at: int | None = None
    updated_at: int | None = None
    last_reviewed_at: int | None = None


def insert_body(conn: sqlite3.Connection, record: BodyRecord) -> None:
    now = record.created_at or int(time.time())
    conn.execute(
        "INSERT INTO bodies("
        "  body_id, file_path, title, scope, archived,"
        "  archive_reason, archived_in_favor_of, source, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            record.body_id,
            record.file_path,
            record.title,
            record.scope,
            int(record.archived),
            record.archive_reason,
            record.archived_in_favor_of,
            record.source,
            now,
            record.updated_at or now,
        ),
    )
    conn.commit()


def get_body(conn: sqlite3.Connection, body_id: str) -> BodyRecord | None:
    row = conn.execute("SELECT * FROM bodies WHERE body_id = ?", (body_id,)).fetchone()
    if row is None:
        return None
    return _row_to_record(row)


def archive_body(
    conn: sqlite3.Connection,
    body_id: str,
    *,
    reason: str,
    in_favor_of: str | None = None,
) -> None:
    conn.execute(
        "UPDATE bodies SET archived = 1, archive_reason = ?, "
        "archived_in_favor_of = ?, updated_at = ? WHERE body_id = ?",
        (reason, in_favor_of, int(time.time()), body_id),
    )
    conn.commit()


def list_bodies_by_scope(conn: sqlite3.Connection, scope: str) -> list[BodyRecord]:
    rows = conn.execute(
        "SELECT * FROM bodies WHERE scope = ? AND archived = 0 ORDER BY updated_at DESC",
        (scope,),
    ).fetchall()
    return [_row_to_record(r) for r in rows]


def update_last_reviewed_at(conn: sqlite3.Connection, body_id: str) -> None:
    """Stamp last_reviewed_at = now for a body."""
    conn.execute(
        "UPDATE bodies SET last_reviewed_at = ? WHERE body_id = ?",
        (int(time.time()), body_id),
    )
    conn.commit()


def find_oldest_unreviewed_active(
    conn: sqlite3.Connection, *, scope: str, limit: int
) -> list[BodyRecord]:
    """Active bodies in scope, NULL last_reviewed_at first, then oldest first.

    Tie-breaks deterministically by body_id ASC.
    """
    rows = conn.execute(
        "SELECT * FROM bodies "
        "WHERE archived = 0 AND scope = ? "
        "ORDER BY COALESCE(last_reviewed_at, 0) ASC, body_id ASC "
        "LIMIT ?",
        (scope, limit),
    ).fetchall()
    return [_row_to_record(r) for r in rows]


def find_active_bodies_by_run_source(
    conn: sqlite3.Connection, *, run_id: int
) -> list[BodyRecord]:
    """Active bodies inserted by this heavy-dream run (by source string)."""
    rows = conn.execute(
        "SELECT * FROM bodies "
        "WHERE archived = 0 AND source = ? "
        "ORDER BY body_id ASC",
        (f"heavy-dream-run:{run_id}",),
    ).fetchall()
    return [_row_to_record(r) for r in rows]


def count_eligible_for_multi_head(
    conn: sqlite3.Connection, *, target_total_heads: int = 3
) -> int:
    """Count bodies that multi_head expansion will process this run.

    Mirrors the eligibility filter in expand_heads_for_eligible_bodies:
    archived=0, < target_total_heads active heads, and at least one head
    with retrieval_count > 0. Used solely for the progress denominator.
    """
    row = conn.execute(
        """
        SELECT COUNT(*) AS c FROM (
            SELECT b.body_id
            FROM bodies b
            LEFT JOIN heads h ON h.body_id = b.body_id AND h.archived = 0
            WHERE b.archived = 0
            GROUP BY b.body_id
            HAVING COUNT(h.head_id) < ?
              AND (
                  SELECT MAX(retrieval_count) FROM heads WHERE body_id = b.body_id
              ) > 0
        )
        """,
        (target_total_heads,),
    ).fetchone()
    return int(row["c"]) if row else 0


def find_merge_candidates(
    conn: sqlite3.Connection,
    *,
    body_id: str,
    threshold: float,
    k: int,
) -> list[tuple[BodyRecord, float]]:
    """Active bodies whose nearest head to any of body_id's heads has cosine >= threshold.

    Returns up to k items, sorted by max similarity DESC. Excludes body_id itself,
    archived bodies, and bodies whose only candidacy is below threshold.
    """
    self_rows = conn.execute(
        "SELECT he.embedding "
        "FROM heads h JOIN head_embeddings he ON he.head_id = h.head_id "
        "WHERE h.body_id = ? AND h.archived = 0",
        (body_id,),
    ).fetchall()
    if not self_rows:
        return []
    self_vecs = [_unpack_embedding(r["embedding"]) for r in self_rows]

    other_rows = conn.execute(
        "SELECT h.body_id AS cand_body_id, he.embedding "
        "FROM heads h JOIN head_embeddings he ON he.head_id = h.head_id "
        "JOIN bodies b ON b.body_id = h.body_id "
        "WHERE h.archived = 0 AND b.archived = 0 AND h.body_id != ? "
        "  AND b.body_id != ?",
        (body_id, body_id),
    ).fetchall()

    best_sim_per_body: dict[str, float] = {}
    for r in other_rows:
        cand_vec = _unpack_embedding(r["embedding"])
        max_sim = max(_cosine_similarity(sv, cand_vec) for sv in self_vecs)
        cand_body = r["cand_body_id"]
        prior = best_sim_per_body.get(cand_body, -2.0)
        if max_sim > prior:
            best_sim_per_body[cand_body] = max_sim

    qualifying = [(cand_id, sim) for cand_id, sim in best_sim_per_body.items() if sim >= threshold]
    qualifying.sort(key=lambda t: t[1], reverse=True)
    qualifying = qualifying[:k]

    out: list[tuple[BodyRecord, float]] = []
    for cand_body_id, sim in qualifying:
        rec = get_body(conn, cand_body_id)
        if rec is not None and not rec.archived:
            out.append((rec, sim))
    return out


def _unpack_embedding(blob: bytes) -> list[float]:
    from hippo.config import EMBEDDING_DIM
    return list(struct.unpack(f"{EMBEDDING_DIM}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _row_to_record(row: sqlite3.Row) -> BodyRecord:
    return BodyRecord(
        body_id=row["body_id"],
        file_path=row["file_path"],
        title=row["title"],
        scope=row["scope"],
        archived=bool(row["archived"]),
        archive_reason=row["archive_reason"],
        archived_in_favor_of=row["archived_in_favor_of"],
        source=row["source"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_reviewed_at=row["last_reviewed_at"],
    )
