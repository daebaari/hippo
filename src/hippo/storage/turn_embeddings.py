"""Turn-level embeddings: raw-turn searchability until heavy dream atomizes."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from hippo.storage.vec import pack_vector


@dataclass
class TurnSearchResult:
    turn_id: int
    capture_id: int
    summary: str
    distance: float


def insert_turn_embedding(
    conn: sqlite3.Connection,
    *,
    capture_id: int,
    summary: str,
    embedding: list[float],
) -> int:
    """Insert a turn-level embedding linked to a capture_queue row.
    Returns the new turn_id.
    """
    cur = conn.execute(
        "INSERT INTO turn_embeddings(capture_id, summary, created_at) "
        "VALUES (?, ?, ?)",
        (capture_id, summary, int(time.time())),
    )
    turn_id = int(cur.lastrowid or 0)
    blob = pack_vector(embedding)
    conn.execute(
        "INSERT INTO turn_embeddings_vec(turn_id, embedding) VALUES (?, ?)",
        (turn_id, blob),
    )
    conn.commit()
    return turn_id


def vector_search_turns(
    conn: sqlite3.Connection, query: list[float], top_k: int
) -> list[TurnSearchResult]:
    blob = pack_vector(query)
    rows = conn.execute(
        "SELECT v.turn_id AS turn_id, v.distance AS distance, "
        "       t.capture_id AS capture_id, t.summary AS summary "
        "FROM turn_embeddings_vec v "
        "JOIN turn_embeddings t ON t.turn_id = v.turn_id "
        "WHERE v.embedding MATCH ? AND k = ? "
        "ORDER BY v.distance",
        (blob, top_k),
    ).fetchall()
    return [
        TurnSearchResult(
            turn_id=int(r["turn_id"]),
            capture_id=int(r["capture_id"]),
            summary=r["summary"],
            distance=float(r["distance"]),
        )
        for r in rows
    ]


def delete_turn_embeddings_for_captures(
    conn: sqlite3.Connection, capture_ids: list[int]
) -> None:
    if not capture_ids:
        return
    placeholders = ",".join("?" for _ in capture_ids)
    # Find the turn_ids first so we can delete from both tables
    rows = conn.execute(
        f"SELECT turn_id FROM turn_embeddings WHERE capture_id IN ({placeholders})",
        tuple(capture_ids),
    ).fetchall()
    turn_ids = [int(r["turn_id"]) for r in rows]
    if turn_ids:
        tph = ",".join("?" for _ in turn_ids)
        conn.execute(f"DELETE FROM turn_embeddings_vec WHERE turn_id IN ({tph})", tuple(turn_ids))
        conn.execute(f"DELETE FROM turn_embeddings WHERE turn_id IN ({tph})", tuple(turn_ids))
    conn.commit()
