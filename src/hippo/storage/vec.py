"""Vector storage and similarity search via sqlite-vec.

Embeddings are stored in vec0 virtual tables. Vectors are passed as
float32 packed bytes (sqlite-vec's native binary format).
"""
from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass

from hippo.config import EMBEDDING_DIM


@dataclass
class HeadSearchResult:
    head_id: str
    distance: float  # cosine distance from query (lower = closer)


def pack_vector(vec: list[float]) -> bytes:
    """Pack a length-EMBEDDING_DIM float list into sqlite-vec's float32 binary format."""
    if len(vec) != EMBEDDING_DIM:
        raise ValueError(
            f"embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(vec)}"
        )
    return struct.pack(f"{EMBEDDING_DIM}f", *vec)


def insert_head_embedding(
    conn: sqlite3.Connection, head_id: str, embedding: list[float]
) -> None:
    blob = pack_vector(embedding)
    conn.execute(
        "INSERT INTO head_embeddings(head_id, embedding) VALUES (?, ?)",
        (head_id, blob),
    )
    conn.commit()


def delete_head_embedding(conn: sqlite3.Connection, head_id: str) -> None:
    conn.execute("DELETE FROM head_embeddings WHERE head_id = ?", (head_id,))
    conn.commit()


def vector_search_heads(
    conn: sqlite3.Connection, query: list[float], top_k: int
) -> list[HeadSearchResult]:
    """Return top_k closest heads by cosine distance (sqlite-vec uses L2 by default;
    we configure cosine via vec0 options if needed; for v1 L2 on normalized vectors
    is equivalent up to monotonic transformation)."""
    blob = pack_vector(query)
    rows = conn.execute(
        "SELECT head_id, distance "
        "FROM head_embeddings "
        "WHERE embedding MATCH ? "
        "ORDER BY distance "
        "LIMIT ?",
        (blob, top_k),
    ).fetchall()
    return [HeadSearchResult(head_id=r["head_id"], distance=float(r["distance"])) for r in rows]
