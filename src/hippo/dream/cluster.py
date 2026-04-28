"""Cluster heads by embedding cosine similarity (single-link, threshold-based).

Returns list of clusters; each cluster is a list of head_ids that all
have at least one neighbor in the cluster above the cosine threshold.
This is intentionally simple — at our scale (~1000s of heads) it's fast
enough.
"""
from __future__ import annotations

import math
import sqlite3
import struct

from hippo.config import CLUSTER_COSINE_THRESHOLD, EMBEDDING_DIM


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{EMBEDDING_DIM}f", blob))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def cluster_active_heads(
    conn: sqlite3.Connection, threshold: float | None = None
) -> list[list[str]]:
    """Single-link clustering of all active heads by cosine similarity threshold.

    Returns list of clusters with len > 1 (singletons are skipped — no edges to propose).
    """
    th = CLUSTER_COSINE_THRESHOLD if threshold is None else threshold
    rows = conn.execute(
        """
        SELECT h.head_id, e.embedding
        FROM heads h JOIN head_embeddings e ON e.head_id = h.head_id
        WHERE h.archived = 0
        """
    ).fetchall()
    items = [(r["head_id"], _unpack(r["embedding"])) for r in rows]

    # Union-Find
    parent: dict[str, str] = {hid: hid for hid, _ in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    # Pairwise scan (n^2 — fine at our scale; Plan 6's contradiction-only pruning lives
    # within these clusters)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _cosine(items[i][1], items[j][1]) >= th:
                union(items[i][0], items[j][0])

    clusters: dict[str, list[str]] = {}
    for hid, _ in items:
        root = find(hid)
        clusters.setdefault(root, []).append(hid)

    return [c for c in clusters.values() if len(c) > 1]
