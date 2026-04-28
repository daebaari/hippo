"""CRUD for the edges table (head-to-head typed graph).

Edges are directed by default. For symmetric relations
(SYMMETRIC_RELATIONS in config), insert_edge_with_reciprocal writes both
directions so 1-hop traversal works from either side.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from hippo.config import SYMMETRIC_RELATIONS


@dataclass
class EdgeRecord:
    from_head: str
    to_head: str
    relation: str
    weight: float = 1.0
    created_at: int | None = None


def insert_edge(conn: sqlite3.Connection, edge: EdgeRecord) -> None:
    now = edge.created_at or int(time.time())
    conn.execute(
        "INSERT INTO edges(from_head, to_head, relation, weight, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (edge.from_head, edge.to_head, edge.relation, edge.weight, now),
    )
    conn.commit()


def insert_edge_with_reciprocal(conn: sqlite3.Connection, edge: EdgeRecord) -> None:
    """For symmetric relations, also insert the reverse edge."""
    insert_edge(conn, edge)
    if edge.relation in SYMMETRIC_RELATIONS:
        try:
            insert_edge(
                conn,
                EdgeRecord(
                    from_head=edge.to_head,
                    to_head=edge.from_head,
                    relation=edge.relation,
                    weight=edge.weight,
                    created_at=edge.created_at,
                ),
            )
        except sqlite3.IntegrityError:
            # Reciprocal already exists — fine
            pass


def get_neighbors_1hop(conn: sqlite3.Connection, head_id: str) -> list[EdgeRecord]:
    rows = conn.execute(
        "SELECT from_head, to_head, relation, weight, created_at "
        "FROM edges WHERE from_head = ?",
        (head_id,),
    ).fetchall()
    return [
        EdgeRecord(
            from_head=r["from_head"],
            to_head=r["to_head"],
            relation=r["relation"],
            weight=float(r["weight"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]


def delete_edge(
    conn: sqlite3.Connection, *, from_head: str, to_head: str, relation: str
) -> None:
    conn.execute(
        "DELETE FROM edges WHERE from_head = ? AND to_head = ? AND relation = ?",
        (from_head, to_head, relation),
    )
    conn.commit()
