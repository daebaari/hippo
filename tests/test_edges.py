"""Tests for edges (head-to-head graph) CRUD."""
from __future__ import annotations

import sqlite3

import pytest

from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.edges import (
    EdgeRecord,
    delete_edge,
    get_neighbors_1hop,
    insert_edge,
    insert_edge_with_reciprocal,
)
from hippo.storage.heads import HeadRecord, insert_head
from hippo.storage.migrations import run_migrations


@pytest.fixture
def conn(sqlite_conn: sqlite3.Connection) -> sqlite3.Connection:
    run_migrations(sqlite_conn)
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="b1",
            file_path="bodies/b1.md",
            title="t",
            scope="global",
            source="manual",
        ),
    )
    for hid in ["h1", "h2", "h3"]:
        insert_head(sqlite_conn, HeadRecord(head_id=hid, body_id="b1", summary=hid))
    return sqlite_conn


def test_insert_directed_edge(conn: sqlite3.Connection) -> None:
    insert_edge(conn, EdgeRecord(from_head="h1", to_head="h2", relation="causes"))
    neighbors = get_neighbors_1hop(conn, "h1")
    assert len(neighbors) == 1
    assert neighbors[0].to_head == "h2"
    assert neighbors[0].relation == "causes"


def test_get_neighbors_does_not_traverse_reverse(conn: sqlite3.Connection) -> None:
    insert_edge(conn, EdgeRecord(from_head="h1", to_head="h2", relation="causes"))
    # No reciprocal — h2 has no outgoing edge
    assert get_neighbors_1hop(conn, "h2") == []


def test_insert_edge_with_reciprocal_for_symmetric_relations(conn: sqlite3.Connection) -> None:
    insert_edge_with_reciprocal(
        conn, EdgeRecord(from_head="h1", to_head="h2", relation="related")
    )
    forward = get_neighbors_1hop(conn, "h1")
    backward = get_neighbors_1hop(conn, "h2")
    assert {n.to_head for n in forward} == {"h2"}
    assert {n.to_head for n in backward} == {"h1"}


def test_insert_edge_with_reciprocal_skips_for_asymmetric_relations(
    conn: sqlite3.Connection,
) -> None:
    insert_edge_with_reciprocal(
        conn, EdgeRecord(from_head="h1", to_head="h2", relation="causes")
    )
    forward = get_neighbors_1hop(conn, "h1")
    backward = get_neighbors_1hop(conn, "h2")
    assert {n.to_head for n in forward} == {"h2"}
    assert backward == []


def test_unique_constraint_prevents_duplicate(conn: sqlite3.Connection) -> None:
    insert_edge(conn, EdgeRecord(from_head="h1", to_head="h2", relation="causes"))
    with pytest.raises(sqlite3.IntegrityError):
        insert_edge(conn, EdgeRecord(from_head="h1", to_head="h2", relation="causes"))


def test_delete_edge_removes_only_targeted_relation(conn: sqlite3.Connection) -> None:
    insert_edge(conn, EdgeRecord(from_head="h1", to_head="h2", relation="causes"))
    insert_edge(conn, EdgeRecord(from_head="h1", to_head="h2", relation="applies_when"))
    delete_edge(conn, from_head="h1", to_head="h2", relation="causes")
    relations = {e.relation for e in get_neighbors_1hop(conn, "h1")}
    assert relations == {"applies_when"}
