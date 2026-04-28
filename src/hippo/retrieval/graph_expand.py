"""Step 2 of retrieval: graph 1-hop expansion from vector seeds.

For each seed head, fetch its 1-hop neighbors (limited per-seed). Add new
heads to the candidate set with synthetic 'graph_distance' computed from
seed distance + edge weight. Edge type modifies a boost factor that's
applied later in the rerank step (we attach the relation here).
"""
from __future__ import annotations

from dataclasses import dataclass

from hippo.config import EDGE_BOOST
from hippo.retrieval.vector_search import VectorHit
from hippo.storage.edges import get_neighbors_1hop
from hippo.storage.heads import HeadRecord, get_head
from hippo.storage.multi_store import Scope, open_store


@dataclass
class GraphHit:
    head_id: str
    distance: float        # synthetic; lower = closer
    scope: str
    head: HeadRecord
    edge_relation: str | None  # None for vector seeds; the relation if added by graph hop


def _to_graph_hit(v: VectorHit) -> GraphHit:
    return GraphHit(head_id=v.head_id, distance=v.distance, scope=v.scope,
                    head=v.head, edge_relation=None)


def expand_via_graph(
    seeds: list[VectorHit], *,
    scopes: list[Scope],
    hop_limit_per_seed: int,
    total_cap: int,
) -> list[GraphHit]:
    """Take vector seed hits, return them plus 1-hop graph neighbors.

    Caps:
    - hop_limit_per_seed: max neighbors fetched per seed
    - total_cap: ceiling on total returned (seeds + expansions). Hard cap.
    """
    out: list[GraphHit] = [_to_graph_hit(s) for s in seeds]
    seen = {s.head_id for s in seeds}

    # Open all scoped stores once; close at the end.
    stores = {s.as_string(): open_store(s) for s in scopes}
    try:
        for seed in seeds:
            if len(out) >= total_cap:
                break
            store = stores.get(seed.scope)
            if store is None:
                continue
            edges = get_neighbors_1hop(store.conn, seed.head_id)
            # Sort by edge type's boost (descending) so high-value relations come first
            edges.sort(key=lambda e: -EDGE_BOOST.get(e.relation, 1.0))
            added_for_this_seed = 0
            for edge in edges:
                if added_for_this_seed >= hop_limit_per_seed:
                    break
                if len(out) >= total_cap:
                    break
                if edge.to_head in seen:
                    continue
                neighbor = get_head(store.conn, edge.to_head)
                if neighbor is None or neighbor.archived:
                    continue
                # Synthetic distance: seed distance + small offset (graph hits go after vector hits)
                out.append(GraphHit(
                    head_id=neighbor.head_id,
                    distance=seed.distance + 0.5,
                    scope=seed.scope,
                    head=neighbor,
                    edge_relation=edge.relation,
                ))
                seen.add(neighbor.head_id)
                added_for_this_seed += 1
    finally:
        for s in stores.values():
            s.conn.close()
    return out[:total_cap]
