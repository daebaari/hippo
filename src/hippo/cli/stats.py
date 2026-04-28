"""memory-stats: print body/head/edge counts + recent dream runs per scope."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from hippo.storage.dream_runs import get_recent_runs
from hippo.storage.multi_store import Scope, open_store


def collect_stats(scopes: list[Scope]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for scope in scopes:
        store = open_store(scope)
        try:
            body_count = int(
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM bodies WHERE archived = 0"
                ).fetchone()["c"]
            )
            head_count = int(
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM heads WHERE archived = 0"
                ).fetchone()["c"]
            )
            edge_count = int(
                store.conn.execute("SELECT COUNT(*) AS c FROM edges").fetchone()["c"]
            )
            recent = get_recent_runs(store.conn, limit=5)
            out[scope.as_string()] = {
                "body_count": body_count,
                "head_count": head_count,
                "edge_count": edge_count,
                "recent_runs": [
                    {
                        "run_id": r.run_id,
                        "type": r.type,
                        "status": r.status,
                        "started_at": r.started_at,
                        "atoms_created": r.atoms_created,
                        "heads_created": r.heads_created,
                        "edges_created": r.edges_created,
                        "contradictions_resolved": r.contradictions_resolved,
                    }
                    for r in recent
                ],
            }
        finally:
            store.conn.close()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memory-stats")
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        help="Project name(s) to include alongside global. Can repeat.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of human format"
    )
    args = parser.parse_args(argv)

    scopes: list[Scope] = [Scope.global_()] + [Scope.project(p) for p in args.project]
    result = collect_stats(scopes)

    if args.json:
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    for scope_name, info in result.items():
        print(f"=== {scope_name} ===")
        print(f"  bodies (active):  {info['body_count']}")
        print(f"  heads  (active):  {info['head_count']}")
        print(f"  edges:            {info['edge_count']}")
        print(f"  recent dream runs:")
        for r in info["recent_runs"]:
            print(
                f"    [{r['type']:5}] run #{r['run_id']:>4} status={r['status']:9} "
                f"atoms={r['atoms_created']} heads={r['heads_created']} "
                f"edges={r['edges_created']} contradictions={r['contradictions_resolved']}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
