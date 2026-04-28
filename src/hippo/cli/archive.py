"""memory-archive <head_id> --reason '...': soft-delete a head."""
from __future__ import annotations

import argparse
import sys

from hippo.storage.heads import archive_head, get_head
from hippo.storage.multi_store import Scope, open_store


def archive_head_cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="memory-archive")
    p.add_argument("head_id")
    p.add_argument("--reason", required=True)
    p.add_argument("--project", action="append", default=[])
    args = p.parse_args(argv)

    scopes = [Scope.global_()] + [Scope.project(proj) for proj in args.project]
    for scope in scopes:
        store = open_store(scope)
        try:
            head = get_head(store.conn, args.head_id)
            if head is None:
                continue
            archive_head(store.conn, args.head_id, reason=args.reason)
            print(f"archived {args.head_id} ({scope.as_string()}): {args.reason}")
            return 0
        finally:
            store.conn.close()
    print(f"head_id {args.head_id} not found", file=sys.stderr)
    return 1


def main() -> int:
    return archive_head_cli(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
