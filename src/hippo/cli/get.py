"""memory-get <head_id>: print the body markdown for the given head."""
from __future__ import annotations

import argparse
import sys

from hippo.config import BODIES_SUBDIR
from hippo.storage.body_files import read_body_file
from hippo.storage.heads import get_head
from hippo.storage.multi_store import Scope, open_store


def get_body_cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="memory-get")
    p.add_argument("head_id")
    p.add_argument("--project", action="append", default=[],
                   help="Project scope(s) to search alongside global. Can repeat.")
    args = p.parse_args(argv)

    scopes = [Scope.global_()] + [Scope.project(proj) for proj in args.project]
    for scope in scopes:
        store = open_store(scope)
        try:
            head = get_head(store.conn, args.head_id)
            if head is None or head.archived:
                continue
            body_path = store.memory_dir / BODIES_SUBDIR / f"{head.body_id}.md"
            body = read_body_file(body_path)
            print(f"# {body.title}")
            print(f"scope: {body.scope}  body_id: {body.body_id}")
            print()
            print(body.content)
            return 0
        finally:
            store.conn.close()
    print(f"head_id {args.head_id} not found in any scope", file=sys.stderr)
    return 1


def main() -> int:
    return get_body_cli(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
