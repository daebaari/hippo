"""memory-get <head_id>: print the body markdown for the given head."""
from __future__ import annotations

import argparse
import os
import sys

from hippo.cli.scope_args import add_scope_args, resolve_scopes
from hippo.config import BODIES_SUBDIR
from hippo.storage.body_files import read_body_file
from hippo.storage.heads import get_head
from hippo.storage.multi_store import open_store


def get_body_cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="memory-get")
    p.add_argument("head_id")
    add_scope_args(p, kind="targeted")
    args = p.parse_args(argv)

    scopes = resolve_scopes(args, kind="targeted", cwd=os.getcwd())
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
