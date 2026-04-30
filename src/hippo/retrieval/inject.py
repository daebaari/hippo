"""Format retrieval results into the <memory> block injected by the hook."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from hippo.config import BODIES_SUBDIR
from hippo.retrieval.graph_expand import GraphHit
from hippo.retrieval.pipeline import RetrievalResult
from hippo.storage.body_files import read_body_file


def load_body_preview(memory_dir: Path, body_id: str, *, max_chars: int = 120) -> str | None:
    path = memory_dir / BODIES_SUBDIR / f"{body_id}.md"
    try:
        body = read_body_file(path)
    except FileNotFoundError:
        return None
    snippet = body.content.strip().replace("\n", " ")
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 1] + "…"
    return snippet


def format_memory_block(
    result: RetrievalResult, *, body_resolver: Callable[[GraphHit], str | None],
) -> str:
    if not result.heads:
        return ""
    lines = [
        "<memory>",
        "Memory candidates relevant to your input. Read summaries; load any body",
        "via Bash `hippo get <head_id>` if you need detail.",
        "",
    ]
    seen_bodies: set[str] = set()
    for hit in result.heads:
        scope_tag = f"[{hit.scope}]"
        rel = f" ({hit.edge_relation})" if hit.edge_relation else ""
        lines.append(f"{scope_tag} {hit.head_id} — {hit.head.summary}{rel}")
        if hit.head.body_id not in seen_bodies:
            preview = body_resolver(hit)
            if preview:
                lines.append(f"    ↳ body preview: {preview}")
            seen_bodies.add(hit.head.body_id)
    lines.append("</memory>")
    return "\n".join(lines) + "\n"
