"""Read/write body markdown files with YAML frontmatter.

Frontmatter mirrors the SQLite metadata so a body file is self-describing
even if the DB is lost. The DB and file MUST stay in sync — every write to
bodies table should be paired with a write to the .md file (in the storage
layer above this module).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import frontmatter  # type: ignore[import-untyped]

from hippo.config import BODIES_SUBDIR


@dataclass
class BodyFile:
    body_id: str
    title: str
    scope: str
    created: datetime
    updated: datetime
    content: str


def _path_for(memory_dir: Path, body_id: str) -> Path:
    return memory_dir / BODIES_SUBDIR / f"{body_id}.md"


def write_body_file(memory_dir: Path, body: BodyFile) -> Path:
    """Write or overwrite the body markdown file, creating bodies/ if needed."""
    bodies_dir = memory_dir / BODIES_SUBDIR
    bodies_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "body_id": body.body_id,
        "title": body.title,
        "scope": body.scope,
        "created": body.created.isoformat(),
        "updated": body.updated.isoformat(),
    }
    post = frontmatter.Post(content=body.content, **metadata)
    path = _path_for(memory_dir, body.body_id)
    path.write_text(frontmatter.dumps(post))
    return path


def read_body_file(path: Path) -> BodyFile:
    """Load a body markdown file. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(path)
    post = frontmatter.load(path)
    return BodyFile(
        body_id=str(post["body_id"]),
        title=str(post["title"]),
        scope=str(post["scope"]),
        created=_parse_dt(post["created"]),
        updated=_parse_dt(post["updated"]),
        content=post.content,
    )


def _parse_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    return datetime.fromisoformat(str(value))
