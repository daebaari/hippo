"""Centralized configuration constants and path helpers.

Single source of truth for paths, thresholds, and tunable parameters.
Pipeline code must NEVER hardcode values that exist here.
"""
from __future__ import annotations

from pathlib import Path

# === Storage paths ===
CLAUDE_HOME = Path.home() / ".claude"
GLOBAL_MEMORY_DIR = CLAUDE_HOME / "memory"
PROJECTS_ROOT = CLAUDE_HOME / "projects"

DB_FILENAME = "memory.db"
BODIES_SUBDIR = "bodies"
LIGHT_LOCK_FILENAME = ".light-lock"
HEAVY_LOCK_FILENAME = ".heavy-lock"


def project_memory_dir(project: str) -> Path:
    """Return the memory dir for a given project name."""
    return PROJECTS_ROOT / project / "memory"


# === Embedding / model dimensions ===
EMBEDDING_DIM = 1024  # mxbai-embed-large-v1

# === Edge relations ===
EDGE_RELATIONS: tuple[str, ...] = (
    "related",
    "causes",
    "supersedes",
    "contradicts",
    "applies_when",
    # 'entity:<X>' is also valid, namespaced; not enumerated
)

SYMMETRIC_RELATIONS: frozenset[str] = frozenset({"related", "contradicts"})

# === Soft-delete / scoring ===
ATOM_SCORE_WEIGHT = 1.0
RAW_TURN_SCORE_WEIGHT = 0.6
EDGE_BOOST: dict[str, float] = {
    "contradicts": 1.3,
    "supersedes": 1.2,
    "applies_when": 1.1,
    "causes": 1.05,
    "related": 1.0,
}

# === Lock semantics ===
STALE_LOCK_AGE_SECONDS = 3600  # 1 hour
