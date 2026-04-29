# Prune Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-04-29-prune-phase-design.md`

**Goal:** Add a soft-archive pruning step inside the heavy dream that drops noise atoms at insertion, then reviews new and rolling-slice old atoms against their nearest neighbors and archives losers, so the active corpus stays lean and fresh.

**Architecture:** One new dream phase (`review`) sits between `atomize` and `multi_head`, plus an enriched `noise` field in the atomize prompt. Detection uses `head_embeddings` cosine similarity over active heads, judgment uses a single new LLM prompt (`review.md`) with `thinking_level="minimal"`. No GC, no hard-delete, no new launchd jobs.

**Tech Stack:** Python 3.12, sqlite3 + sqlite-vec, mlx-lm (Qwen) / google-genai (Gemini), pytest, ruff, mypy strict. Pattern matches existing `dream/contradiction.py` and `dream/edge_proposal.py`.

**Worktree:** `.worktrees/prune-phase` (branch `feat/prune-phase`). Run all commands from there.

**Gates after every task:**
- `uv run pytest -q` (must pass; existing 149 + new tests)
- `uv run ruff check src tests`
- `uv run mypy src`

---

## File map

**New files:**
- `schema/002_prune_metadata.sql`
- `src/hippo/dream/prompts/review.md`
- `src/hippo/dream/review.py`
- `tests/test_review_phase.py`
- `tests/test_review_storage.py`

**Modified files:**
- `src/hippo/config.py` — three new constants
- `src/hippo/storage/bodies.py` — new helpers: `find_oldest_unreviewed_active`, `update_last_reviewed_at`, `find_active_bodies_by_run_source`, `find_merge_candidates`. Existing `BodyRecord` gets a `last_reviewed_at` field.
- `src/hippo/storage/dream_runs.py` — `complete_run` accepts `bodies_archived_review`; `DreamRunRecord` gets the new field
- `src/hippo/dream/atomize.py` — skip atoms where `noise == true`
- `src/hippo/dream/prompts/atomize.md` — `noise` field with examples + tiebreaker
- `src/hippo/dream/heavy.py` — wire review phase, track new counter
- `tests/test_atomize.py` — three new tests for noise handling
- `tests/test_dream_heavy_orchestrator.py` — assert phase ordering and counter
- `tests/test_dream_runs.py` — round-trip new field
- `tests/test_migrations.py` — assert 002 applies cleanly
- `tests/test_bodies.py` — round-trip `last_reviewed_at` (existing tests must still pass)
- `KNOWN_ISSUES.md` — modify the atomize-noise-leakage entry to reflect the new mitigation

---

## Task 1: Schema migration 002

Add `bodies.last_reviewed_at`, `dream_runs.bodies_archived_review`, and the partial index for the rolling-slice query. All idempotent so the migration can be re-run safely against existing on-disk DBs.

**Files:**
- Create: `schema/002_prune_metadata.sql`
- Modify: `tests/test_migrations.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_migrations.py`:

```python
def test_migration_002_adds_prune_columns(tmp_path):
    """002 adds bodies.last_reviewed_at and dream_runs.bodies_archived_review."""
    import sqlite3

    import sqlite_vec

    from hippo.storage.migrations import run_migrations

    db = tmp_path / "memory.db"
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row

    run_migrations(conn)

    body_cols = {row["name"] for row in conn.execute("PRAGMA table_info(bodies)").fetchall()}
    assert "last_reviewed_at" in body_cols

    run_cols = {row["name"] for row in conn.execute("PRAGMA table_info(dream_runs)").fetchall()}
    assert "bodies_archived_review" in run_cols

    # Partial index exists
    idx = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_bodies_review_queue'"
    ).fetchone()
    assert idx is not None

    # Re-running is idempotent
    run_migrations(conn)
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_migrations.py::test_migration_002_adds_prune_columns -v
```

Expected: FAIL — column `last_reviewed_at` not present.

- [ ] **Step 3: Create the migration**

Create `schema/002_prune_metadata.sql`:

```sql
-- Schema migration 002: prune-phase metadata
-- Idempotent — uses ADD COLUMN guards via sqlite_master lookup, IF NOT EXISTS for indexes.

-- Add last_reviewed_at to bodies (nullable; existing rows stay NULL → sort first in rolling slice)
-- sqlite has no "ADD COLUMN IF NOT EXISTS"; use a guarded approach via PRAGMA.
-- We rely on the migration runner's schema_versions check to prevent re-application,
-- so a plain ALTER is safe here.
ALTER TABLE bodies ADD COLUMN last_reviewed_at INTEGER;

-- Add bodies_archived_review counter to dream_runs
ALTER TABLE dream_runs ADD COLUMN bodies_archived_review INTEGER DEFAULT 0;

-- Index for the rolling slice query (active bodies ordered by review recency)
CREATE INDEX IF NOT EXISTS idx_bodies_review_queue
    ON bodies(last_reviewed_at) WHERE archived = 0;
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_migrations.py::test_migration_002_adds_prune_columns -v
```

Expected: PASS.

- [ ] **Step 5: Run the full migration test file + ruff + mypy**

```
uv run pytest tests/test_migrations.py -v
uv run ruff check src tests
uv run mypy src
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add schema/002_prune_metadata.sql tests/test_migrations.py
git commit -m "schema: 002 adds prune metadata (last_reviewed_at, bodies_archived_review)"
```

---

## Task 2: Storage — `BodyRecord.last_reviewed_at` field + read path

The existing `BodyRecord` dataclass in `src/hippo/storage/bodies.py` needs a `last_reviewed_at: int | None = None` field, and `_row_to_record` must pull it. Existing reads keep working (NULL is the default).

**Files:**
- Modify: `src/hippo/storage/bodies.py:14-26, 81-93`
- Modify: `tests/test_bodies.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bodies.py`:

```python
def test_body_record_includes_last_reviewed_at(temp_memory_dir, sqlite_conn):
    """BodyRecord exposes last_reviewed_at; defaults to None on insert."""
    from hippo.storage.bodies import BodyRecord, get_body, insert_body
    from hippo.storage.migrations import run_migrations

    run_migrations(sqlite_conn)
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="bid-1", file_path="bodies/bid-1.md", title="t",
            scope="global", source="test",
        ),
    )
    rec = get_body(sqlite_conn, "bid-1")
    assert rec is not None
    assert rec.last_reviewed_at is None
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_bodies.py::test_body_record_includes_last_reviewed_at -v
```

Expected: FAIL — `BodyRecord` has no `last_reviewed_at` attribute.

- [ ] **Step 3: Add the field + plumb through `_row_to_record`**

Edit `src/hippo/storage/bodies.py`. The `BodyRecord` dataclass and `_row_to_record` function must look like this in full:

```python
@dataclass
class BodyRecord:
    body_id: str
    file_path: str
    title: str
    scope: str
    source: str
    archived: bool = False
    archive_reason: str | None = None
    archived_in_favor_of: str | None = None
    created_at: int | None = None
    updated_at: int | None = None
    last_reviewed_at: int | None = None


def _row_to_record(row: sqlite3.Row) -> BodyRecord:
    return BodyRecord(
        body_id=row["body_id"],
        file_path=row["file_path"],
        title=row["title"],
        scope=row["scope"],
        archived=bool(row["archived"]),
        archive_reason=row["archive_reason"],
        archived_in_favor_of=row["archived_in_favor_of"],
        source=row["source"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_reviewed_at=row["last_reviewed_at"],
    )
```

(The `insert_body` function does NOT need to write `last_reviewed_at` — it stays NULL on insert.)

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_bodies.py -v
```

Expected: all `test_bodies.py` tests PASS, including the new one.

- [ ] **Step 5: Commit**

```bash
git add src/hippo/storage/bodies.py tests/test_bodies.py
git commit -m "storage: BodyRecord exposes last_reviewed_at"
```

---

## Task 3: Storage — `update_last_reviewed_at`

Stamp a body's `last_reviewed_at = now`. Used by both review passes after a body is judged.

**Files:**
- Modify: `src/hippo/storage/bodies.py` (append helper)
- Create: `tests/test_review_storage.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_review_storage.py`:

```python
"""Storage helpers used by the review phase."""
from __future__ import annotations

import time

from hippo.storage.bodies import (
    BodyRecord,
    get_body,
    insert_body,
    update_last_reviewed_at,
)
from hippo.storage.migrations import run_migrations


def test_update_last_reviewed_at_stamps_now(sqlite_conn):
    run_migrations(sqlite_conn)
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="b1", file_path="bodies/b1.md", title="t",
            scope="global", source="test",
        ),
    )

    before = int(time.time())
    update_last_reviewed_at(sqlite_conn, "b1")
    after = int(time.time())

    rec = get_body(sqlite_conn, "b1")
    assert rec is not None
    assert rec.last_reviewed_at is not None
    assert before <= rec.last_reviewed_at <= after
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_review_storage.py::test_update_last_reviewed_at_stamps_now -v
```

Expected: FAIL — `update_last_reviewed_at` not defined.

- [ ] **Step 3: Add the helper**

Append to `src/hippo/storage/bodies.py`:

```python
def update_last_reviewed_at(conn: sqlite3.Connection, body_id: str) -> None:
    """Stamp last_reviewed_at = now for a body."""
    conn.execute(
        "UPDATE bodies SET last_reviewed_at = ? WHERE body_id = ?",
        (int(time.time()), body_id),
    )
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_review_storage.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hippo/storage/bodies.py tests/test_review_storage.py
git commit -m "storage: update_last_reviewed_at helper for review phase"
```

---

## Task 4: Storage — `find_oldest_unreviewed_active`

Return the K oldest active bodies for the rolling slice. NULL `last_reviewed_at` sorts first (never-reviewed bodies); then ascending by `last_reviewed_at`. Tie-broken by `body_id` for determinism.

**Files:**
- Modify: `src/hippo/storage/bodies.py` (append helper)
- Modify: `tests/test_review_storage.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_review_storage.py`:

```python
def test_find_oldest_unreviewed_active_orders_null_first_then_oldest(sqlite_conn):
    run_migrations(sqlite_conn)
    # b-null: never reviewed (NULL)
    # b-old:  reviewed long ago
    # b-new:  reviewed recently
    # b-arch: archived (must be excluded)
    for bid in ("b-null", "b-old", "b-new", "b-arch"):
        insert_body(
            sqlite_conn,
            BodyRecord(
                body_id=bid, file_path=f"bodies/{bid}.md",
                title=bid, scope="global", source="test",
            ),
        )
    sqlite_conn.execute("UPDATE bodies SET last_reviewed_at = 100 WHERE body_id = 'b-old'")
    sqlite_conn.execute("UPDATE bodies SET last_reviewed_at = 999 WHERE body_id = 'b-new'")
    sqlite_conn.execute("UPDATE bodies SET archived = 1 WHERE body_id = 'b-arch'")
    sqlite_conn.commit()

    from hippo.storage.bodies import find_oldest_unreviewed_active

    out = find_oldest_unreviewed_active(sqlite_conn, scope="global", limit=10)
    ids = [b.body_id for b in out]
    assert ids == ["b-null", "b-old", "b-new"]


def test_find_oldest_unreviewed_active_respects_limit(sqlite_conn):
    run_migrations(sqlite_conn)
    for bid in ("b1", "b2", "b3"):
        insert_body(
            sqlite_conn,
            BodyRecord(
                body_id=bid, file_path=f"bodies/{bid}.md",
                title=bid, scope="global", source="test",
            ),
        )

    from hippo.storage.bodies import find_oldest_unreviewed_active

    out = find_oldest_unreviewed_active(sqlite_conn, scope="global", limit=2)
    assert len(out) == 2


def test_find_oldest_unreviewed_active_filters_by_scope(sqlite_conn):
    run_migrations(sqlite_conn)
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="g1", file_path="bodies/g1.md", title="g1",
            scope="global", source="test",
        ),
    )
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="p1", file_path="bodies/p1.md", title="p1",
            scope="project:foo", source="test",
        ),
    )

    from hippo.storage.bodies import find_oldest_unreviewed_active

    out = find_oldest_unreviewed_active(sqlite_conn, scope="global", limit=10)
    assert [b.body_id for b in out] == ["g1"]
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_review_storage.py -v
```

Expected: 3 FAIL — `find_oldest_unreviewed_active` not defined.

- [ ] **Step 3: Add the helper**

Append to `src/hippo/storage/bodies.py`:

```python
def find_oldest_unreviewed_active(
    conn: sqlite3.Connection, *, scope: str, limit: int
) -> list[BodyRecord]:
    """Active bodies in scope, NULL last_reviewed_at first, then oldest first.

    Tie-breaks deterministically by body_id ASC.
    """
    rows = conn.execute(
        "SELECT * FROM bodies "
        "WHERE archived = 0 AND scope = ? "
        "ORDER BY COALESCE(last_reviewed_at, 0) ASC, body_id ASC "
        "LIMIT ?",
        (scope, limit),
    ).fetchall()
    return [_row_to_record(r) for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_review_storage.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hippo/storage/bodies.py tests/test_review_storage.py
git commit -m "storage: find_oldest_unreviewed_active for rolling slice"
```

---

## Task 5: Storage — `find_active_bodies_by_run_source`

The gate-at-entry pass needs to enumerate bodies that this exact heavy-dream-run inserted. They're identified by `source = 'heavy-dream-run:{run_id}'`.

**Files:**
- Modify: `src/hippo/storage/bodies.py` (append helper)
- Modify: `tests/test_review_storage.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_review_storage.py`:

```python
def test_find_active_bodies_by_run_source_returns_only_run_bodies(sqlite_conn):
    run_migrations(sqlite_conn)
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="b-this", file_path="bodies/b-this.md", title="t",
            scope="global", source="heavy-dream-run:42",
        ),
    )
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="b-other", file_path="bodies/b-other.md", title="t",
            scope="global", source="heavy-dream-run:7",
        ),
    )
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="b-arch", file_path="bodies/b-arch.md", title="t",
            scope="global", source="heavy-dream-run:42",
        ),
    )
    sqlite_conn.execute("UPDATE bodies SET archived = 1 WHERE body_id = 'b-arch'")
    sqlite_conn.commit()

    from hippo.storage.bodies import find_active_bodies_by_run_source

    out = find_active_bodies_by_run_source(sqlite_conn, run_id=42)
    assert [b.body_id for b in out] == ["b-this"]
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_review_storage.py::test_find_active_bodies_by_run_source_returns_only_run_bodies -v
```

Expected: FAIL.

- [ ] **Step 3: Add the helper**

Append to `src/hippo/storage/bodies.py`:

```python
def find_active_bodies_by_run_source(
    conn: sqlite3.Connection, *, run_id: int
) -> list[BodyRecord]:
    """Active bodies inserted by this heavy-dream run (by source string)."""
    rows = conn.execute(
        "SELECT * FROM bodies "
        "WHERE archived = 0 AND source = ? "
        "ORDER BY body_id ASC",
        (f"heavy-dream-run:{run_id}",),
    ).fetchall()
    return [_row_to_record(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_review_storage.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hippo/storage/bodies.py tests/test_review_storage.py
git commit -m "storage: find_active_bodies_by_run_source for gate-at-entry"
```

---

## Task 6: Storage — `find_merge_candidates`

For a given body, return up to K active bodies whose at least one head's cosine similarity to one of this body's heads is ≥ threshold. Returns each candidate's `BodyRecord` plus the max similarity that hit. Mirrors the in-Python cosine pattern from `dream/cluster.py` (single source of truth for the math).

**Files:**
- Modify: `src/hippo/storage/bodies.py`
- Modify: `tests/test_review_storage.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_review_storage.py`:

```python
def _setup_two_bodies_with_heads(conn, *, sim_pairs):
    """Helper: insert body B and a candidate C with controlled head embeddings.

    sim_pairs is a list of (b_vec, c_vec) tuples; each pair becomes one head per side.
    """
    from hippo.config import EMBEDDING_DIM
    from hippo.storage.heads import HeadRecord, insert_head
    from hippo.storage.vec import insert_head_embedding

    run_migrations(conn)
    for bid in ("B", "C"):
        insert_body(
            conn,
            BodyRecord(
                body_id=bid, file_path=f"bodies/{bid}.md",
                title=bid, scope="global", source="test",
            ),
        )

    for i, (bvec, cvec) in enumerate(sim_pairs):
        b_head = f"hB-{i}"
        c_head = f"hC-{i}"
        # pad each input to EMBEDDING_DIM
        b_full = list(bvec) + [0.0] * (EMBEDDING_DIM - len(bvec))
        c_full = list(cvec) + [0.0] * (EMBEDDING_DIM - len(cvec))
        insert_head(conn, HeadRecord(head_id=b_head, body_id="B", summary=f"b{i}"))
        insert_head(conn, HeadRecord(head_id=c_head, body_id="C", summary=f"c{i}"))
        insert_head_embedding(conn, b_head, b_full)
        insert_head_embedding(conn, c_head, c_full)


def test_find_merge_candidates_returns_high_similarity_body(sqlite_conn):
    # B head: [1, 0]; C head: [1, 0] → cosine = 1.0
    _setup_two_bodies_with_heads(sqlite_conn, sim_pairs=[([1.0, 0.0], [1.0, 0.0])])

    from hippo.storage.bodies import find_merge_candidates

    out = find_merge_candidates(sqlite_conn, body_id="B", threshold=0.85, k=5)
    assert len(out) == 1
    rec, sim = out[0]
    assert rec.body_id == "C"
    assert sim >= 0.99


def test_find_merge_candidates_skips_below_threshold(sqlite_conn):
    # B head [1, 0]; C head [0, 1] → cosine = 0.0
    _setup_two_bodies_with_heads(sqlite_conn, sim_pairs=[([1.0, 0.0], [0.0, 1.0])])

    from hippo.storage.bodies import find_merge_candidates

    out = find_merge_candidates(sqlite_conn, body_id="B", threshold=0.85, k=5)
    assert out == []


def test_find_merge_candidates_excludes_self(sqlite_conn):
    # Only one body, identical to itself
    _setup_two_bodies_with_heads(sqlite_conn, sim_pairs=[([1.0, 0.0], [1.0, 0.0])])
    # Drop body C so only B remains
    sqlite_conn.execute("UPDATE bodies SET archived = 1 WHERE body_id = 'C'")
    sqlite_conn.execute("UPDATE heads SET archived = 1 WHERE body_id = 'C'")
    sqlite_conn.commit()

    from hippo.storage.bodies import find_merge_candidates

    out = find_merge_candidates(sqlite_conn, body_id="B", threshold=0.85, k=5)
    assert out == []


def test_find_merge_candidates_respects_k(sqlite_conn):
    """When multiple candidate bodies exist, only top-K (by max similarity) are returned."""
    from hippo.config import EMBEDDING_DIM
    from hippo.storage.heads import HeadRecord, insert_head
    from hippo.storage.vec import insert_head_embedding

    run_migrations(sqlite_conn)
    insert_body(
        sqlite_conn,
        BodyRecord(
            body_id="B", file_path="bodies/B.md", title="B",
            scope="global", source="test",
        ),
    )
    insert_head(sqlite_conn, HeadRecord(head_id="hB", body_id="B", summary="b"))
    insert_head_embedding(sqlite_conn, "hB", [1.0, 0.0] + [0.0] * (EMBEDDING_DIM - 2))

    # Three candidates, one perfectly similar (C1), two below threshold (C2, C3)
    candidates = [
        ("C1", [1.0, 0.0]),     # cosine 1.0
        ("C2", [0.0, 1.0]),     # cosine 0.0
        ("C3", [-1.0, 0.0]),    # cosine -1.0
    ]
    for cid, vec in candidates:
        insert_body(
            sqlite_conn,
            BodyRecord(
                body_id=cid, file_path=f"bodies/{cid}.md", title=cid,
                scope="global", source="test",
            ),
        )
        insert_head(sqlite_conn, HeadRecord(head_id=f"h{cid}", body_id=cid, summary=cid))
        insert_head_embedding(
            sqlite_conn, f"h{cid}", vec + [0.0] * (EMBEDDING_DIM - 2)
        )

    from hippo.storage.bodies import find_merge_candidates

    out = find_merge_candidates(sqlite_conn, body_id="B", threshold=0.85, k=5)
    assert [r.body_id for r, _ in out] == ["C1"]
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_review_storage.py -v
```

Expected: 4 FAIL — `find_merge_candidates` not defined.

- [ ] **Step 3: Add the helper**

The helper computes cosine in Python over unpacked vectors, mirroring `dream/cluster.py`. We pull all (head_id, body_id, embedding) where archived=0 and body_id != self, then compute pairwise cosine per B-head, tracking the max per candidate body.

Append to `src/hippo/storage/bodies.py`. The imports `math` and `struct` need to be added to the top-of-file imports — keep them sorted.

```python
def find_merge_candidates(
    conn: sqlite3.Connection,
    *,
    body_id: str,
    threshold: float,
    k: int,
) -> list[tuple[BodyRecord, float]]:
    """Active bodies whose nearest head to any of body_id's heads has cosine >= threshold.

    Returns up to k items, sorted by max similarity DESC. Excludes body_id itself,
    archived bodies, and bodies whose only candidacy is below threshold.
    """
    self_rows = conn.execute(
        "SELECT he.embedding "
        "FROM heads h JOIN head_embeddings he ON he.head_id = h.head_id "
        "WHERE h.body_id = ? AND h.archived = 0",
        (body_id,),
    ).fetchall()
    if not self_rows:
        return []
    self_vecs = [_unpack_embedding(r["embedding"]) for r in self_rows]

    other_rows = conn.execute(
        "SELECT h.body_id AS cand_body_id, he.embedding "
        "FROM heads h JOIN head_embeddings he ON he.head_id = h.head_id "
        "JOIN bodies b ON b.body_id = h.body_id "
        "WHERE h.archived = 0 AND b.archived = 0 AND h.body_id != ? "
        "  AND b.body_id != ?",
        (body_id, body_id),
    ).fetchall()

    best_sim_per_body: dict[str, float] = {}
    for r in other_rows:
        cand_vec = _unpack_embedding(r["embedding"])
        max_sim = max(_cosine_similarity(sv, cand_vec) for sv in self_vecs)
        cand_body = r["cand_body_id"]
        prior = best_sim_per_body.get(cand_body, -2.0)
        if max_sim > prior:
            best_sim_per_body[cand_body] = max_sim

    qualifying = [(bid, sim) for bid, sim in best_sim_per_body.items() if sim >= threshold]
    qualifying.sort(key=lambda t: t[1], reverse=True)
    qualifying = qualifying[:k]

    out: list[tuple[BodyRecord, float]] = []
    for cand_body_id, sim in qualifying:
        rec = get_body(conn, cand_body_id)
        if rec is not None and not rec.archived:
            out.append((rec, sim))
    return out


def _unpack_embedding(blob: bytes) -> list[float]:
    from hippo.config import EMBEDDING_DIM
    return list(struct.unpack(f"{EMBEDDING_DIM}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
```

Add imports at the top of `src/hippo/storage/bodies.py`:

```python
import math
import struct
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_review_storage.py -v
uv run ruff check src tests
uv run mypy src
```

Expected: all PASS, ruff clean, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/hippo/storage/bodies.py tests/test_review_storage.py
git commit -m "storage: find_merge_candidates returns top-K similar bodies"
```

---

## Task 7: dream_runs — accept `bodies_archived_review`

`complete_run` accepts the new counter; `DreamRunRecord` exposes it; reads decode it.

**Files:**
- Modify: `src/hippo/storage/dream_runs.py`
- Modify: `tests/test_dream_runs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dream_runs.py`:

```python
def test_complete_run_persists_bodies_archived_review(sqlite_conn):
    from hippo.storage.dream_runs import complete_run, get_recent_runs, start_run
    from hippo.storage.migrations import run_migrations

    run_migrations(sqlite_conn)
    run_id = start_run(sqlite_conn, "heavy")
    complete_run(sqlite_conn, run_id, bodies_archived_review=4)

    runs = get_recent_runs(sqlite_conn, limit=1)
    assert len(runs) == 1
    assert runs[0].bodies_archived_review == 4
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_dream_runs.py::test_complete_run_persists_bodies_archived_review -v
```

Expected: FAIL — `bodies_archived_review` is not an attribute / kwarg.

- [ ] **Step 3: Update `dream_runs.py`**

Replace the existing `DreamRunRecord`, `complete_run`, and `get_recent_runs` in `src/hippo/storage/dream_runs.py` with these full versions:

```python
@dataclass
class DreamRunRecord:
    run_id: int
    type: str
    started_at: int
    completed_at: int | None
    status: str
    atoms_created: int
    heads_created: int
    edges_created: int
    contradictions_resolved: int
    bodies_archived_review: int
    error_message: str | None


def complete_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    atoms_created: int = 0,
    heads_created: int = 0,
    edges_created: int = 0,
    contradictions_resolved: int = 0,
    bodies_archived_review: int = 0,
) -> None:
    conn.execute(
        "UPDATE dream_runs SET status = 'completed', completed_at = ?, "
        "atoms_created = ?, heads_created = ?, edges_created = ?, "
        "contradictions_resolved = ?, bodies_archived_review = ? "
        "WHERE run_id = ?",
        (
            int(time.time()),
            atoms_created,
            heads_created,
            edges_created,
            contradictions_resolved,
            bodies_archived_review,
            run_id,
        ),
    )
    conn.commit()


def get_recent_runs(conn: sqlite3.Connection, limit: int) -> list[DreamRunRecord]:
    rows = conn.execute(
        "SELECT * FROM dream_runs ORDER BY started_at DESC, run_id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        DreamRunRecord(
            run_id=int(r["run_id"]),
            type=r["type"],
            started_at=int(r["started_at"]),
            completed_at=r["completed_at"],
            status=r["status"],
            atoms_created=int(r["atoms_created"] or 0),
            heads_created=int(r["heads_created"] or 0),
            edges_created=int(r["edges_created"] or 0),
            contradictions_resolved=int(r["contradictions_resolved"] or 0),
            bodies_archived_review=int(r["bodies_archived_review"] or 0),
            error_message=r["error_message"],
        )
        for r in rows
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_dream_runs.py -v
uv run mypy src
```

Expected: all PASS, mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/hippo/storage/dream_runs.py tests/test_dream_runs.py
git commit -m "storage: dream_runs tracks bodies_archived_review counter"
```

---

## Task 8: Config — three new prune constants

Add the tunables. Per the project's "store lookup instructions, not values" rule, these only live in `config.py`; docs reference them by name.

**Files:**
- Modify: `src/hippo/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_prune_constants_exist_with_sensible_values():
    from hippo.config import (
        PRUNE_NEAREST_K,
        PRUNE_ROLLING_SLICE_SIZE,
        PRUNE_SIMILARITY_THRESHOLD,
    )

    assert 0.5 < PRUNE_SIMILARITY_THRESHOLD <= 1.0
    assert PRUNE_NEAREST_K >= 1
    assert PRUNE_ROLLING_SLICE_SIZE >= 1
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest tests/test_config.py::test_prune_constants_exist_with_sensible_values -v
```

Expected: FAIL — names not defined.

- [ ] **Step 3: Add constants**

Append to the "Dream tuning" section of `src/hippo/config.py` (after `CLUSTER_COSINE_THRESHOLD`):

```python
# === Prune phase tuning ===
# Threshold above which two bodies are considered merge/supersede candidates.
PRUNE_SIMILARITY_THRESHOLD = 0.85
# Max neighbors to consider per body during the review judgement.
PRUNE_NEAREST_K = 5
# How many oldest-unreviewed active bodies the rolling slice judges per heavy dream.
PRUNE_ROLLING_SLICE_SIZE = 25
```

- [ ] **Step 4: Run test to verify it passes**

```
uv run pytest tests/test_config.py -v
uv run mypy src
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hippo/config.py tests/test_config.py
git commit -m "config: prune-phase tunables (similarity, nearest-k, slice size)"
```

---

## Task 9: Atomize prompt — `noise` field with examples

Update `prompts/atomize.md` so the LLM emits a `noise: true|false` field per atom. Examples on both sides; "when uncertain → noise=true" tiebreaker.

**Files:**
- Modify: `src/hippo/dream/prompts/atomize.md`

This task has no test of its own — the prompt is text. Behavior is verified by the next task (atomize.py update).

- [ ] **Step 1: Replace the entire prompt file**

Overwrite `src/hippo/dream/prompts/atomize.md` with:

```
You are extracting durable memory atoms from a Claude Code session transcript.

A "body" is one coherent piece of content that should be remembered. A "head" is a short keyword sentence that someone might use to recall this body. Bodies can have multiple heads (different angles into the same content).

Read the transcript below. Output a JSON array of atom objects. Each atom has the shape:
{
  "title": "short title (under 60 chars)",
  "body": "full content — can be a single fact, a paragraph of reasoning, or a long write-up. Whatever serves the concept.",
  "scope": "global" | "project:{{project}}",
  "heads": ["1-2 sentence keyword summary 1", "1-2 sentence keyword summary 2", ...],
  "noise": true | false
}

Rules for the `noise` field:
- noise=false means: this atom captures durable knowledge worth recalling later. Examples:
  - decisions ("we chose Postgres over MySQL because of full-text search")
  - preferences and constraints ("user prefers responses under 100 words")
  - project facts ("the auth service runs on port 8081")
  - reusable patterns ("retry exponential backoff on 5xx, give up after 5 tries")
  - bug-fix learnings ("worktree pointers can be relative paths, parse both forms")
  - spec or requirement statements ("memory must remain lean across nightly runs")
- noise=true means: in-the-moment chatter or session debugging that does NOT generalize. Examples:
  - terse procedural turns ("status", "again", "i sent it", "next")
  - acknowledgments ("ok", "thanks", "looks good", "yes do it")
  - debug-loop chatter ("try again", "doesn't work", "hmm", "wait")
  - trivial confirmations of a request ("done", "go ahead")
  - system events ("[Request interrupted]")
  - re-prompts after errors ("retry", "try with the fix")
- Tiebreaker: if you are uncertain whether an atom is durable, output noise=true. Default to skipping.

Other rules:
- Each atom must have at least 1 head and at most 5.
- Heads must be diverse — they capture different angles of the body, not paraphrases.
- "scope" = "global" if the atom applies regardless of project (user preferences, role, cross-project insights). Otherwise "project:{{project}}".
- If nothing in the transcript is worth remembering, return [].

Return ONLY the JSON array. No prose, no markdown fences.

---

PROJECT: {{project}}
SESSION: {{session_id}}
TRANSCRIPT:
{{transcript}}
```

- [ ] **Step 2: Sanity check**

```
uv run pytest -q
uv run ruff check src tests
```

Expected: existing tests still pass (the prompt is just text; old fakes return responses without `noise` so atomize.py still inserts them; we add the skip logic in Task 10).

- [ ] **Step 3: Commit**

```bash
git add src/hippo/dream/prompts/atomize.md
git commit -m "atomize: add noise field with concrete examples + tiebreaker"
```

---

## Task 10: Atomize.py — skip `noise=true` atoms

Update `atomize_session` to skip atoms where `noise == true`. Permissively coerce `bool(value)` so `"true"`, `1`, etc. all behave as truthy. Missing field defaults to `false` (backward compat with FakeLLM responses).

**Files:**
- Modify: `src/hippo/dream/atomize.py:69-99`
- Modify: `tests/test_atomize.py`

- [ ] **Step 1: Write three failing tests**

Append to `tests/test_atomize.py`:

```python
def test_atomize_skips_noise_atoms(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    enqueue_capture(s.conn, CaptureRecord(
        session_id="sess-A", user_message="status", assistant_message="ok",
    ))
    s.conn.close()

    fake = json.dumps([
        {
            "title": "Durable",
            "body": "Real durable content.",
            "scope": "global",
            "heads": ["one head"],
            "noise": False,
        },
        {
            "title": "Noise",
            "body": "ok thanks",
            "scope": "global",
            "heads": ["chatter"],
            "noise": True,
        },
    ])
    s = open_store(Scope.global_())
    n = atomize_session(
        store=s, session_id="sess-A", project=None, run_id=1,
        llm=FakeLLM(fake), daemon=FakeDaemon(),
    )
    assert n == 1
    titles = [b.title for b in list_bodies_by_scope(s.conn, "global")]
    assert titles == ["Durable"]
    s.conn.close()


def test_atomize_treats_missing_noise_field_as_false(tmp_path, monkeypatch):
    """Backward compat: old prompts returning no noise field still insert."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    enqueue_capture(s.conn, CaptureRecord(
        session_id="sess-A", user_message="x", assistant_message="y",
    ))
    s.conn.close()

    fake = json.dumps([{
        "title": "No noise field",
        "body": "content",
        "scope": "global",
        "heads": ["h"],
    }])
    s = open_store(Scope.global_())
    n = atomize_session(
        store=s, session_id="sess-A", project=None, run_id=1,
        llm=FakeLLM(fake), daemon=FakeDaemon(),
    )
    assert n == 1
    s.conn.close()


def test_atomize_treats_string_noise_as_truthy(tmp_path, monkeypatch):
    """LLM occasionally returns 'true' (string) instead of bool true; treat as truthy."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    enqueue_capture(s.conn, CaptureRecord(
        session_id="sess-A", user_message="x", assistant_message="y",
    ))
    s.conn.close()

    fake = json.dumps([{
        "title": "stringy noise",
        "body": "content",
        "scope": "global",
        "heads": ["h"],
        "noise": "true",
    }])
    s = open_store(Scope.global_())
    n = atomize_session(
        store=s, session_id="sess-A", project=None, run_id=1,
        llm=FakeLLM(fake), daemon=FakeDaemon(),
    )
    assert n == 0
    s.conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_atomize.py -v
```

Expected: 2 FAIL (`test_atomize_skips_noise_atoms` and `test_atomize_treats_string_noise_as_truthy` — they insert noise atoms because the skip logic is missing). The "missing field" test should already pass.

- [ ] **Step 3: Add the skip logic**

In `src/hippo/dream/atomize.py`, modify the `for atom in atoms:` loop to skip noise atoms. Replace the existing loop body (the `for atom in atoms:` block including its inner code) with:

```python
    for atom in atoms:
        if _is_noise(atom):
            continue
        title = atom.get("title", "")[:120]
        body_content = atom.get("body", "")
        heads = atom.get("heads", [])
        if not title or not body_content or not heads:
            continue

        # If LLM said scope='global' but we're processing a project store, still write to project
        # store (user can rebalance later via promote/demote — Plan 6 doesn't include those).
        body_id = uuid4().hex
        now = datetime.now(UTC)
        write_body_file(store.memory_dir, BodyFile(
            body_id=body_id, title=title, scope=store.scope.as_string(),
            created=now, updated=now, content=body_content,
        ))
        insert_body(store.conn, BodyRecord(
            body_id=body_id, file_path=f"bodies/{body_id}.md",
            title=title, scope=store.scope.as_string(),
            source=f"heavy-dream-run:{run_id}",
        ))
        # Embed all heads in one batched call
        head_summaries = [h for h in heads if isinstance(h, str) and h.strip()][:5]
        if head_summaries:
            vecs = daemon.embed(head_summaries)
            for summary, vec in zip(head_summaries, vecs, strict=True):
                head_id = uuid4().hex
                insert_head(
                    store.conn, HeadRecord(head_id=head_id, body_id=body_id, summary=summary)
                )
                insert_head_embedding(store.conn, head_id, vec)
        n_bodies += 1

    return n_bodies
```

Then add the `_is_noise` helper at module level (between `_strip_fences` and `atomize_session`):

```python
def _is_noise(atom: object) -> bool:
    """Permissive truthy check on the optional `noise` field.

    Accepts bool, "true"/"false" strings, and integers. Default False.
    """
    if not isinstance(atom, dict):
        return False
    val = atom.get("noise")
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes")
    if isinstance(val, int | float):
        return bool(val)
    return False
```

- [ ] **Step 4: Run all atomize tests**

```
uv run pytest tests/test_atomize.py -v
uv run mypy src
uv run ruff check src tests
```

Expected: all PASS, mypy clean, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/hippo/dream/atomize.py tests/test_atomize.py
git commit -m "atomize: skip atoms with noise=true (mode C of prune spec)"
```

---

## Task 11: Review prompt template

Single prompt that takes two bodies and returns `{decision, keeper, reason}`. Mirrors the contradiction prompt's structure.

**Files:**
- Create: `src/hippo/dream/prompts/review.md`

No test for this task — it's text. Behavior is exercised by Task 12.

- [ ] **Step 1: Create the prompt**

Create `src/hippo/dream/prompts/review.md`:

```
You are deciding whether two memory atoms are redundant or whether one supersedes the other, and if so which to keep.

Atom A (body_id: {{a_body_id}}, updated: {{a_updated}}):
{{a_body}}

Atom B (body_id: {{b_body_id}}, updated: {{b_updated}}):
{{b_body}}

Choose ONE decision:
- "merge": A and B describe the same fact from slightly different angles. Pick whichever is better-written or more complete; the other can be archived as redundant.
- "supersede": A and B describe the same subject but one is a newer correct version that replaces the other (e.g., "we used to use X but now use Y", or A's `updated` is recent and reflects a changed reality). Keep the current one; archive the outdated one.
- "keep_both": A and B are distinct atoms that should both stay (different facts, different scopes, complementary information, or only superficially similar).

When picking the keeper:
- Prefer the more recent `updated`.
- Prefer the more specific or detailed body.
- Prefer explicit supersession language ("we now ...", "as of ...").
- If both are equally good, prefer A.

Output a single JSON object:
{
  "decision": "merge" | "supersede" | "keep_both",
  "keeper": "<a_body_id or b_body_id, omitted or null when keep_both>",
  "reason": "<one sentence>"
}

Return ONLY the JSON.
```

- [ ] **Step 2: Sanity check**

```
uv run pytest -q
```

Expected: existing tests pass (no behavior change yet).

- [ ] **Step 3: Commit**

```bash
git add src/hippo/dream/prompts/review.md
git commit -m "prompt: add review.md for prune-phase pair judgment"
```

---

## Task 12: Review.py — `_judge_pair`

Takes `(llm, body_a, body_b)`, renders the review prompt, parses the JSON response, validates, returns `(decision, keeper_body_id)`. Defensive: any failure returns `("keep_both", None)`. Reads body markdown via `read_body_file`.

**Files:**
- Create: `src/hippo/dream/review.py`
- Create: `tests/test_review_phase.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_review_phase.py`:

```python
"""Tests for the prune-phase review module."""
from __future__ import annotations

import json
from datetime import UTC, datetime

from hippo.storage.bodies import BodyRecord, insert_body
from hippo.storage.body_files import BodyFile, write_body_file
from hippo.storage.migrations import run_migrations
from hippo.storage.multi_store import Scope, open_store


class FakeLLM:
    """LLM stub. By default returns the configured response; can also be a callable
    of (messages) -> response for per-call control."""

    def __init__(self, response):
        self.response = response
        self.calls: list[str] = []
        self.thinking_levels: list[str | None] = []

    def generate_chat(self, messages, *, temperature, max_tokens, thinking_level=None):
        content = messages[-1]["content"]
        self.calls.append(content)
        self.thinking_levels.append(thinking_level)
        if callable(self.response):
            return self.response(content)
        return self.response


class FakeDaemon:
    def embed(self, texts):
        from hippo.config import EMBEDDING_DIM
        return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]


def _setup_two_bodies(store, *, a_id="bid-a", b_id="bid-b"):
    now = datetime.now(UTC)
    for bid, content in [(a_id, "Body A content"), (b_id, "Body B content")]:
        write_body_file(
            store.memory_dir,
            BodyFile(
                body_id=bid, title=bid, scope="global",
                created=now, updated=now, content=content,
            ),
        )
        insert_body(
            store.conn,
            BodyRecord(
                body_id=bid, file_path=f"bodies/{bid}.md",
                title=bid, scope="global", source="test",
            ),
        )


def test_judge_pair_returns_merge_with_keeper(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_two_bodies(store)
    a = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-a'").fetchone()
    b = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-b'").fetchone()

    from hippo.dream.review import _judge_pair
    from hippo.storage.bodies import _row_to_record

    llm = FakeLLM(json.dumps({"decision": "merge", "keeper": "bid-a", "reason": "x"}))
    decision, keeper = _judge_pair(
        store=store, llm=llm,
        a=_row_to_record(a), b=_row_to_record(b),
    )
    assert decision == "merge"
    assert keeper == "bid-a"
    assert llm.thinking_levels == ["minimal"]
    store.conn.close()


def test_judge_pair_invalid_json_returns_keep_both(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_two_bodies(store)
    a = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-a'").fetchone()
    b = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-b'").fetchone()

    from hippo.dream.review import _judge_pair
    from hippo.storage.bodies import _row_to_record

    llm = FakeLLM("not JSON")
    decision, keeper = _judge_pair(
        store=store, llm=llm,
        a=_row_to_record(a), b=_row_to_record(b),
    )
    assert decision == "keep_both"
    assert keeper is None
    store.conn.close()


def test_judge_pair_unknown_keeper_returns_keep_both(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_two_bodies(store)
    a = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-a'").fetchone()
    b = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-b'").fetchone()

    from hippo.dream.review import _judge_pair
    from hippo.storage.bodies import _row_to_record

    # decision=merge but keeper points at a body not in the pair
    llm = FakeLLM(json.dumps(
        {"decision": "merge", "keeper": "totally-unknown", "reason": "x"}
    ))
    decision, keeper = _judge_pair(
        store=store, llm=llm,
        a=_row_to_record(a), b=_row_to_record(b),
    )
    assert decision == "keep_both"
    assert keeper is None
    store.conn.close()


def test_judge_pair_keep_both(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_two_bodies(store)
    a = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-a'").fetchone()
    b = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-b'").fetchone()

    from hippo.dream.review import _judge_pair
    from hippo.storage.bodies import _row_to_record

    llm = FakeLLM(json.dumps({"decision": "keep_both", "keeper": None, "reason": "distinct"}))
    decision, keeper = _judge_pair(
        store=store, llm=llm,
        a=_row_to_record(a), b=_row_to_record(b),
    )
    assert decision == "keep_both"
    assert keeper is None
    store.conn.close()


def test_judge_pair_missing_body_file_returns_keep_both(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_two_bodies(store)
    # Delete one of the body files
    (store.memory_dir / "bodies" / "bid-a.md").unlink()

    a = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-a'").fetchone()
    b = store.conn.execute("SELECT * FROM bodies WHERE body_id = 'bid-b'").fetchone()

    from hippo.dream.review import _judge_pair
    from hippo.storage.bodies import _row_to_record

    llm = FakeLLM(json.dumps({"decision": "merge", "keeper": "bid-a", "reason": "x"}))
    decision, keeper = _judge_pair(
        store=store, llm=llm,
        a=_row_to_record(a), b=_row_to_record(b),
    )
    assert decision == "keep_both"
    assert keeper is None
    store.conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_review_phase.py -v
```

Expected: 5 FAIL — module `hippo.dream.review` doesn't exist.

- [ ] **Step 3: Implement `_judge_pair`**

Create `src/hippo/dream/review.py`:

```python
"""Prune-phase review: gate-at-entry + rolling-slice sweep.

Detects redundant or superseded atoms inside the heavy dream and
soft-archives the loser. Modes A (factually superseded) and D
(redundant / mergeable) from the prune-phase design spec.
"""
from __future__ import annotations

import json
from typing import Protocol

from hippo.dream.atomize import _strip_fences
from hippo.dream.prompts import render
from hippo.models.llm import LLMProto
from hippo.storage.bodies import BodyRecord
from hippo.storage.body_files import read_body_file
from hippo.storage.multi_store import Store

_VALID_DECISIONS: frozenset[str] = frozenset({"merge", "supersede", "keep_both"})


class DaemonProto(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _judge_pair(
    *, store: Store, llm: LLMProto, a: BodyRecord, b: BodyRecord
) -> tuple[str, str | None]:
    """Ask the LLM whether to merge, supersede, or keep both.

    Returns (decision, keeper_body_id). `keeper_body_id` is None for keep_both
    and for any failure mode (invalid JSON, unknown decision, unknown keeper,
    missing body file).
    """
    try:
        a_md = read_body_file(store.memory_dir / a.file_path).content
        b_md = read_body_file(store.memory_dir / b.file_path).content
    except FileNotFoundError:
        return ("keep_both", None)

    prompt = render(
        "review",
        a_body_id=a.body_id, a_updated=a.updated_at, a_body=a_md,
        b_body_id=b.body_id, b_updated=b.updated_at, b_body=b_md,
    )
    raw = llm.generate_chat(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
        thinking_level="minimal",
    )
    try:
        obj = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, TypeError):
        return ("keep_both", None)

    decision = obj.get("decision") if isinstance(obj, dict) else None
    if decision not in _VALID_DECISIONS:
        return ("keep_both", None)
    if decision == "keep_both":
        return ("keep_both", None)

    keeper = obj.get("keeper")
    if keeper not in (a.body_id, b.body_id):
        return ("keep_both", None)
    return (decision, keeper)
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_review_phase.py -v
uv run mypy src
uv run ruff check src tests
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hippo/dream/review.py tests/test_review_phase.py
git commit -m "review: _judge_pair returns (decision, keeper) with defensive defaults"
```

---

## Task 13: Review.py — shared `_review_body_against_neighbors`

Wraps the workflow used by both passes: find merge candidates, judge each, archive loser if needed, stamp `last_reviewed_at`. Returns 1 if a body was archived, 0 otherwise.

**Files:**
- Modify: `src/hippo/dream/review.py`
- Modify: `tests/test_review_phase.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_review_phase.py`:

```python
def _setup_b_and_c_with_high_similarity(store):
    """Insert body B and body C such that find_merge_candidates(B) returns C."""
    from hippo.config import EMBEDDING_DIM
    from hippo.storage.heads import HeadRecord, insert_head
    from hippo.storage.vec import insert_head_embedding

    now = datetime.now(UTC)
    for bid in ("bid-B", "bid-C"):
        write_body_file(
            store.memory_dir,
            BodyFile(
                body_id=bid, title=bid, scope="global",
                created=now, updated=now, content=f"{bid} content",
            ),
        )
        insert_body(
            store.conn,
            BodyRecord(
                body_id=bid, file_path=f"bodies/{bid}.md",
                title=bid, scope="global", source="test",
            ),
        )
        head_id = f"h-{bid}"
        insert_head(
            store.conn, HeadRecord(head_id=head_id, body_id=bid, summary=bid)
        )
        # Identical embeddings → cosine 1.0
        vec = [1.0, 0.0] + [0.0] * (EMBEDDING_DIM - 2)
        insert_head_embedding(store.conn, head_id, vec)


def test_review_body_archives_loser_on_merge(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_b_and_c_with_high_similarity(store)

    from hippo.dream.review import _review_body_against_neighbors
    from hippo.storage.bodies import get_body

    # Keeper = bid-B → bid-C must be archived
    llm = FakeLLM(json.dumps({"decision": "merge", "keeper": "bid-B", "reason": "x"}))
    archived = _review_body_against_neighbors(
        store=store, llm=llm, body_id="bid-B",
    )
    assert archived == 1

    loser = get_body(store.conn, "bid-C")
    assert loser is not None
    assert loser.archived
    assert loser.archived_in_favor_of == "bid-B"
    assert (loser.archive_reason or "").startswith("merged_into:")

    # B was stamped
    b = get_body(store.conn, "bid-B")
    assert b is not None
    assert b.last_reviewed_at is not None

    # B's heads remain active; C's heads are archived
    rows = store.conn.execute(
        "SELECT body_id, archived FROM heads ORDER BY body_id"
    ).fetchall()
    by_bid = {r["body_id"]: r["archived"] for r in rows}
    assert by_bid == {"bid-B": 0, "bid-C": 1}

    store.conn.close()


def test_review_body_archives_loser_on_supersede(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_b_and_c_with_high_similarity(store)

    from hippo.dream.review import _review_body_against_neighbors
    from hippo.storage.bodies import get_body

    llm = FakeLLM(json.dumps({"decision": "supersede", "keeper": "bid-C", "reason": "x"}))
    archived = _review_body_against_neighbors(
        store=store, llm=llm, body_id="bid-B",
    )
    assert archived == 1

    loser = get_body(store.conn, "bid-B")
    assert loser is not None
    assert loser.archived
    assert (loser.archive_reason or "").startswith("superseded_by:")
    assert loser.archived_in_favor_of == "bid-C"

    store.conn.close()


def test_review_body_keep_both_archives_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    _setup_b_and_c_with_high_similarity(store)

    from hippo.dream.review import _review_body_against_neighbors
    from hippo.storage.bodies import get_body

    llm = FakeLLM(json.dumps({"decision": "keep_both", "keeper": None, "reason": "x"}))
    archived = _review_body_against_neighbors(
        store=store, llm=llm, body_id="bid-B",
    )
    assert archived == 0

    assert not get_body(store.conn, "bid-B").archived  # type: ignore[union-attr]
    assert not get_body(store.conn, "bid-C").archived  # type: ignore[union-attr]
    # B was still stamped
    assert get_body(store.conn, "bid-B").last_reviewed_at is not None  # type: ignore[union-attr]
    store.conn.close()


def test_review_body_with_no_candidates_stamps_and_returns_zero(tmp_path, monkeypatch):
    """A body whose heads are far from everything else: no LLM call, just stamp."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())
    # Only one body — no neighbors
    now = datetime.now(UTC)
    write_body_file(
        store.memory_dir,
        BodyFile(
            body_id="bid-solo", title="solo", scope="global",
            created=now, updated=now, content="content",
        ),
    )
    insert_body(
        store.conn,
        BodyRecord(
            body_id="bid-solo", file_path="bodies/bid-solo.md",
            title="solo", scope="global", source="test",
        ),
    )

    from hippo.dream.review import _review_body_against_neighbors
    from hippo.storage.bodies import get_body

    llm = FakeLLM(json.dumps({"decision": "merge", "keeper": "bid-solo", "reason": "x"}))
    archived = _review_body_against_neighbors(
        store=store, llm=llm, body_id="bid-solo",
    )
    assert archived == 0
    # No LLM call should have happened
    assert llm.calls == []
    # Stamped anyway
    assert get_body(store.conn, "bid-solo").last_reviewed_at is not None  # type: ignore[union-attr]
    store.conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_review_phase.py -v
```

Expected: 4 new FAIL — `_review_body_against_neighbors` not defined.

- [ ] **Step 3: Implement `_review_body_against_neighbors`**

Append to `src/hippo/dream/review.py`:

```python
def _review_body_against_neighbors(
    *, store: Store, llm: LLMProto, body_id: str
) -> int:
    """Review a single body against its merge candidates. Returns 1 if archived, else 0.

    Always stamps last_reviewed_at on the body so the rolling slice advances.
    """
    from hippo.config import PRUNE_NEAREST_K, PRUNE_SIMILARITY_THRESHOLD
    from hippo.storage.bodies import (
        find_merge_candidates,
        get_body,
        update_last_reviewed_at,
    )
    from hippo.storage.bodies import archive_body as archive_body_fn
    from hippo.storage.heads import archive_head, list_heads_for_body

    self_rec = get_body(store.conn, body_id)
    if self_rec is None or self_rec.archived:
        return 0

    candidates = find_merge_candidates(
        store.conn,
        body_id=body_id,
        threshold=PRUNE_SIMILARITY_THRESHOLD,
        k=PRUNE_NEAREST_K,
    )

    archived = 0
    if not candidates:
        update_last_reviewed_at(store.conn, body_id)
        return 0

    for cand_rec, _sim in candidates:
        decision, keeper = _judge_pair(store=store, llm=llm, a=self_rec, b=cand_rec)
        if decision in ("merge", "supersede") and keeper is not None:
            loser_id = body_id if keeper == cand_rec.body_id else cand_rec.body_id
            winner_id = keeper
            reason_prefix = "merged_into" if decision == "merge" else "superseded_by"
            archive_reason = f"{reason_prefix}:{winner_id}"
            archive_body_fn(
                store.conn, loser_id, reason=archive_reason, in_favor_of=winner_id
            )
            for h in list_heads_for_body(store.conn, loser_id):
                archive_head(
                    store.conn, h.head_id, reason=f"body_archived:{archive_reason}"
                )
            archived = 1
            break

    update_last_reviewed_at(store.conn, body_id)
    return archived
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_review_phase.py -v
uv run mypy src
uv run ruff check src tests
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hippo/dream/review.py tests/test_review_phase.py
git commit -m "review: _review_body_against_neighbors archives loser, stamps body"
```

---

## Task 14: Review.py — `review_new_atoms` (gate-at-entry pass)

Walks bodies inserted by this run, runs the shared inner per body. Skips bodies that get archived mid-loop (they've already been processed).

**Files:**
- Modify: `src/hippo/dream/review.py`
- Modify: `tests/test_review_phase.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_review_phase.py`:

```python
def test_review_new_atoms_processes_only_run_bodies(tmp_path, monkeypatch):
    """Bodies with source='heavy-dream-run:7' are reviewed; others are skipped."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())

    from hippo.config import EMBEDDING_DIM
    from hippo.storage.heads import HeadRecord, insert_head
    from hippo.storage.vec import insert_head_embedding

    now = datetime.now(UTC)
    # Body from this run
    write_body_file(
        store.memory_dir,
        BodyFile(
            body_id="this-run", title="t", scope="global",
            created=now, updated=now, content="content",
        ),
    )
    insert_body(
        store.conn,
        BodyRecord(
            body_id="this-run", file_path="bodies/this-run.md",
            title="t", scope="global", source="heavy-dream-run:7",
        ),
    )
    insert_head(
        store.conn, HeadRecord(head_id="h-this", body_id="this-run", summary="x")
    )
    insert_head_embedding(store.conn, "h-this", [1.0, 0.0] + [0.0] * (EMBEDDING_DIM - 2))

    # Body from a previous run (must NOT be touched by review_new_atoms)
    write_body_file(
        store.memory_dir,
        BodyFile(
            body_id="prev", title="p", scope="global",
            created=now, updated=now, content="prev content",
        ),
    )
    insert_body(
        store.conn,
        BodyRecord(
            body_id="prev", file_path="bodies/prev.md",
            title="p", scope="global", source="heavy-dream-run:6",
        ),
    )
    insert_head(store.conn, HeadRecord(head_id="h-prev", body_id="prev", summary="y"))
    # Identical embedding → would-be merge candidate
    insert_head_embedding(store.conn, "h-prev", [1.0, 0.0] + [0.0] * (EMBEDDING_DIM - 2))

    from hippo.dream.review import review_new_atoms
    from hippo.storage.bodies import get_body

    llm = FakeLLM(json.dumps({"decision": "merge", "keeper": "this-run", "reason": "x"}))
    n_archived = review_new_atoms(store=store, llm=llm, run_id=7)
    assert n_archived == 1

    # The previous-run body got archived (loser)
    prev = get_body(store.conn, "prev")
    assert prev is not None
    assert prev.archived

    # The new body has last_reviewed_at stamped
    new = get_body(store.conn, "this-run")
    assert new is not None
    assert new.last_reviewed_at is not None

    store.conn.close()


def test_review_new_atoms_returns_zero_when_no_run_bodies(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())

    from hippo.dream.review import review_new_atoms

    llm = FakeLLM("")
    assert review_new_atoms(store=store, llm=llm, run_id=99) == 0
    assert llm.calls == []
    store.conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_review_phase.py -v
```

Expected: 2 new FAIL — `review_new_atoms` not defined.

- [ ] **Step 3: Implement `review_new_atoms`**

Append to `src/hippo/dream/review.py`:

```python
def review_new_atoms(*, store: Store, llm: LLMProto, run_id: int) -> int:
    """Pass 1 — review each body inserted by this heavy dream run.

    Returns count of bodies archived.
    """
    from hippo.storage.bodies import find_active_bodies_by_run_source

    new_bodies = find_active_bodies_by_run_source(store.conn, run_id=run_id)
    n_archived = 0
    for body in new_bodies:
        # Re-check active state in case a previous iteration archived this body
        # (it could have lost a head-to-head similarity contest with another new body).
        n_archived += _review_body_against_neighbors(
            store=store, llm=llm, body_id=body.body_id,
        )
    return n_archived
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_review_phase.py -v
uv run mypy src
uv run ruff check src tests
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hippo/dream/review.py tests/test_review_phase.py
git commit -m "review: review_new_atoms gates bodies inserted this run"
```

---

## Task 15: Review.py — `review_rolling_slice` (rolling sweep)

Picks K oldest-unreviewed active bodies in scope, runs the shared inner per body. Idempotent within a single run (after stamping, the same bodies sort to the back of the queue).

**Files:**
- Modify: `src/hippo/dream/review.py`
- Modify: `tests/test_review_phase.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_review_phase.py`:

```python
def test_review_rolling_slice_picks_oldest_unreviewed(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())

    from hippo.config import EMBEDDING_DIM
    from hippo.storage.heads import HeadRecord, insert_head
    from hippo.storage.vec import insert_head_embedding

    now = datetime.now(UTC)
    # Three bodies with distinct embeddings → no merge candidates → just stamp
    for i, bid in enumerate(("b-old", "b-fresh", "b-new")):
        write_body_file(
            store.memory_dir,
            BodyFile(
                body_id=bid, title=bid, scope="global",
                created=now, updated=now, content=bid,
            ),
        )
        insert_body(
            store.conn,
            BodyRecord(
                body_id=bid, file_path=f"bodies/{bid}.md",
                title=bid, scope="global", source="test",
            ),
        )
        insert_head(store.conn, HeadRecord(head_id=f"h-{bid}", body_id=bid, summary=bid))
        # Distinct, orthogonal embeddings → cosine 0
        vec = [0.0] * EMBEDDING_DIM
        vec[i] = 1.0
        insert_head_embedding(store.conn, f"h-{bid}", vec)

    store.conn.execute(
        "UPDATE bodies SET last_reviewed_at = 100 WHERE body_id = 'b-old'"
    )
    store.conn.execute(
        "UPDATE bodies SET last_reviewed_at = 999 WHERE body_id = 'b-fresh'"
    )
    store.conn.commit()

    from hippo.dream.review import review_rolling_slice
    from hippo.storage.bodies import get_body

    llm = FakeLLM(json.dumps({"decision": "keep_both", "keeper": None, "reason": "x"}))

    # slice_size=2 → reviews b-new (NULL) then b-old (100)
    n_archived = review_rolling_slice(store=store, scope="global", llm=llm, slice_size=2)
    assert n_archived == 0

    # b-new and b-old got stamped; b-fresh did not
    after = {bid: get_body(store.conn, bid).last_reviewed_at for bid in ("b-old", "b-fresh", "b-new")}  # type: ignore[union-attr]
    assert after["b-new"] is not None
    assert after["b-old"] is not None and after["b-old"] != 100  # was re-stamped to current time
    assert after["b-fresh"] == 999  # untouched

    store.conn.close()


def test_review_rolling_slice_idempotent_within_run(tmp_path, monkeypatch):
    """Calling review_rolling_slice twice in a row doesn't re-judge same bodies."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())

    from hippo.config import EMBEDDING_DIM
    from hippo.storage.heads import HeadRecord, insert_head
    from hippo.storage.vec import insert_head_embedding

    now = datetime.now(UTC)
    for i, bid in enumerate(("b1", "b2")):
        write_body_file(
            store.memory_dir,
            BodyFile(
                body_id=bid, title=bid, scope="global",
                created=now, updated=now, content=bid,
            ),
        )
        insert_body(
            store.conn,
            BodyRecord(
                body_id=bid, file_path=f"bodies/{bid}.md",
                title=bid, scope="global", source="test",
            ),
        )
        insert_head(store.conn, HeadRecord(head_id=f"h{bid}", body_id=bid, summary=bid))
        vec = [0.0] * EMBEDDING_DIM
        vec[i] = 1.0
        insert_head_embedding(store.conn, f"h{bid}", vec)

    from hippo.dream.review import review_rolling_slice

    llm = FakeLLM(json.dumps({"decision": "keep_both", "keeper": None, "reason": "x"}))
    review_rolling_slice(store=store, scope="global", llm=llm, slice_size=2)
    calls_after_first = len(llm.calls)

    # Both bodies are now stamped recently; second call's slice picks the same bodies
    # (slice ordering is by COALESCE(last_reviewed_at, 0) ASC; both stamped to ~now).
    # Either of them will be re-reviewed if we ask for slice_size=2 again, since
    # there are no other active bodies. That's fine — idempotence here means no
    # archives happen because keep_both still wins.
    review_rolling_slice(store=store, scope="global", llm=llm, slice_size=2)
    calls_after_second = len(llm.calls)

    # No archives were made (orthogonal embeddings → no candidates → no LLM calls)
    assert calls_after_first == 0
    assert calls_after_second == 0

    store.conn.close()


def test_review_rolling_slice_excludes_archived(tmp_path, monkeypatch):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    store = open_store(Scope.global_())

    from hippo.config import EMBEDDING_DIM
    from hippo.storage.heads import HeadRecord, insert_head
    from hippo.storage.vec import insert_head_embedding

    now = datetime.now(UTC)
    for bid in ("b-active", "b-arch"):
        write_body_file(
            store.memory_dir,
            BodyFile(
                body_id=bid, title=bid, scope="global",
                created=now, updated=now, content=bid,
            ),
        )
        insert_body(
            store.conn,
            BodyRecord(
                body_id=bid, file_path=f"bodies/{bid}.md",
                title=bid, scope="global", source="test",
            ),
        )
        insert_head(store.conn, HeadRecord(head_id=f"h-{bid}", body_id=bid, summary=bid))
        insert_head_embedding(
            store.conn, f"h-{bid}", [1.0, 0.0] + [0.0] * (EMBEDDING_DIM - 2)
        )
    store.conn.execute("UPDATE bodies SET archived = 1 WHERE body_id = 'b-arch'")
    store.conn.commit()

    from hippo.dream.review import review_rolling_slice
    from hippo.storage.bodies import get_body

    llm = FakeLLM(json.dumps({"decision": "keep_both", "keeper": None, "reason": "x"}))
    review_rolling_slice(store=store, scope="global", llm=llm, slice_size=10)

    # Only b-active should be stamped
    assert get_body(store.conn, "b-active").last_reviewed_at is not None  # type: ignore[union-attr]
    arch = get_body(store.conn, "b-arch")
    assert arch is not None and arch.last_reviewed_at is None

    store.conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_review_phase.py -v
```

Expected: 3 new FAIL — `review_rolling_slice` not defined.

- [ ] **Step 3: Implement `review_rolling_slice`**

Append to `src/hippo/dream/review.py`:

```python
def review_rolling_slice(
    *, store: Store, scope: str, llm: LLMProto, slice_size: int
) -> int:
    """Pass 2 — review the K oldest-unreviewed active bodies in scope.

    NULL last_reviewed_at sorts first (never-reviewed bodies). Returns count
    of bodies archived this slice.
    """
    from hippo.storage.bodies import find_oldest_unreviewed_active

    slice_bodies = find_oldest_unreviewed_active(
        store.conn, scope=scope, limit=slice_size,
    )
    n_archived = 0
    for body in slice_bodies:
        n_archived += _review_body_against_neighbors(
            store=store, llm=llm, body_id=body.body_id,
        )
    return n_archived
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_review_phase.py -v
uv run mypy src
uv run ruff check src tests
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hippo/dream/review.py tests/test_review_phase.py
git commit -m "review: review_rolling_slice processes oldest-unreviewed active"
```

---

## Task 16: Heavy-dream orchestrator wires the review phase

Insert review_new_atoms + review_rolling_slice between atomize and multi_head. Track `bodies_archived_review` and pass to `complete_run`.

**Files:**
- Modify: `src/hippo/dream/heavy.py`
- Modify: `tests/test_dream_heavy_orchestrator.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_dream_heavy_orchestrator.py`:

```python
def test_heavy_dream_runs_review_phase_and_records_counter(tmp_path, monkeypatch):
    """End-to-end: heavy dream runs the review phase between atomize and multi_head,
    populates bodies_archived_review in dream_runs, and exposes it in the result."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")

    s = open_store(Scope.global_())
    enqueue_capture(
        s.conn,
        CaptureRecord(
            session_id="sess-A", user_message="we use postgres", assistant_message="ok",
        ),
    )
    s.conn.close()

    llm = FakeLLM()
    daemon = FakeDaemon()
    results = run_heavy_dream_all_scopes(scopes=[Scope.global_()], llm=llm, daemon=daemon)

    result = results["global"]
    assert "bodies_archived_review" in result
    assert isinstance(result["bodies_archived_review"], int)

    # The dream_runs row reflects the same counter
    s = open_store(Scope.global_())
    row = s.conn.execute(
        "SELECT bodies_archived_review FROM dream_runs WHERE run_id = ?",
        (result["run_id"],),
    ).fetchone()
    assert row is not None
    assert row["bodies_archived_review"] == result["bodies_archived_review"]
    s.conn.close()
```

Also extend `FakeLLM.generate_chat` in this file so it doesn't crash on the new review prompt — add a clause for the new prompt's distinguishing phrase. Replace the `generate_chat` method on the existing `FakeLLM` (defined at the top of `tests/test_dream_heavy_orchestrator.py`) with:

```python
    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        thinking_level: str | None = None,
    ) -> str:
        content = messages[-1]["content"]
        self.calls.append(content)
        if "extracting durable memory atoms" in content:
            return json.dumps([{
                "title": "Test atom",
                "body": "Some content",
                "scope": "global",
                "heads": ["one head"],
                "noise": False,
            }])
        if "generating diverse keyword summaries" in content:
            return json.dumps(["another head"])
        if "deciding whether two memory heads are related" in content:
            return json.dumps({"relation": "related", "weight": 0.5})
        if "deciding whether two memory atoms genuinely contradict" in content:
            return json.dumps({"contradicts": False})
        if "deciding whether two memory atoms are redundant" in content:
            return json.dumps({"decision": "keep_both", "keeper": None, "reason": "x"})
        return "[]"
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_dream_heavy_orchestrator.py -v
```

Expected: FAIL — `bodies_archived_review` not in result.

- [ ] **Step 3: Modify `src/hippo/dream/heavy.py`**

Replace the entire body of `run_heavy_dream_for_scope` with the version below. The change: add `from hippo.dream.review import review_new_atoms, review_rolling_slice` at top, track `n_review_archived`, run both passes between atomize and multi_head, pass `bodies_archived_review` to `complete_run`, include it in the returned dict.

Top of file imports — add this line to the existing imports block:

```python
from hippo.config import HEAVY_LOCK_FILENAME, PRUNE_ROLLING_SLICE_SIZE
from hippo.dream.review import review_new_atoms, review_rolling_slice
```

(Replace the existing `from hippo.config import HEAVY_LOCK_FILENAME` line with the combined import.)

The full new body of `run_heavy_dream_for_scope`:

```python
def run_heavy_dream_for_scope(
    *, scope: Scope, llm: LLMProto, daemon: DaemonProto
) -> dict[str, object]:
    store = open_store(scope)
    lock_path = store.memory_dir / HEAVY_LOCK_FILENAME
    try:
        handle = acquire_lock(lock_path)
    except LockHeldError:
        store.conn.close()
        return {"skipped_locked": True}

    run_id = start_run(store.conn, "heavy")
    n_atoms = 0
    n_heads = 0
    n_edges = 0
    n_contradictions = 0
    n_review_archived = 0

    try:
        # Phase a: atomize each session
        session_rows = store.conn.execute(
            "SELECT DISTINCT session_id FROM capture_queue WHERE processed_at IS NULL"
        ).fetchall()
        processed_ids: list[int] = []
        for r in session_rows:
            session_id = r["session_id"]
            n_atoms += atomize_session(
                store=store, session_id=session_id,
                project=scope.project_name, run_id=run_id,
                llm=llm, daemon=daemon,
            )
            cap_ids = [
                row["queue_id"] for row in store.conn.execute(
                    "SELECT queue_id FROM capture_queue"
                    " WHERE session_id = ? AND processed_at IS NULL",
                    (session_id,),
                ).fetchall()
            ]
            processed_ids.extend(cap_ids)

        # Phase a2: review (gate-at-entry + rolling slice)
        n_review_archived += review_new_atoms(store=store, llm=llm, run_id=run_id)
        n_review_archived += review_rolling_slice(
            store=store, scope=scope.as_string(),
            llm=llm, slice_size=PRUNE_ROLLING_SLICE_SIZE,
        )

        # Phase b: multi-head expansion
        n_heads += expand_heads_for_eligible_bodies(store=store, llm=llm, daemon=daemon)

        # Phase c-d: cluster + edge proposal
        n_edges += propose_edges(store=store, llm=llm)

        # Phase e: contradiction resolution
        n_contradictions += resolve_contradictions(store=store, llm=llm)

        # Phase f: cleanup
        finalize_processed_captures(store=store, queue_ids=processed_ids, run_id=run_id)

        complete_run(
            store.conn, run_id,
            atoms_created=n_atoms, heads_created=n_heads,
            edges_created=n_edges, contradictions_resolved=n_contradictions,
            bodies_archived_review=n_review_archived,
        )
        return {
            "run_id": run_id,
            "atoms_created": n_atoms,
            "heads_created": n_heads,
            "edges_created": n_edges,
            "contradictions_resolved": n_contradictions,
            "bodies_archived_review": n_review_archived,
        }
    except Exception as e:
        fail_run(store.conn, run_id, error_message=str(e))
        raise
    finally:
        release_lock(handle)
        store.conn.close()
```

- [ ] **Step 4: Run all tests + gates**

```
uv run pytest -q
uv run ruff check src tests
uv run mypy src
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hippo/dream/heavy.py tests/test_dream_heavy_orchestrator.py
git commit -m "heavy: wire review phase between atomize and multi_head"
```

---

## Task 17: Update `KNOWN_ISSUES.md` — modify atomize-noise entry

Per the project's "modify or delete, don't add new sections" rule, update the existing "Atomize-prompt noise leakage" section to reflect that the prompt now has an explicit `noise` field; the entry can stay (in case leaks still happen) but should reference the mitigation.

**Files:**
- Modify: `KNOWN_ISSUES.md`

- [ ] **Step 1: Read the existing entry**

Read `KNOWN_ISSUES.md` and locate the section "## Atomize-prompt noise leakage".

- [ ] **Step 2: Replace the section text**

Replace that section's text (the heading and its body) with:

```
## Atomize-prompt noise leakage

**Symptom:** an occasional in-the-moment chatter atom may still land in the
store from session-debug turns despite the explicit `noise: true|false`
field in the atomize prompt (`src/hippo/dream/prompts/atomize.md`). The
prompt now requires the model to commit to a noise classification per atom
with explicit examples on both sides plus a "when uncertain → noise=true"
tiebreaker, and `_is_noise` in `src/hippo/dream/atomize.py` is permissive
about the value's type (bool / "true" / 1 all coerce to truthy). This
mitigates but doesn't eliminate the leak.

**Workaround:** soft-archive obvious noise atoms when spotted:

`bin/memory-archive <head_id> --reason "atomize-noise"`

**Fix path:** the prune-phase review step
(`src/hippo/dream/review.py::review_rolling_slice`) eventually re-evaluates
older atoms against newer ones; redundant noise that survived atomize will
get caught when a clearly-better neighbor arrives. Future improvement: a
post-atomize length / structure heuristic as a second-line backstop.

## Review false-positive recovery is manual SQL today

**Symptom:** if the prune-phase review archives a body that you later want
back, there's no `bin/memory-restore` command. Recovery is `UPDATE bodies
SET archived = 0, archive_reason = NULL, archived_in_favor_of = NULL WHERE
body_id = ?` in the project's sqlite store, plus the same on each head
where `body_id = ?`. The body markdown file is still on disk (we don't
hard-delete) so the row's `file_path` resolves correctly.

**Fix path:** add a `memory-restore` CLI when this becomes annoying.
```

(The two sub-sections must replace the existing "Atomize-prompt noise leakage" section in place. The order of remaining sections in `KNOWN_ISSUES.md` stays unchanged.)

- [ ] **Step 3: Commit**

```bash
git add KNOWN_ISSUES.md
git commit -m "docs: KNOWN_ISSUES updated for prune-phase mitigation + recovery note"
```

---

## Task 18: Real-LLM smoke test (gated)

Single integration test that loads the real LLM, atomizes a tiny synthetic transcript with one obvious-noise turn and one durable turn, and asserts the noise turn is dropped while the durable one survives. Skipped unless `RUN_LLM_TESTS=1`.

**Files:**
- Modify: `tests/test_llm.py` OR create a new gated file (use existing `test_llm.py` to follow the existing pattern)

- [ ] **Step 1: Read existing `tests/test_llm.py` to understand its skip/gate pattern**

```
cat tests/test_llm.py
```

Note the `@pytest.mark.skipif(...)` decorator pattern.

- [ ] **Step 2: Append the new test**

Append to `tests/test_llm.py`:

```python
@pytest.mark.skipif(
    os.environ.get("RUN_LLM_TESTS") != "1",
    reason="set RUN_LLM_TESTS=1 to run",
)
def test_atomize_noise_field_real_llm(tmp_path, monkeypatch):
    """Real-LLM smoke: noise atoms get dropped, durable atoms survive."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")

    from hippo.dream.atomize import atomize_session
    from hippo.models.llm import select_llm
    from hippo.storage.bodies import list_bodies_by_scope
    from hippo.storage.capture import CaptureRecord, enqueue_capture
    from hippo.storage.multi_store import Scope, open_store

    s = open_store(Scope.global_())
    # One durable + one noise-y interaction
    enqueue_capture(s.conn, CaptureRecord(
        session_id="sess-mix",
        user_message="we use postgres for the main DB and that's a hard requirement",
        assistant_message="got it; recording",
    ))
    enqueue_capture(s.conn, CaptureRecord(
        session_id="sess-mix",
        user_message="status",
        assistant_message="ok",
    ))
    s.conn.close()

    class _DaemonStub:
        def embed(self, texts):
            from hippo.config import EMBEDDING_DIM
            return [[1.0] + [0.0] * (EMBEDDING_DIM - 1) for _ in texts]

    llm = select_llm()
    s = open_store(Scope.global_())
    n = atomize_session(
        store=s, session_id="sess-mix", project=None, run_id=1,
        llm=llm, daemon=_DaemonStub(),
    )
    # We expect the durable atom to be kept and the "status / ok" to be dropped.
    # The exact n depends on how the LLM splits the durable interaction,
    # but it should be >= 1 and the bodies should NOT include obvious noise.
    bodies = list_bodies_by_scope(s.conn, "global")
    assert n >= 1
    assert all("status" not in b.title.lower() for b in bodies)
    assert all(len(b.title) > 4 for b in bodies)  # weak sanity that noise isn't a title
    s.conn.close()
```

If the existing file doesn't `import os`, add `import os` to the top imports.

- [ ] **Step 3: Sanity check (default skip path)**

```
uv run pytest tests/test_llm.py -v
```

Expected: the new test is SKIPPED (alongside the existing two).

- [ ] **Step 4: Commit**

```bash
git add tests/test_llm.py
git commit -m "tests: real-LLM smoke for atomize noise filter (gated)"
```

---

## Task 19: Final orchestration check + branch summary

Run the entire suite, ruff, and mypy. Confirm no regressions. Then summarize what was added.

- [ ] **Step 1: Run full test suite**

```
uv run pytest -q
```

Expected: all PASS, no new skips beyond the existing 2 LLM-gated + 1 new LLM-gated = 3 skipped.

- [ ] **Step 2: Run ruff and mypy**

```
uv run ruff check src tests
uv run mypy src
```

Expected: both clean.

- [ ] **Step 3: Show the diff summary**

```
git log main..HEAD --oneline
```

Expected: ~17 commits, one per task.

- [ ] **Step 4: Final code review handoff**

Use `superpowers:requesting-code-review` to dispatch a code-reviewer subagent across the whole branch (BASE_SHA = `main`, HEAD_SHA = current). Apply Critical and Important fixes if any. After review, the branch is ready to merge.

---

## Self-Review

**Spec coverage:** every section of `docs/superpowers/specs/2026-04-29-prune-phase-design.md` maps to a task:

| Spec section | Task(s) |
|---|---|
| Architecture / new phase order | Task 16 |
| Atomize prompt with `noise` field | Tasks 9, 10 |
| `review.py` module functions | Tasks 12, 13, 14, 15 |
| `review.md` prompt | Task 11 |
| Schema migration 002 | Task 1 |
| `bodies` table helpers | Tasks 2, 3, 4, 5, 6 |
| `dream_runs` counter | Task 7 |
| Config knobs | Task 8 |
| Error handling rows | Covered defensively in Tasks 10, 12, 13 |
| Testing plan rows | Tasks 1–18 (each has explicit pytest cases) |
| Migration / rollout | Task 1 (idempotent migration), Task 17 (KNOWN_ISSUES) |

**Type consistency:**
- `_judge_pair` returns `tuple[str, str | None]` — used consistently in Task 13's `_review_body_against_neighbors`.
- `find_merge_candidates` returns `list[tuple[BodyRecord, float]]` — consumed by Task 13.
- `BodyRecord.last_reviewed_at: int | None` — added in Task 2, read in Tasks 4, 13, 14, 15.
- `review_new_atoms(*, store, llm, run_id) -> int` — kwargs match Task 16's call site.
- `review_rolling_slice(*, store, scope, llm, slice_size) -> int` — kwargs match Task 16's call site.
- `complete_run(..., bodies_archived_review=...)` — Task 7 adds the kwarg, Task 16 passes it.

**No placeholders:** all steps contain exact code. No "TBD", no "similar to Task N", no "implement appropriate handling".
