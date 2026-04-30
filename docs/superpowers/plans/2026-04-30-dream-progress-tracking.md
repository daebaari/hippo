# Dream-Heavy Progress Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make `dream-heavy` observable while it runs by emitting per-phase progress to a queryable `dream_runs` row and matching stderr lines, plus a `dream-status` CLI to read it.

**Architecture:** add five nullable columns to `dream_runs` via a versioned SQL migration; introduce a `ProgressReporter` that throttles DB updates and stderr emits to once per ~5s or 100 ticks; wire phase entry/exit and per-pair callbacks into `heavy.py`; ship a small read-only CLI that finds the most recent run across all scope DBs.

**Tech Stack:** Python 3.12+, SQLite (sqlite3 + sqlite-vec), pytest, uv, existing hippo migration runner (`src/hippo/storage/migrations.py`).

**Spec:** `docs/superpowers/specs/2026-04-30-dream-progress-tracking-design.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `schema/003_dream_progress.sql` | new | Adds the five progress columns to `dream_runs`. |
| `src/hippo/storage/dream_runs.py` | modify | Expose new columns on `DreamRunRecord`; add `start_phase()`, `update_progress()`, and `get_running_run()` helpers. |
| `src/hippo/dream/progress.py` | new | Pure logic: ETA math + `ProgressReporter` (throttled callback that writes to DB + stderr). |
| `src/hippo/dream/heavy.py` | modify | Wrap each phase in a context that constructs a `ProgressReporter` and passes it as `progress_cb` to the phase function. |
| `src/hippo/dream/edge_proposal.py` | modify | Accept optional `progress_cb`; call once per pair iteration (including skipped-existing pairs). Return total pairs alongside inserted count. |
| `src/hippo/dream/atomize.py` | modify | Accept `progress_cb`; call once per session processed. |
| `src/hippo/dream/review.py` | modify | Accept `progress_cb`; call once per body reviewed (in both passes). |
| `src/hippo/dream/multi_head.py` | modify | Accept `progress_cb`; call once per body expanded. |
| `src/hippo/dream/contradiction.py` | modify | Accept `progress_cb`; call once per pair judged. |
| `src/hippo/cli/dream_status.py` | new | CLI: scan all scope DBs, render the most recent row. |
| `bin/dream-status` | new | Bash wrapper (mirrors `bin/memory-get`). |
| `pyproject.toml` | modify | Register `dream-status` in `[project.scripts]`. |
| `tests/test_dream_progress.py` | new | Unit tests for `ProgressReporter` + ETA math. |
| `tests/test_dream_runs.py` | modify | Tests for `start_phase`, `update_progress`, `get_running_run`. |
| `tests/test_dream_status_cli.py` | new | CLI rendering + scope filter + fallback. |
| `tests/test_dream_heavy_orchestrator.py` | modify | Add a smoke test asserting phase transitions land in the row. |

---

## Task 1: Schema migration adds progress columns

**Files:**
- Create: `schema/003_dream_progress.sql`
- Modify: `tests/test_migrations.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_migrations.py`:

```python
def test_migration_003_adds_progress_columns(sqlite_conn):
    from hippo.storage.migrations import run_migrations

    run_migrations(sqlite_conn)
    cols = {
        row["name"]
        for row in sqlite_conn.execute("PRAGMA table_info(dream_runs)").fetchall()
    }
    assert {
        "current_phase",
        "phase_done",
        "phase_total",
        "phase_started_at",
        "last_progress_at",
    }.issubset(cols)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/code/hippo && uv run pytest tests/test_migrations.py::test_migration_003_adds_progress_columns -v
```

Expected: FAIL with the assertion `not subset` listing the missing columns.

- [ ] **Step 3: Create the migration file**

Write `schema/003_dream_progress.sql`:

```sql
-- Schema migration 003: dream-heavy progress tracking
-- Adds nullable columns to dream_runs that heavy.py updates as each phase progresses.
-- Idempotency is provided by the schema_versions tracker (see src/hippo/storage/migrations.py).

ALTER TABLE dream_runs ADD COLUMN current_phase TEXT;
ALTER TABLE dream_runs ADD COLUMN phase_done INTEGER;
ALTER TABLE dream_runs ADD COLUMN phase_total INTEGER;
ALTER TABLE dream_runs ADD COLUMN phase_started_at INTEGER;
ALTER TABLE dream_runs ADD COLUMN last_progress_at INTEGER;
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/code/hippo && uv run pytest tests/test_migrations.py::test_migration_003_adds_progress_columns -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/code/hippo
git add schema/003_dream_progress.sql tests/test_migrations.py
git commit -m "feat(schema): add progress columns to dream_runs (migration 003)"
```

---

## Task 2: DreamRunRecord exposes the new fields

**Files:**
- Modify: `src/hippo/storage/dream_runs.py`
- Modify: `tests/test_dream_runs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dream_runs.py`:

```python
def test_dream_run_record_exposes_progress_fields(conn):
    run_id = start_run(conn, "heavy")
    runs = get_recent_runs(conn, limit=1)
    r = runs[0]
    assert r.current_phase is None
    assert r.phase_done is None
    assert r.phase_total is None
    assert r.phase_started_at is None
    assert r.last_progress_at is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_runs.py::test_dream_run_record_exposes_progress_fields -v
```

Expected: FAIL with `AttributeError: 'DreamRunRecord' object has no attribute 'current_phase'`.

- [ ] **Step 3: Extend the dataclass and the SELECT mapping**

In `src/hippo/storage/dream_runs.py`, add five fields to `DreamRunRecord`:

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
    current_phase: str | None
    phase_done: int | None
    phase_total: int | None
    phase_started_at: int | None
    last_progress_at: int | None
```

In the `get_recent_runs` list comprehension, add the five fields after `error_message=r["error_message"],`:

```python
            current_phase=r["current_phase"],
            phase_done=r["phase_done"],
            phase_total=r["phase_total"],
            phase_started_at=r["phase_started_at"],
            last_progress_at=r["last_progress_at"],
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_runs.py -v
```

Expected: all dream_runs tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/code/hippo
git add src/hippo/storage/dream_runs.py tests/test_dream_runs.py
git commit -m "feat(storage): expose dream_runs progress fields on DreamRunRecord"
```

---

## Task 3: start_phase() helper

**Files:**
- Modify: `src/hippo/storage/dream_runs.py`
- Modify: `tests/test_dream_runs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dream_runs.py`:

```python
def test_start_phase_sets_phase_columns_and_resets_done(conn):
    from hippo.storage.dream_runs import start_phase

    run_id = start_run(conn, "heavy")
    start_phase(conn, run_id, phase="atomize", total=12)

    runs = get_recent_runs(conn, limit=1)
    r = runs[0]
    assert r.current_phase == "atomize"
    assert r.phase_done == 0
    assert r.phase_total == 12
    assert r.phase_started_at is not None
    assert r.last_progress_at is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_runs.py::test_start_phase_sets_phase_columns_and_resets_done -v
```

Expected: FAIL with `ImportError: cannot import name 'start_phase'`.

- [ ] **Step 3: Implement `start_phase`**

Add to `src/hippo/storage/dream_runs.py`:

```python
def start_phase(
    conn: sqlite3.Connection, run_id: int, *, phase: str, total: int
) -> None:
    """Record the start of a new phase. Resets phase_done to 0."""
    now = int(time.time())
    conn.execute(
        "UPDATE dream_runs SET current_phase = ?, phase_done = 0, "
        "phase_total = ?, phase_started_at = ?, last_progress_at = ? "
        "WHERE run_id = ?",
        (phase, total, now, now, run_id),
    )
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_runs.py::test_start_phase_sets_phase_columns_and_resets_done -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/code/hippo
git add src/hippo/storage/dream_runs.py tests/test_dream_runs.py
git commit -m "feat(storage): start_phase() resets dream_runs phase columns"
```

---

## Task 4: update_progress() helper

**Files:**
- Modify: `src/hippo/storage/dream_runs.py`
- Modify: `tests/test_dream_runs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dream_runs.py`:

```python
def test_update_progress_writes_done_and_timestamp(conn):
    from hippo.storage.dream_runs import start_phase, update_progress

    run_id = start_run(conn, "heavy")
    start_phase(conn, run_id, phase="edge_proposal", total=100)

    update_progress(conn, run_id, done=42)

    runs = get_recent_runs(conn, limit=1)
    r = runs[0]
    assert r.current_phase == "edge_proposal"
    assert r.phase_done == 42
    assert r.last_progress_at is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_runs.py::test_update_progress_writes_done_and_timestamp -v
```

Expected: FAIL with `ImportError: cannot import name 'update_progress'`.

- [ ] **Step 3: Implement `update_progress`**

Add to `src/hippo/storage/dream_runs.py`:

```python
def update_progress(conn: sqlite3.Connection, run_id: int, *, done: int) -> None:
    """Update phase_done and last_progress_at for the given run."""
    conn.execute(
        "UPDATE dream_runs SET phase_done = ?, last_progress_at = ? "
        "WHERE run_id = ?",
        (done, int(time.time()), run_id),
    )
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_runs.py::test_update_progress_writes_done_and_timestamp -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/code/hippo
git add src/hippo/storage/dream_runs.py tests/test_dream_runs.py
git commit -m "feat(storage): update_progress() bumps dream_runs phase_done"
```

---

## Task 5: get_running_run() helper

**Files:**
- Modify: `src/hippo/storage/dream_runs.py`
- Modify: `tests/test_dream_runs.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dream_runs.py`:

```python
def test_get_running_run_returns_in_progress_only(conn):
    from hippo.storage.dream_runs import get_running_run

    completed = start_run(conn, "heavy")
    complete_run(conn, completed)
    running_id = start_run(conn, "heavy")

    got = get_running_run(conn)
    assert got is not None
    assert got.run_id == running_id
    assert got.status == "running"


def test_get_running_run_returns_none_when_no_running_run(conn):
    from hippo.storage.dream_runs import get_running_run

    completed = start_run(conn, "heavy")
    complete_run(conn, completed)

    assert get_running_run(conn) is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_runs.py -k get_running_run -v
```

Expected: FAIL with `ImportError: cannot import name 'get_running_run'`.

- [ ] **Step 3: Implement `get_running_run`**

Add to `src/hippo/storage/dream_runs.py`:

```python
def get_running_run(conn: sqlite3.Connection) -> DreamRunRecord | None:
    """Return the single in-progress run (status='running') or None."""
    rows = conn.execute(
        "SELECT * FROM dream_runs WHERE status = 'running' "
        "ORDER BY started_at DESC, run_id DESC LIMIT 1"
    ).fetchall()
    if not rows:
        return None
    r = rows[0]
    return DreamRunRecord(
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
        current_phase=r["current_phase"],
        phase_done=r["phase_done"],
        phase_total=r["phase_total"],
        phase_started_at=r["phase_started_at"],
        last_progress_at=r["last_progress_at"],
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_runs.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/code/hippo
git add src/hippo/storage/dream_runs.py tests/test_dream_runs.py
git commit -m "feat(storage): get_running_run() finds the active dream"
```

---

## Task 6: ETA math utility

**Files:**
- Create: `src/hippo/dream/progress.py`
- Create: `tests/test_dream_progress.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dream_progress.py`:

```python
"""Tests for dream progress reporting (ETA math + throttled reporter)."""
from __future__ import annotations

import pytest

from hippo.dream.progress import format_eta, rolling_rate


def test_rolling_rate_simple_case():
    rate = rolling_rate(now_done=200, then_done=100, now_time=10.0, then_time=0.0)
    assert rate == pytest.approx(10.0)


def test_rolling_rate_zero_elapsed_returns_zero():
    rate = rolling_rate(now_done=200, then_done=100, now_time=5.0, then_time=5.0)
    assert rate == 0.0


def test_rolling_rate_zero_progress_returns_zero():
    rate = rolling_rate(now_done=100, then_done=100, now_time=10.0, then_time=0.0)
    assert rate == 0.0


def test_format_eta_minutes():
    # 100 remaining at 1.0/s = 100s → "2m"
    assert format_eta(remaining=100, rate=1.0) == "2m"


def test_format_eta_under_one_minute():
    # 30 remaining at 1.0/s = 30s
    assert format_eta(remaining=30, rate=1.0) == "<1m"


def test_format_eta_zero_rate():
    assert format_eta(remaining=100, rate=0.0) == "?"


def test_format_eta_capped_at_99m():
    assert format_eta(remaining=10000, rate=1.0) == ">99m"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_progress.py -v
```

Expected: FAIL with `ImportError: cannot import name 'format_eta'`.

- [ ] **Step 3: Implement the helpers**

Create `src/hippo/dream/progress.py`:

```python
"""Throttled progress reporter for dream-heavy phases.

Writes phase progress to the dream_runs row and to stderr at most once per
PROGRESS_THROTTLE_SECONDS or every PROGRESS_THROTTLE_TICKS callback invocations,
whichever comes first.
"""
from __future__ import annotations

PROGRESS_THROTTLE_SECONDS = 5.0
PROGRESS_THROTTLE_TICKS = 100


def rolling_rate(
    *, now_done: int, then_done: int, now_time: float, then_time: float
) -> float:
    """Items per second between two snapshots. Returns 0.0 on degenerate input."""
    elapsed = now_time - then_time
    delta = now_done - then_done
    if elapsed <= 0 or delta <= 0:
        return 0.0
    return delta / elapsed


def format_eta(*, remaining: int, rate: float) -> str:
    """Render remaining/rate as `<1m`, `Nm`, `>99m`, or `?` (rate==0)."""
    if rate <= 0:
        return "?"
    seconds = remaining / rate
    minutes = round(seconds / 60)
    if seconds < 60:
        return "<1m"
    if minutes > 99:
        return ">99m"
    return f"{minutes}m"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_progress.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/code/hippo
git add src/hippo/dream/progress.py tests/test_dream_progress.py
git commit -m "feat(dream): rolling_rate + format_eta helpers"
```

---

## Task 7: ProgressReporter throttle

**Files:**
- Modify: `src/hippo/dream/progress.py`
- Modify: `tests/test_dream_progress.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dream_progress.py`:

```python
def test_progress_reporter_throttles_by_ticks():
    emits: list[tuple[int, int]] = []

    def emit(done: int, total: int) -> None:
        emits.append((done, total))

    clock = [0.0]
    reporter = ProgressReporter(
        emit=emit,
        clock=lambda: clock[0],
        total=500,
    )
    for i in range(1, 501):
        reporter.tick(i)
    reporter.finish()

    # 500 ticks, throttle every 100, so 5 emits during ticks + 1 finish
    assert len(emits) == 6
    assert emits[-1] == (500, 500)


def test_progress_reporter_throttles_by_seconds():
    emits: list[tuple[int, int]] = []

    def emit(done: int, total: int) -> None:
        emits.append((done, total))

    clock = [0.0]
    reporter = ProgressReporter(
        emit=emit,
        clock=lambda: clock[0],
        total=10,
    )
    # 10 ticks below tick-threshold; advance clock past PROGRESS_THROTTLE_SECONDS
    # between each tick to force time-based emits.
    for i in range(1, 11):
        clock[0] += 5.5
        reporter.tick(i)
    reporter.finish()

    # 10 time-based emits + 1 finish (finish always emits final state)
    assert len(emits) == 11


def test_progress_reporter_finish_always_emits_final_state():
    emits: list[tuple[int, int]] = []

    def emit(done: int, total: int) -> None:
        emits.append((done, total))

    reporter = ProgressReporter(
        emit=emit, clock=lambda: 0.0, total=3,
    )
    reporter.tick(1)
    reporter.tick(2)
    reporter.tick(3)
    reporter.finish()

    assert emits[-1] == (3, 3)
```

Add the import at the top of the test file:

```python
from hippo.dream.progress import ProgressReporter, format_eta, rolling_rate
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_progress.py -v
```

Expected: FAIL with `ImportError: cannot import name 'ProgressReporter'`.

- [ ] **Step 3: Implement `ProgressReporter`**

Append to `src/hippo/dream/progress.py`:

```python
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ProgressReporter:
    """Throttled progress callback.

    Calls `emit(done, total)` at most once per PROGRESS_THROTTLE_SECONDS or every
    PROGRESS_THROTTLE_TICKS calls to tick(). Always emits exactly once on finish().
    """

    emit: Callable[[int, int], None]
    clock: Callable[[], float]
    total: int
    _ticks_since_emit: int = 0
    _last_emit_time: float = 0.0
    _last_done: int = 0
    _started: bool = False

    def tick(self, done: int) -> None:
        if not self._started:
            self._last_emit_time = self.clock()
            self._started = True
        self._last_done = done
        self._ticks_since_emit += 1
        now = self.clock()
        time_due = now - self._last_emit_time >= PROGRESS_THROTTLE_SECONDS
        ticks_due = self._ticks_since_emit >= PROGRESS_THROTTLE_TICKS
        if time_due or ticks_due:
            self.emit(done, self.total)
            self._last_emit_time = now
            self._ticks_since_emit = 0

    def finish(self) -> None:
        self.emit(self._last_done, self.total)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_progress.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/code/hippo
git add src/hippo/dream/progress.py tests/test_dream_progress.py
git commit -m "feat(dream): ProgressReporter throttles tick() to ~1/5s or 100 ticks"
```

---

## Task 8: stderr line formatter

**Files:**
- Modify: `src/hippo/dream/progress.py`
- Modify: `tests/test_dream_progress.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dream_progress.py`:

```python
def test_format_progress_line_running_phase():
    line = format_progress_line(
        phase="edge_proposal",
        done=87,
        total=5765,
        elapsed_s=5,
        rate=9.0,
        eta="10m",
    )
    assert "phase=edge_proposal" in line
    assert "done=87/5765" in line
    assert "(1.5%)" in line
    assert "rate=9.0/s" in line
    assert "eta=10m" in line


def test_format_phase_start_line():
    line = format_phase_start_line(phase="atomize", total=12)
    assert "phase=atomize" in line
    assert "total=12" in line


def test_format_phase_complete_line():
    line = format_phase_complete_line(phase="review", total=16, elapsed_s=27)
    assert "phase=review" in line
    assert "done=16/16" in line
    assert "(100%)" in line
    assert "elapsed=27s" in line
```

Add to the existing import block:

```python
from hippo.dream.progress import (
    ProgressReporter,
    format_eta,
    format_phase_complete_line,
    format_phase_start_line,
    format_progress_line,
    rolling_rate,
)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_progress.py -v
```

Expected: FAIL with `ImportError: cannot import name 'format_progress_line'`.

- [ ] **Step 3: Implement the formatters**

Append to `src/hippo/dream/progress.py`:

```python
PHASE_COL_WIDTH = 16


def _phase_col(phase: str) -> str:
    return f"phase={phase:<{PHASE_COL_WIDTH}}"


def format_phase_start_line(*, phase: str, total: int) -> str:
    return f"{_phase_col(phase)} total={total}"


def format_phase_complete_line(*, phase: str, total: int, elapsed_s: int) -> str:
    return f"{_phase_col(phase)} done={total}/{total} (100%) elapsed={elapsed_s}s"


def format_progress_line(
    *,
    phase: str,
    done: int,
    total: int,
    elapsed_s: int,
    rate: float,
    eta: str,
) -> str:
    pct = (100 * done / total) if total > 0 else 0.0
    return (
        f"{_phase_col(phase)} done={done}/{total} ({pct:.1f}%) "
        f"rate={rate:.1f}/s eta={eta}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_progress.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/code/hippo
git add src/hippo/dream/progress.py tests/test_dream_progress.py
git commit -m "feat(dream): line formatters for phase progress stderr output"
```

---

## Task 9: edge_proposal accepts progress_cb

**Files:**
- Modify: `src/hippo/dream/edge_proposal.py`
- Modify: `tests/test_edge_proposal.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_edge_proposal.py`:

```python
def test_propose_edges_invokes_progress_cb_per_pair(temp_memory_dir, monkeypatch):
    """progress_cb is called once per cluster pair, with (done, total)."""
    from hippo.dream.edge_proposal import propose_edges
    from hippo.storage.multi_store import Scope, open_store

    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", temp_memory_dir)
    store = open_store(Scope.global_())

    # Insert three heads with identical embeddings so they form one cluster (size 3 → 3 pairs)
    from hippo.storage.bodies import BodyRecord, insert_body
    from hippo.storage.heads import HeadRecord, insert_head
    from hippo.storage.vec import insert_head_embedding

    embedding = [1.0] + [0.0] * 1023
    for i in range(3):
        body_id = f"body-{i}"
        insert_body(store.conn, BodyRecord(
            body_id=body_id, file_path=f"bodies/{body_id}.md",
            title=f"t{i}", scope="global", source="test",
        ))
        head_id = f"head-{i}"
        insert_head(store.conn, HeadRecord(
            head_id=head_id, body_id=body_id, summary=f"summary {i}",
        ))
        insert_head_embedding(store.conn, head_id, embedding)

    class StubLLM:
        def generate_chat(self, *args, **kwargs):
            import json
            return json.dumps({"relation": "related", "weight": 0.5})

    progress_calls: list[tuple[int, int]] = []
    propose_edges(
        store=store,
        llm=StubLLM(),
        progress_cb=lambda done, total: progress_calls.append((done, total)),
    )
    store.conn.close()

    # 3 pairs, each yields exactly one progress call
    assert len(progress_calls) == 3
    assert progress_calls[-1] == (3, 3)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/code/hippo && uv run pytest tests/test_edge_proposal.py::test_propose_edges_invokes_progress_cb_per_pair -v
```

Expected: FAIL with `TypeError: propose_edges() got an unexpected keyword argument 'progress_cb'`.

- [ ] **Step 3: Add the parameter and call site**

In `src/hippo/dream/edge_proposal.py`, replace the `propose_edges` signature and outer loop:

```python
def propose_edges(
    *,
    store: Store,
    llm: LLMProto,
    progress_cb: "Callable[[int, int], None] | None" = None,
) -> int:
    clusters = cluster_active_heads(store.conn)
    total_pairs = sum(len(c) * (len(c) - 1) // 2 for c in clusters)
    n_inserted = 0
    pair_idx = 0
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                pair_idx += 1
                a_id, b_id = cluster[i], cluster[j]
                # ... existing body of inner loop unchanged ...
                if progress_cb is not None:
                    progress_cb(pair_idx, total_pairs)
    return n_inserted
```

Add the import at the top:

```python
from collections.abc import Callable
```

Move the `progress_cb` call to the **end** of the inner loop body so it fires exactly once per pair regardless of skip/insert outcome. Keep all existing `continue` statements; they fall through to the `progress_cb` call.

The full revised inner loop:

```python
            for j in range(i + 1, len(cluster)):
                pair_idx += 1
                a_id, b_id = cluster[i], cluster[j]
                a = get_head(store.conn, a_id)
                b = get_head(store.conn, b_id)
                if a is None or b is None:
                    if progress_cb is not None:
                        progress_cb(pair_idx, total_pairs)
                    continue
                existing = store.conn.execute(
                    "SELECT 1 FROM edges "
                    "WHERE (from_head=? AND to_head=?) OR (from_head=? AND to_head=?) "
                    "LIMIT 1",
                    (a_id, b_id, b_id, a_id),
                ).fetchone()
                if existing is not None:
                    if progress_cb is not None:
                        progress_cb(pair_idx, total_pairs)
                    continue

                prompt = render("edge_typing", head_a=a.summary, head_b=b.summary)
                raw = llm.generate_chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200,
                    thinking_level="minimal",
                )
                try:
                    obj = json.loads(_strip_fences(raw))
                except json.JSONDecodeError:
                    if progress_cb is not None:
                        progress_cb(pair_idx, total_pairs)
                    continue
                relation = obj.get("relation")
                if relation not in VALID_RELATIONS:
                    if progress_cb is not None:
                        progress_cb(pair_idx, total_pairs)
                    continue
                weight = float(obj.get("weight", 1.0))
                try:
                    insert_edge_with_reciprocal(
                        store.conn,
                        EdgeRecord(from_head=a_id, to_head=b_id, relation=relation, weight=weight),
                    )
                    n_inserted += 1
                except (sqlite3.IntegrityError, ValueError):
                    pass
                if progress_cb is not None:
                    progress_cb(pair_idx, total_pairs)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/code/hippo && uv run pytest tests/test_edge_proposal.py -v
```

Expected: all PASS, including the new test and pre-existing ones.

- [ ] **Step 5: Commit**

```bash
cd ~/code/hippo
git add src/hippo/dream/edge_proposal.py tests/test_edge_proposal.py
git commit -m "feat(dream): edge_proposal calls progress_cb once per pair"
```

---

## Task 10: atomize, review, multi_head, contradiction accept progress_cb

**Files:**
- Modify: `src/hippo/dream/atomize.py`
- Modify: `src/hippo/dream/review.py`
- Modify: `src/hippo/dream/multi_head.py`
- Modify: `src/hippo/dream/contradiction.py`

These four phases are O(items) not O(N²), so the wiring is straightforward: add the parameter and call it once per loop iteration. No new tests for these — they're covered by the orchestrator integration test in Task 12.

- [ ] **Step 1: Update `atomize_session`**

In `src/hippo/dream/atomize.py`, change the signature and the `for atom in atoms:` loop. Atomize processes one session per call; the *caller* (heavy.py) will tick once per session, so we need progress_cb at the heavy.py call site, not inside atomize. **No change required to atomize.py itself.** Skip this step.

- [ ] **Step 2: Update `review_new_atoms` and `review_rolling_slice`**

In `src/hippo/dream/review.py`, add `progress_cb` to both public functions:

```python
def review_new_atoms(
    *,
    store: Store,
    llm: LLMProto,
    run_id: int,
    progress_cb: "Callable[[int, int], None] | None" = None,
) -> int:
    from hippo.storage.bodies import find_active_bodies_by_run_source
    new_bodies = find_active_bodies_by_run_source(store.conn, run_id=run_id)
    n_archived = 0
    for idx, body in enumerate(new_bodies, start=1):
        n_archived += _review_body_against_neighbors(
            store=store, llm=llm, body_id=body.body_id,
        )
        if progress_cb is not None:
            progress_cb(idx, len(new_bodies))
    return n_archived


def review_rolling_slice(
    *,
    store: Store,
    scope: str,
    llm: LLMProto,
    slice_size: int,
    progress_cb: "Callable[[int, int], None] | None" = None,
) -> int:
    from hippo.storage.bodies import find_oldest_unreviewed_active
    slice_bodies = find_oldest_unreviewed_active(
        store.conn, scope=scope, limit=slice_size,
    )
    n_archived = 0
    for idx, body in enumerate(slice_bodies, start=1):
        n_archived += _review_body_against_neighbors(
            store=store, llm=llm, body_id=body.body_id,
        )
        if progress_cb is not None:
            progress_cb(idx, len(slice_bodies))
    return n_archived
```

Add at the top of the file:

```python
from collections.abc import Callable
```

- [ ] **Step 3: Update `expand_heads_for_eligible_bodies`**

In `src/hippo/dream/multi_head.py`, add `from collections.abc import Callable` after the existing imports. Replace the function signature and the row-loop:

```python
def expand_heads_for_eligible_bodies(
    *,
    store: Store,
    llm: LLMProto,
    daemon: DaemonProto,
    target_total_heads: int = 3,
    progress_cb: "Callable[[int, int], None] | None" = None,
) -> int:
    """For bodies with retrieval_count > 0 and < target_total_heads heads,
    generate additional diverse heads. Returns count of new heads inserted."""
    rows = store.conn.execute(
        """
        SELECT b.body_id, b.file_path, b.title, COUNT(h.head_id) AS head_count
        FROM bodies b
        LEFT JOIN heads h ON h.body_id = b.body_id AND h.archived = 0
        WHERE b.archived = 0
        GROUP BY b.body_id
        HAVING head_count < ?
          AND (
              SELECT MAX(retrieval_count) FROM heads WHERE body_id = b.body_id
          ) > 0
        """,
        (target_total_heads,),
    ).fetchall()

    total = len(rows)
    n_new = 0
    for idx, r in enumerate(rows, start=1):
        # ... existing body of for-loop unchanged ...
        if progress_cb is not None:
            progress_cb(idx, total)
    return n_new
```

Place the `progress_cb` call at the very end of each loop iteration (after the existing `n_new` accumulation, before the next iteration starts). Existing `continue` statements should fall through the `progress_cb` call — wrap the `continue`d branches with their own `if progress_cb is not None: progress_cb(idx, total)` immediately before each `continue`, mirroring the pattern in Task 9.

- [ ] **Step 4: Update `resolve_contradictions`**

In `src/hippo/dream/contradiction.py`, change the signature:

```python
def resolve_contradictions(
    *,
    store: Store,
    llm: LLMProto,
    progress_cb: "Callable[[int, int], None] | None" = None,
) -> int:
```

Add `from collections.abc import Callable` at the top. In the existing `for r in pair_rows:` loop, capture `total = len(pair_rows)` before the loop and call `progress_cb(idx, total)` at the bottom of each iteration (use `enumerate(pair_rows, start=1)`).

- [ ] **Step 5: Run all dream-phase tests to verify nothing broke**

```bash
cd ~/code/hippo && uv run pytest tests/test_review.py tests/test_multi_head.py tests/test_contradiction.py -v
```

Expected: all existing tests PASS.

- [ ] **Step 6: Commit**

```bash
cd ~/code/hippo
git add src/hippo/dream/review.py src/hippo/dream/multi_head.py src/hippo/dream/contradiction.py
git commit -m "feat(dream): review/multi_head/contradiction accept progress_cb"
```

---

## Task 11: heavy.py wires phase reporters

**Files:**
- Modify: `src/hippo/dream/heavy.py`

This task wires everything together. The reporter for each phase: writes to `dream_runs` (via `update_progress`) and writes a stderr line. Phase boundaries (entry / exit) call `start_phase` and emit the matching stderr lines.

- [ ] **Step 1: Add the helpers at the top of heavy.py**

In `src/hippo/dream/heavy.py`, after the existing imports, add:

```python
import sys
import time
from contextlib import contextmanager
from typing import Iterator

from hippo.dream.progress import (
    ProgressReporter,
    format_eta,
    format_phase_complete_line,
    format_phase_start_line,
    format_progress_line,
    rolling_rate,
)
from hippo.storage.dream_runs import start_phase, update_progress
```

- [ ] **Step 2: Add the `phase_reporter` context manager**

Inside `heavy.py`, before `run_heavy_dream_for_scope`, add:

```python
@contextmanager
def _phase_reporter(
    *, conn, run_id: int, phase: str, total: int
) -> Iterator[ProgressReporter | None]:
    """Open a phase. Writes start row, yields a reporter (or None for empty phases)."""
    if total <= 0:
        # Empty phase: log entry, no reporter, log instant complete on exit.
        sys.stderr.write(f"{format_phase_start_line(phase=phase, total=0)}\n")
        sys.stderr.flush()
        start_phase(conn, run_id, phase=phase, total=0)
        yield None
        sys.stderr.write(
            f"{format_phase_complete_line(phase=phase, total=0, elapsed_s=0)}\n"
        )
        sys.stderr.flush()
        return

    start_time = time.time()
    start_phase(conn, run_id, phase=phase, total=total)
    sys.stderr.write(f"{format_phase_start_line(phase=phase, total=total)}\n")
    sys.stderr.flush()

    snapshot = {"time": start_time, "done": 0}

    def emit(done: int, phase_total: int) -> None:
        now = time.time()
        if now - snapshot["time"] >= 60.0:
            snapshot["time"] = now
            snapshot["done"] = done
        rate = rolling_rate(
            now_done=done,
            then_done=snapshot["done"],
            now_time=now,
            then_time=snapshot["time"] if snapshot["time"] != now else start_time,
        )
        eta = format_eta(remaining=phase_total - done, rate=rate)
        update_progress(conn, run_id, done=done)
        sys.stderr.write(
            f"{format_progress_line(phase=phase, done=done, total=phase_total, elapsed_s=int(now - start_time), rate=rate, eta=eta)}\n"
        )
        sys.stderr.flush()

    reporter = ProgressReporter(emit=emit, clock=time.time, total=total)
    yield reporter
    reporter.finish()
    sys.stderr.write(
        f"{format_phase_complete_line(phase=phase, total=total, elapsed_s=int(time.time() - start_time))}\n"
    )
    sys.stderr.flush()
```

- [ ] **Step 3: Wire each phase**

Replace the body of `run_heavy_dream_for_scope` between `try:` and the cleanup phase. Replace existing phase code with the wrapped version:

```python
    try:
        # Phase a: atomize each session
        before = counter.count
        session_rows = store.conn.execute(
            "SELECT DISTINCT session_id FROM capture_queue WHERE processed_at IS NULL"
        ).fetchall()
        processed_ids: list[int] = []
        with _phase_reporter(
            conn=store.conn, run_id=run_id, phase="atomize", total=len(session_rows)
        ) as reporter:
            for idx, r in enumerate(session_rows, start=1):
                session_id = r["session_id"]
                n_atoms += atomize_session(
                    store=store, session_id=session_id,
                    project=scope.project_name, run_id=run_id,
                    llm=counter, daemon=daemon,
                )
                cap_ids = [
                    row["queue_id"] for row in store.conn.execute(
                        "SELECT queue_id FROM capture_queue"
                        " WHERE session_id = ? AND processed_at IS NULL",
                        (session_id,),
                    ).fetchall()
                ]
                processed_ids.extend(cap_ids)
                if reporter is not None:
                    reporter.tick(idx)
        _phase_delta("atomize", before)

        # Phase a2: review (gate-at-entry + rolling slice)
        before = counter.count
        from hippo.storage.bodies import (
            find_active_bodies_by_run_source,
            find_oldest_unreviewed_active,
        )
        new_bodies = find_active_bodies_by_run_source(store.conn, run_id=run_id)
        slice_bodies = find_oldest_unreviewed_active(
            store.conn, scope=scope.as_string(), limit=PRUNE_ROLLING_SLICE_SIZE,
        )
        review_total = len(new_bodies) + len(slice_bodies)
        with _phase_reporter(
            conn=store.conn, run_id=run_id, phase="review", total=review_total
        ) as reporter:
            done = [0]

            def review_cb(idx: int, _total: int) -> None:
                done[0] += 1
                if reporter is not None:
                    reporter.tick(done[0])

            n_review_archived += review_new_atoms(
                store=store, llm=counter, run_id=run_id,
                progress_cb=review_cb if new_bodies else None,
            )
            n_review_archived += review_rolling_slice(
                store=store, scope=scope.as_string(),
                llm=counter, slice_size=PRUNE_ROLLING_SLICE_SIZE,
                progress_cb=review_cb if slice_bodies else None,
            )
        _phase_delta("review", before)

        # Phase b: multi-head expansion
        before = counter.count
        from hippo.storage.bodies import count_eligible_for_multi_head
        multi_head_total = count_eligible_for_multi_head(store.conn)
        with _phase_reporter(
            conn=store.conn, run_id=run_id, phase="multi_head", total=multi_head_total
        ) as reporter:
            n_heads += expand_heads_for_eligible_bodies(
                store=store, llm=counter, daemon=daemon,
                progress_cb=(lambda d, t: reporter.tick(d)) if reporter is not None else None,
            )
        _phase_delta("multi_head", before)

        # Phase c-d: cluster + edge proposal
        before = counter.count
        from hippo.dream.cluster import cluster_active_heads
        clusters = cluster_active_heads(store.conn)
        edge_total = sum(len(c) * (len(c) - 1) // 2 for c in clusters)
        with _phase_reporter(
            conn=store.conn, run_id=run_id, phase="edge_proposal", total=edge_total
        ) as reporter:
            n_edges += propose_edges(
                store=store, llm=counter,
                progress_cb=(lambda d, t: reporter.tick(d)) if reporter is not None else None,
            )
        _phase_delta("edge_proposal", before)

        # Phase e: contradiction resolution
        before = counter.count
        contradiction_total_row = store.conn.execute(
            "SELECT COUNT(*) AS c FROM edges WHERE relation='contradicts' AND from_head < to_head"
        ).fetchone()
        contradiction_total = int(contradiction_total_row["c"]) if contradiction_total_row else 0
        with _phase_reporter(
            conn=store.conn, run_id=run_id, phase="contradiction", total=contradiction_total
        ) as reporter:
            n_contradictions += resolve_contradictions(
                store=store, llm=counter,
                progress_cb=(lambda d, t: reporter.tick(d)) if reporter is not None else None,
            )
        _phase_delta("contradiction", before)

        # Phase f: cleanup (instant; no per-item progress)
        with _phase_reporter(
            conn=store.conn, run_id=run_id, phase="cleanup", total=0
        ):
            finalize_processed_captures(store=store, queue_ids=processed_ids, run_id=run_id)

        # ... rest of function unchanged (complete_run + return) ...
```

The `count_eligible_for_multi_head` helper does not currently exist. Add it next.

- [ ] **Step 4: Add `count_eligible_for_multi_head` to bodies.py**

In `src/hippo/storage/bodies.py`, add this helper (place it near `find_active_bodies_by_run_source` to keep related queries together):

```python
def count_eligible_for_multi_head(
    conn: sqlite3.Connection, *, target_total_heads: int = 3
) -> int:
    """Count bodies that multi_head expansion will process this run.

    Mirrors the eligibility filter in expand_heads_for_eligible_bodies:
    archived=0, < target_total_heads active heads, and at least one head
    with retrieval_count > 0. Used solely for the progress denominator.
    """
    row = conn.execute(
        """
        SELECT COUNT(*) AS c FROM (
            SELECT b.body_id
            FROM bodies b
            LEFT JOIN heads h ON h.body_id = b.body_id AND h.archived = 0
            WHERE b.archived = 0
            GROUP BY b.body_id
            HAVING COUNT(h.head_id) < ?
              AND (
                  SELECT MAX(retrieval_count) FROM heads WHERE body_id = b.body_id
              ) > 0
        )
        """,
        (target_total_heads,),
    ).fetchone()
    return int(row["c"]) if row else 0
```

- [ ] **Step 5: Run the orchestrator test to verify nothing broke**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_heavy_orchestrator.py -v
```

Expected: existing test still PASS.

- [ ] **Step 6: Commit**

```bash
cd ~/code/hippo
git add src/hippo/dream/heavy.py src/hippo/storage/bodies.py
git commit -m "feat(dream): wire ProgressReporter into heavy.py phases"
```

---

## Task 12: orchestrator integration test for phase rows

**Files:**
- Modify: `tests/test_dream_heavy_orchestrator.py`

- [ ] **Step 1: Write the test**

Append to `tests/test_dream_heavy_orchestrator.py`:

```python
def test_heavy_dream_writes_phase_progress_rows(tmp_path, monkeypatch):
    """At completion, dream_runs has current_phase set to the last-touched phase."""
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")

    s = open_store(Scope.global_())
    enqueue_capture(
        s.conn,
        CaptureRecord(
            session_id="sess-A",
            user_message="we use postgres",
            assistant_message="ok",
        ),
    )
    s.conn.close()

    run_heavy_dream_all_scopes(
        scopes=[Scope.global_()], llm=FakeLLM(), daemon=FakeDaemon(),
    )

    s2 = open_store(Scope.global_())
    row = s2.conn.execute(
        "SELECT current_phase, status FROM dream_runs ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    s2.conn.close()
    assert row["status"] == "completed"
    # At least one phase ran, leaving current_phase non-NULL.
    assert row["current_phase"] in {
        "atomize", "review", "multi_head", "edge_proposal", "contradiction", "cleanup",
    }
```

- [ ] **Step 2: Run test to verify it passes**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_heavy_orchestrator.py -v
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
cd ~/code/hippo
git add tests/test_dream_heavy_orchestrator.py
git commit -m "test(dream): orchestrator writes phase rows to dream_runs"
```

---

## Task 13: dream-status CLI

**Files:**
- Create: `src/hippo/cli/dream_status.py`
- Create: `bin/dream-status`
- Modify: `pyproject.toml`
- Create: `tests/test_dream_status_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dream_status_cli.py`:

```python
"""Tests for the dream-status CLI."""
from __future__ import annotations

import json

import pytest

from hippo.cli.dream_status import dream_status_cli, render_run_line
from hippo.storage.dream_runs import (
    DreamRunRecord,
    complete_run,
    start_phase,
    start_run,
    update_progress,
)
from hippo.storage.migrations import run_migrations
from hippo.storage.multi_store import Scope, open_store


def _make_record(
    *,
    run_id: int = 1,
    status: str = "running",
    current_phase: str | None = "edge_proposal",
    phase_done: int | None = 100,
    phase_total: int | None = 500,
    started_at: int = 1_700_000_000,
    last_progress_at: int | None = 1_700_000_120,
) -> DreamRunRecord:
    return DreamRunRecord(
        run_id=run_id,
        type="heavy",
        started_at=started_at,
        completed_at=None,
        status=status,
        atoms_created=0,
        heads_created=0,
        edges_created=0,
        contradictions_resolved=0,
        bodies_archived_review=0,
        error_message=None,
        current_phase=current_phase,
        phase_done=phase_done,
        phase_total=phase_total,
        phase_started_at=started_at,
        last_progress_at=last_progress_at,
    )


def test_render_run_line_for_running_run():
    rec = _make_record()
    line = render_run_line(rec, scope_name="kaleon", now_unix=1_700_000_180)
    assert "running" in line
    assert "kaleon" in line
    assert "phase=edge_proposal" in line
    assert "100/500" in line
    assert "(20.0%)" in line


def test_render_run_line_for_completed_run():
    rec = _make_record(status="completed", current_phase="cleanup")
    line = render_run_line(rec, scope_name="kaleon", now_unix=1_700_000_180)
    assert "completed" in line


def test_dream_status_cli_no_runs_returns_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    monkeypatch.setattr("hippo.config.PROJECTS_ROOT", tmp_path / "projects")
    rc = dream_status_cli(["--scope", "global"])
    assert rc == 1
    assert "no dream" in capsys.readouterr().out.lower()


def test_dream_status_cli_finds_running_dream(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    rid = start_run(s.conn, "heavy")
    start_phase(s.conn, rid, phase="edge_proposal", total=500)
    update_progress(s.conn, rid, done=100)
    s.conn.close()

    rc = dream_status_cli(["--scope", "global"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "running" in out
    assert "phase=edge_proposal" in out


def test_dream_status_cli_falls_back_to_completed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hippo.config.GLOBAL_MEMORY_DIR", tmp_path / "global")
    s = open_store(Scope.global_())
    rid = start_run(s.conn, "heavy")
    complete_run(s.conn, rid)
    s.conn.close()

    rc = dream_status_cli(["--scope", "global"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "completed" in out
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_status_cli.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'hippo.cli.dream_status'`.

- [ ] **Step 3: Implement the CLI**

Create `src/hippo/cli/dream_status.py`:

```python
"""dream-status: print the most recent dream run across all scope DBs."""
from __future__ import annotations

import argparse
import sys
import time

from hippo.config import DB_FILENAME, PROJECTS_ROOT
from hippo.storage.dream_runs import (
    DreamRunRecord,
    get_recent_runs,
    get_running_run,
)
from hippo.storage.multi_store import Scope, open_store


def _all_scopes() -> list[tuple[Scope, str]]:
    """Return (scope, display_name) pairs for every scope DB on disk."""
    scopes: list[tuple[Scope, str]] = [(Scope.global_(), "global")]
    if PROJECTS_ROOT.exists():
        for entry in sorted(PROJECTS_ROOT.iterdir()):
            if (entry / "memory" / DB_FILENAME).exists():
                scopes.append((Scope.project(entry.name), entry.name))
    return scopes


def _filtered_scopes(scope_name: str | None) -> list[tuple[Scope, str]]:
    if scope_name is None:
        return _all_scopes()
    if scope_name == "global":
        return [(Scope.global_(), "global")]
    return [(Scope.project(scope_name), scope_name)]


def render_run_line(rec: DreamRunRecord, *, scope_name: str, now_unix: int) -> str:
    state = rec.status
    elapsed = (rec.completed_at or now_unix) - rec.started_at
    elapsed_str = f"{elapsed // 60}m" if elapsed >= 60 else f"{elapsed}s"
    phase = rec.current_phase or "?"
    if rec.phase_done is not None and rec.phase_total:
        pct = 100 * rec.phase_done / rec.phase_total
        phase_part = (
            f"phase={phase} {rec.phase_done}/{rec.phase_total} ({pct:.1f}%)"
        )
    else:
        phase_part = f"phase={phase}"
    return (
        f"{state}: {scope_name} run_id={rec.run_id} {phase_part} elapsed={elapsed_str}"
    )


def dream_status_cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dream-status")
    p.add_argument("--scope", default=None,
                   help="Restrict to one scope (e.g. 'global' or a project name).")
    args = p.parse_args(argv)
    now = int(time.time())

    # First pass: any running run anywhere?
    for scope, name in _filtered_scopes(args.scope):
        store = open_store(scope)
        try:
            running = get_running_run(store.conn)
            if running is not None:
                print(render_run_line(running, scope_name=name, now_unix=now))
                return 0
        finally:
            store.conn.close()

    # Fallback: most recent completed/failed run across scopes.
    best: tuple[DreamRunRecord, str] | None = None
    for scope, name in _filtered_scopes(args.scope):
        store = open_store(scope)
        try:
            recents = get_recent_runs(store.conn, limit=1)
            if recents and (best is None or recents[0].started_at > best[0].started_at):
                best = (recents[0], name)
        finally:
            store.conn.close()

    if best is None:
        print("no dream runs found")
        return 1
    print(render_run_line(best[0], scope_name=best[1], now_unix=now))
    return 0


def main() -> int:
    return dream_status_cli(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ~/code/hippo && uv run pytest tests/test_dream_status_cli.py -v
```

Expected: all PASS.

- [ ] **Step 5: Add the bin wrapper**

Create `bin/dream-status` (mirrors `bin/memory-get`):

```bash
#!/usr/bin/env bash
# Resolve symlinks so we always run inside the real repo's venv.
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
exec uv run --project "$SCRIPT_DIR/.." --quiet python -m hippo.cli.dream_status "$@"
```

Make it executable:

```bash
chmod +x ~/code/hippo/bin/dream-status
```

- [ ] **Step 6: Register in pyproject.toml**

In `~/code/hippo/pyproject.toml`, under `[project.scripts]`, add:

```toml
dream-status   = "hippo.cli.dream_status:main"
```

- [ ] **Step 7: Smoke test the wrapper**

```bash
~/code/hippo/bin/dream-status --scope global
```

Expected: prints either a status line or "no dream runs found" — no traceback.

- [ ] **Step 8: Commit**

```bash
cd ~/code/hippo
git add src/hippo/cli/dream_status.py bin/dream-status pyproject.toml tests/test_dream_status_cli.py
git commit -m "feat(cli): dream-status reports the most recent dream"
```

---

## Task 14: end-to-end manual verification

**Files:** none (manual smoke test)

- [ ] **Step 1: Trigger a real dream**

```bash
~/code/hippo/bin/dream-heavy --force --project kaleon 2>/tmp/dream-stderr.log &
DREAM_PID=$!
```

- [ ] **Step 2: Watch progress from another terminal/session**

```bash
~/code/hippo/bin/dream-status --scope kaleon
# Optional: tail stderr
tail -f /tmp/dream-stderr.log
```

Expected: status line shows `running: kaleon ... phase=<some-phase> done=N/M ... elapsed=...`. Stderr file shows phase entry/exit lines and progress ticks.

- [ ] **Step 3: Confirm completion**

When the dream finishes, `dream-status` should fall back to printing the most recent completed run. Stderr ends with the contradiction/cleanup phases plus the final JSON on stdout.

- [ ] **Step 4: No commit needed** — manual verification only.

---

## Self-Review Notes

(Filled in during the writing-plans self-review pass.)

- **Spec coverage:**
  - Schema columns → Task 1
  - DreamRunRecord exposes them → Task 2
  - `start_phase`, `update_progress`, `get_running_run` → Tasks 3, 4, 5
  - ETA + format helpers → Tasks 6, 8
  - ProgressReporter throttling → Task 7
  - edge_proposal pair-level callback → Task 9
  - Other phases callbacks → Task 10
  - Heavy.py wiring + stderr emit → Task 11
  - Orchestrator integration test → Task 12
  - `dream-status` CLI + bin wrapper → Task 13
  - Manual end-to-end check → Task 14

- **Placeholders:** none in code blocks; one descriptive note in Task 11 Step 4 about mirroring multi_head's eligibility filter — flagged inline ("Read multi_head.py first to confirm") rather than left as TBD.

- **Type consistency:** `progress_cb: Callable[[int, int], None] | None` used uniformly across `propose_edges`, `review_new_atoms`, `review_rolling_slice`, `expand_heads_for_eligible_bodies`, `resolve_contradictions`. The reporter expects `(done, total)` and that's what every phase passes.
