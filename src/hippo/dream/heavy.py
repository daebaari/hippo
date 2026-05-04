"""Heavy dream orchestrator. Runs nightly via launchd or via /dream."""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator, Protocol

from hippo.config import HEAVY_LOCK_FILENAME, PRUNE_ROLLING_SLICE_SIZE
from hippo.dream.atomize import atomize_session
from hippo.dream.cleanup import finalize_processed_captures
from hippo.dream.cluster import cluster_active_heads
from hippo.dream.contradiction import resolve_contradictions
from hippo.dream.edge_proposal import collect_pending_pairs, propose_edges
from hippo.dream.multi_head import expand_heads_for_eligible_bodies
from hippo.dream.progress import (
    ProgressReporter,
    format_eta,
    format_phase_complete_line,
    format_phase_start_line,
    format_progress_line,
    rolling_rate,
)
from hippo.dream.review import review_new_atoms, review_rolling_slice
from hippo.lock import LockHeldError, acquire_lock, release_lock
from hippo.models.llm import LLMProto
from hippo.storage.dream_runs import (
    complete_run,
    fail_run,
    mark_orphan_runs_failed,
    start_phase,
    start_run,
    update_progress,
)
from hippo.storage.multi_store import Scope, open_store


class DaemonProto(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class _CountingLLM:
    """Transparent LLMProto wrapper that counts LLM invocations.

    `count` reflects the number of model responses produced (one per chat
    message), regardless of whether they came in via generate_chat or were
    multiplexed through generate_chat_batch."""

    def __init__(self, inner: LLMProto) -> None:
        self._inner = inner
        self.count = 0

    def generate_chat(self, *args: Any, **kwargs: Any) -> str:
        self.count += 1
        return self._inner.generate_chat(*args, **kwargs)

    def generate_chat_batch(self, *args: Any, **kwargs: Any) -> list[str]:
        out = self._inner.generate_chat_batch(*args, **kwargs)
        self.count += len(out)
        return out


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

    # We hold the per-scope lock — any 'running' row in dream_runs is therefore
    # an orphan from a previously-killed process. Flip them to failed before
    # creating ours so dream-status, get_running_run, etc. don't see ghosts.
    mark_orphan_runs_failed(store.conn)

    run_id = start_run(store.conn, "heavy")
    n_atoms = 0
    n_heads = 0
    n_edges = 0
    n_contradictions = 0
    n_review_archived = 0
    counter = _CountingLLM(llm)
    llm_calls: dict[str, int] = {}

    def _phase_delta(label: str, before: int) -> None:
        llm_calls[label] = counter.count - before

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
            done_counter = [0]

            def review_cb(idx: int, _sub_total: int) -> None:
                done_counter[0] += 1
                if reporter is not None:
                    reporter.tick(done_counter[0])

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
        clusters = cluster_active_heads(store.conn)
        # Filter to pairs that are actually pending LLM evaluation — pairs that
        # already have an edge from a prior dream run are skipped inside
        # propose_edges, so the raw cluster pair count would over-state the
        # phase total and stall the progress %% at pending/raw forever.
        pending_pairs = collect_pending_pairs(store, clusters)
        edge_total = len(pending_pairs)
        with _phase_reporter(
            conn=store.conn, run_id=run_id, phase="edge_proposal", total=edge_total
        ) as reporter:
            n_edges += propose_edges(
                store=store, llm=counter, pending=pending_pairs,
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
            "llm_calls": {"total": counter.count, **llm_calls},
        }
    except Exception as e:
        fail_run(store.conn, run_id, error_message=str(e))
        raise
    finally:
        release_lock(handle)
        store.conn.close()


def run_heavy_dream_all_scopes(
    *, scopes: list[Scope], llm: LLMProto, daemon: DaemonProto
) -> dict[str, object]:
    out: dict[str, object] = {}
    for scope in scopes:
        out[scope.as_string()] = run_heavy_dream_for_scope(scope=scope, llm=llm, daemon=daemon)
    return out
