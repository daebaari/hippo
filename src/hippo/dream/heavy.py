"""Heavy dream orchestrator. Runs nightly via launchd or via /dream."""
from __future__ import annotations

from typing import Any, Protocol

from hippo.config import HEAVY_LOCK_FILENAME, PRUNE_ROLLING_SLICE_SIZE
from hippo.dream.atomize import atomize_session
from hippo.dream.cleanup import finalize_processed_captures
from hippo.dream.contradiction import resolve_contradictions
from hippo.dream.edge_proposal import propose_edges
from hippo.dream.multi_head import expand_heads_for_eligible_bodies
from hippo.dream.review import review_new_atoms, review_rolling_slice
from hippo.lock import LockHeldError, acquire_lock, release_lock
from hippo.models.llm import LLMProto
from hippo.storage.dream_runs import complete_run, fail_run, start_run
from hippo.storage.multi_store import Scope, open_store


class DaemonProto(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class _CountingLLM:
    """Transparent LLMProto wrapper that counts generate_chat invocations."""

    def __init__(self, inner: LLMProto) -> None:
        self._inner = inner
        self.count = 0

    def generate_chat(self, *args: Any, **kwargs: Any) -> str:
        self.count += 1
        return self._inner.generate_chat(*args, **kwargs)


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
        for r in session_rows:
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
        _phase_delta("atomize", before)

        # Phase a2: review (gate-at-entry + rolling slice)
        before = counter.count
        n_review_archived += review_new_atoms(store=store, llm=counter, run_id=run_id)
        n_review_archived += review_rolling_slice(
            store=store, scope=scope.as_string(),
            llm=counter, slice_size=PRUNE_ROLLING_SLICE_SIZE,
        )
        _phase_delta("review", before)

        # Phase b: multi-head expansion
        before = counter.count
        n_heads += expand_heads_for_eligible_bodies(store=store, llm=counter, daemon=daemon)
        _phase_delta("multi_head", before)

        # Phase c-d: cluster + edge proposal
        before = counter.count
        n_edges += propose_edges(store=store, llm=counter)
        _phase_delta("edge_proposal", before)

        # Phase e: contradiction resolution
        before = counter.count
        n_contradictions += resolve_contradictions(store=store, llm=counter)
        _phase_delta("contradiction", before)

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
