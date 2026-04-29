"""Heavy dream orchestrator. Runs nightly via launchd or via /dream."""
from __future__ import annotations

from typing import Protocol

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


def run_heavy_dream_all_scopes(
    *, scopes: list[Scope], llm: LLMProto, daemon: DaemonProto
) -> dict[str, object]:
    out: dict[str, object] = {}
    for scope in scopes:
        out[scope.as_string()] = run_heavy_dream_for_scope(scope=scope, llm=llm, daemon=daemon)
    return out
