"""Cleanup: mark captures processed + delete their turn_embeddings."""
from __future__ import annotations

from hippo.storage.capture import mark_captures_processed
from hippo.storage.multi_store import Store
from hippo.storage.turn_embeddings import delete_turn_embeddings_for_captures


def finalize_processed_captures(
    *, store: Store, queue_ids: list[int], run_id: int
) -> None:
    """Mark captures as processed and delete their associated turn embeddings.

    After atomize succeeds for a session, delete turn_embeddings rows for those
    captures (atoms now cover the same content; raw-turn redundancy goes away).
    Mark capture_queue rows as processed.

    Args:
        store: The storage store containing the database connection.
        queue_ids: List of capture queue IDs to finalize.
        run_id: The run ID to record in processed_by_run.
    """
    if not queue_ids:
        return
    mark_captures_processed(store.conn, queue_ids, run_id=run_id)
    delete_turn_embeddings_for_captures(store.conn, queue_ids)
