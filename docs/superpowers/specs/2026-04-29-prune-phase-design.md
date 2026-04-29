# Prune Phase Design

**Date:** 2026-04-29
**Status:** approved
**Goal:** keep the active hippo memory corpus lean and fresh by detecting and soft-archiving outdated, noise, and redundant atoms inside the existing nightly heavy dream. No new launchd jobs, no separate cadence, no hard-delete.

## Why

Today the only pruning paths in hippo are:

1. `resolve_contradictions` — runs after `edge_proposal`, archives the loser of any LLM-confirmed `contradicts` edge.
2. Manual `bin/memory-archive <head_id> --reason ...`.

That misses three real failure modes that accumulate in the active corpus:

| Mode | What rots | Why current pipeline misses it |
|---|---|---|
| **A. Factually superseded** | Old body says "default backend is X" after the world moved on | Only caught when both atoms cluster *and* the LLM happens to flag them as `contradicts`. Drift over months gets missed entirely. |
| **C. Noise atoms** | Short procedural turns ("status", "again", "ok"), session-debug chatter | Atomize prompt tells LLM to skip them, but per `KNOWN_ISSUES.md` they leak through anyway. |
| **D. Redundant / mergeable** | Two bodies say the same fact from slightly different angles | No mechanism today. Both bodies sit active, both contribute heads to clusters, both grow next dream's N² edge step. |

Mode B (never-retrieved data debt) is **explicitly out of scope** — we don't want time/usage-based pruning that risks discarding rare-but-valuable atoms.

## Architecture

The feature lives entirely inside `run_heavy_dream_for_scope`. The new phase order:

1. **atomize** *(modified)* — atomize prompt now requires a `noise: true|false` field per atom with explicit examples; atoms with `noise=true` are skipped at insertion. Mode C, zero added LLM cost (same call).
2. **review** *(new phase)* — runs after atomize, before multi_head. Two passes inside one phase:
   - **Pass 1 — gate new atoms.** Each body inserted by atomize this run is checked against its top-K nearest active neighbors. Pairs above a similarity threshold get an LLM judgment ("merge", "supersede", "keep_both"). Loser is soft-archived. Handles modes A and D for newly-arrived content.
   - **Pass 2 — rolling slice sweep.** Picks the K oldest unreviewed active bodies, runs the same neighbor + judgment loop, stamps `last_reviewed_at`. Catches drift over time without a full N² sweep. The whole corpus eventually cycles through.
3. **multi_head** — unchanged.
4. **cluster + edge_proposal** — unchanged. Now operates on a leaner head set, smaller N² cost.
5. **contradiction resolution** — unchanged.
6. **cleanup** — unchanged.

**Total LLM cost added per dream:** O(new_atoms + slice_size) judgment calls. Bounded, predictable, doesn't grow with corpus size. The runtime payoff: subsequent dreams' edge_proposal step works on fewer active heads, so its N²-in-cluster-size cost shrinks.

**No GC / hard-delete.** Soft-archive already keeps the active memory lean — archived rows are filtered from retrieval and from `cluster_active_heads`. Disk grows monotonically and that's accepted; the audit trail is preserved indefinitely.

## Components

### New files

- `src/hippo/dream/review.py` (~150 LOC):
  - `review_new_atoms(*, store, llm, daemon, run_id) -> int` — pulls bodies inserted this run via `source = 'heavy-dream-run:{run_id}'`, runs gate-at-entry on each.
  - `review_rolling_slice(*, store, llm, daemon, slice_size) -> int` — picks oldest-unreviewed active bodies, runs the same loop.
  - `_review_body_against_neighbors(...)` — shared inner: cosine search via `head_embeddings`, filter to neighbors above the similarity threshold, dispatch pair judgment, archive loser, stamp `last_reviewed_at`.
  - `_judge_pair(llm, body_a, body_b)` — calls the new review prompt with `thinking_level="minimal"`. Returns `(decision, keeper_body_id)` where `decision ∈ {"merge", "supersede", "keep_both"}`.

- `src/hippo/dream/prompts/review.md` — single prompt taking two bodies, asking same-fact / supersession / keep-both. Required JSON output `{decision, keeper, reason}`. Per the project's "store lookup instructions, not values" rule, the prompt's examples teach behavior patterns rather than referencing specific tunable values.

- `schema/002_prune_metadata.sql` — idempotent migration:
  - `ALTER TABLE bodies ADD COLUMN last_reviewed_at INTEGER` (nullable).
  - `ALTER TABLE dream_runs ADD COLUMN bodies_archived_review INTEGER DEFAULT 0`.
  - `CREATE INDEX IF NOT EXISTS idx_bodies_review_queue ON bodies(last_reviewed_at) WHERE archived = 0`.

- `tests/test_review_phase.py` — see Testing.

### Modified files

- `src/hippo/dream/prompts/atomize.md` — add required `noise: true|false` field. Examples on both sides:
  - **Noise:** terse procedural turns ("status", "again", "i sent it"), acknowledgements ("ok", "thanks", "looks good"), debug-loop chatter ("try again", "doesn't work"), trivial confirmations, request interrupts.
  - **Durable:** decisions, preferences/constraints, project facts, reusable patterns, bug-fix learnings, spec requirements.
  - Tiebreaker: "when uncertain → noise=true".

- `src/hippo/dream/atomize.py` — skip atoms where `noise == true` at insertion. Default `noise` to `false` if the LLM omits the field (backward compat). Use `bool(value)` for permissive parsing.

- `src/hippo/dream/heavy.py` — insert `review_new_atoms` then `review_rolling_slice` between atomize and multi_head. Track `bodies_archived_review` and pass it to `complete_run`.

- `src/hippo/storage/bodies.py` — add:
  - `find_oldest_unreviewed_active(conn, scope, limit) -> list[BodyRecord]`
  - `update_last_reviewed_at(conn, body_id)` — stamps `last_reviewed_at = now`.

- `src/hippo/storage/dream_runs.py` — accept `bodies_archived_review` keyword in `complete_run`.

- `src/hippo/config.py` — new constants:
  - `PRUNE_SIMILARITY_THRESHOLD` — cosine cutoff for considering two bodies as merge/supersede candidates.
  - `PRUNE_NEAREST_K` — max neighbors per body to consider.
  - `PRUNE_ROLLING_SLICE_SIZE` — slice size for the rolling sweep.

  Per the project's memory rule, the values themselves live only in `config.py`; docs reference the constants by name.

## Data flow

### Atomize with noise filter

```
session captures → atomize prompt (now with noise field)
→ LLM returns [{title, body, heads, noise: bool}, …]
→ skip atoms where noise=true
→ remaining atoms inserted as today (body file, body row, heads, head_embeddings)
```

### Review Pass 1 — gate new atoms

```
query bodies WHERE source='heavy-dream-run:{run_id}' AND archived=0
for each new body B:
    candidates ← top-K nearest active bodies via head_embeddings cosine
                 (filter: cosine ≥ PRUNE_SIMILARITY_THRESHOLD,
                  exclude B itself, dedupe to body_id, limit PRUNE_NEAREST_K)
    if not candidates:
        update_last_reviewed_at(B); continue
    for each candidate C:
        decision, keeper ← _judge_pair(B, C)   # one LLM call
        if decision in {"merge", "supersede"}:
            loser  ← B.body_id if keeper == C.body_id else C.body_id
            winner ← keeper
            reason ← {"merge":"merged_into", "supersede":"superseded_by"}[decision] + f":{winner}"
            archive_body(loser, reason=reason, in_favor_of=winner)
            for h in active heads of loser: archive_head(h, reason=f"body_archived:{reason}")
            break    # one judgment per body per run
    update_last_reviewed_at(B)
return count_archived
```

### Review Pass 2 — rolling slice sweep

```
slice ← find_oldest_unreviewed_active(scope, PRUNE_ROLLING_SLICE_SIZE)
for each body B in slice:
    same loop as Pass 1
return count_archived
```

`find_oldest_unreviewed_active` orders `ORDER BY COALESCE(last_reviewed_at, 0) ASC`. Never-reviewed bodies (NULL) come first, then oldest reviews. Each run advances the rolling window by `slice_size`; with a corpus of N active bodies, the whole corpus cycles every `ceil(N / slice_size)` nights.

### Pair judgment

```
read body_a markdown file
read body_b markdown file
prompt ← render("review", a_id, a_updated, a_body, b_id, b_updated, b_body)
raw ← llm.generate_chat([{role: user, content: prompt}],
                         temperature=0.0, max_tokens=400,
                         thinking_level="minimal")
parse JSON {decision, keeper, reason}
validate: decision ∈ {"merge", "supersede", "keep_both"}
          keeper ∈ {a_id, b_id}    (or null when keep_both)
on parse / validation failure: return ("keep_both", None)
```

Same defensive pattern as `resolve_contradictions`: any failure means "do nothing", never raise.

### Graph integrity

| Stage | What happens | Graph state |
|---|---|---|
| Soft-archive (review) | `bodies.archived=1`, all its heads `archived=1`. Edges stay in table. | `cluster_active_heads` filters by `heads.archived=0`, so archived heads exit clustering. Retrieval already filters archived heads. No N² impact next run. |
| Indefinite retention | Archived rows + body markdown files remain on disk. | `archived_in_favor_of` stays valid (its target also stays on disk; column is informational, never traversed during retrieval). |

## Error handling

Same defensive philosophy as the rest of the dream pipeline: any per-item failure logs and continues. Phase-level exceptions propagate to `run_heavy_dream_for_scope`'s try/except, which calls `fail_run` and reraises.

| Failure | Behavior |
|---|---|
| Atomize LLM omits `noise` field | Default to `noise=false` (insert as today). |
| Atomize LLM sends `noise` as non-bool | `bool(value)` cast — handles `"true"`, `"false"`, `1`, `0`. |
| Review LLM returns invalid JSON | `_judge_pair` returns `("keep_both", None)`. No archive. |
| Review LLM returns unknown `decision` or unknown `keeper` body_id | Treat as `keep_both`. |
| Body markdown file missing on disk | Catch `FileNotFoundError`, skip the pair. |
| `head_embeddings` cosine query returns nothing for a body | Skip the body for this run; stamp `last_reviewed_at` so the slice still advances. |
| Schema migration on old DB | `ALTER TABLE … ADD COLUMN` and `CREATE INDEX … IF NOT EXISTS` are safe on existing DBs. New `last_reviewed_at` is NULL for all existing rows — they sort first in the rolling slice (NULL coalesces to 0), so old corpora get reviewed first when the feature ships. |
| Review phase exceeds reasonable wall-time | Bounded by `slice_size + new_atom_count`, each judgment has `max_tokens=400`. Worst case is comparable to one extra contradiction-resolution pass. |

**Recovery from a wrong archive:** soft-archive only — a wrongly-pruned atom can be inspected and recovered indefinitely via SQL. The recovery command itself is out of scope for this spec; flag in `KNOWN_ISSUES.md` that "review false-positive recovery" is a manual SQL job today.

**Lock semantics:** review runs inside the existing `.heavy-lock` acquired at the start of `run_heavy_dream_for_scope`. No new locks.

## Testing

Same patterns the codebase already uses: small SQLite DB in `tmp_path`, fake daemon/LLM with predictable outputs, no real models loaded.

### `tests/test_atomize.py` (additions)

- LLM returns mix of `noise=true` and `noise=false` atoms → only `noise=false` atoms inserted; counter reflects that.
- LLM omits `noise` field → atom is inserted (default false, backward compat).
- LLM returns `noise: "true"` (string) → coerced via `bool()`-equivalent parse, treated as truthy. Verifies no crash.
- LLM returns invalid JSON entirely → returns 0 (existing behavior unchanged).

### `tests/test_review_phase.py`

- **Gate-at-entry, no neighbors above threshold:** new body with embeddings far from everything else → no archive, `last_reviewed_at` stamped.
- **Gate-at-entry, neighbor above threshold, LLM says "merge":** loser archived, `archived_in_favor_of` set, all heads of loser archived.
- **Gate-at-entry, neighbor above threshold, LLM says "supersede":** same, with `archive_reason` reflecting the supersede decision.
- **Gate-at-entry, LLM says "keep_both":** no archive, `last_reviewed_at` stamped.
- **Gate-at-entry, LLM returns malformed JSON:** treated as `keep_both`. No exception escapes.
- **Gate-at-entry, body file missing on disk:** pair skipped, no exception.
- **Rolling slice, ordering:** mix of bodies with NULL / old / recent `last_reviewed_at` → SQL returns NULL bodies first, then oldest.
- **Rolling slice, slice_size cap:** corpus larger than slice_size → only slice_size bodies are judged.
- **Rolling slice, archived body excluded:** archived bodies never re-enter the slice.
- **Idempotence within a run:** running review twice in the same run doesn't re-judge the same body twice (second call's slice is empty because everything was just stamped).
- **Edge graph integrity after archive:** archive a body in review → `cluster_active_heads` no longer returns its heads, so subsequent edge proposal sees a smaller cluster.

### `tests/test_heavy_dream.py` (extensions)

- End-to-end heavy dream with the new phase ordering: stub atomize, stub LLM, stub embedder. Assert phase sequence: atomize → review_new → review_slice → multi_head → edge_proposal → contradiction → cleanup. Assert `dream_runs.bodies_archived_review` counter is correct.
- Schema migration applies cleanly to a DB pre-dating `last_reviewed_at`: load 001-only schema, run migration 002, verify column exists.

### `tests/test_storage_bodies.py` (extensions)

- `find_oldest_unreviewed_active(scope, limit)` — covers NULL-first ordering, scope filter, archived exclusion, limit.
- `update_last_reviewed_at(body_id)` — basic stamp.

### LLM-real test (gated on `RUN_LLM_TESTS=1`)

- Single integration test: load Qwen, atomize a tiny synthetic transcript with one obvious-noise turn and one durable turn, assert the noise turn is dropped and the durable one survives. Validates the prompt change in real conditions, doesn't run in CI.

### Coverage gaps (accepted)

- No automated test for "pruning over 30 simulated dream cycles produces stable corpus size" — empirical property, not a unit-test target.
- No fuzz tests on the review prompt — same approach as the rest of the dream prompts (real-LLM smoke + integration).

## Migration / rollout

- Schema migration 002 is idempotent and safe on existing DBs.
- On first dream run after deploy, the rolling slice processes never-reviewed bodies first (NULL `last_reviewed_at` sorts first). Existing corpora get a full sweep over `ceil(N / slice_size)` nights without any operator action.
- The atomize prompt change is backward compatible: missing `noise` field defaults to `false`. No data migration needed.
- No new launchd plist, no new bin entry point, no new slash command.
