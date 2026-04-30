# Dream-Heavy Progress Tracking Design

**Date:** 2026-04-30
**Status:** draft
**Goal:** make the long-running `dream-heavy` process observable while it runs — both for a human watching the terminal and for a separate process (Claude session, status script, future UI) that wants to ask "how far along is the dream?"

## Why

Today, `dream-heavy` is a black box from start to finish:
- Stdout is empty until the run completes, then prints one final JSON blob with phase totals (added in commit `e20731a`).
- Stderr carries Python tracebacks but no progress.
- The DB commits intermediate state (`bodies`, `edges`, archived rows) but there is no row keyed to the *active* run that says "I'm in phase X, item Y of Z."

For runs that finish in seconds, this is fine. For heavy dreams with a backlog of captures — see the kaleon scope's 90+ unprocessed captures — the edge-proposal phase alone can run 30–60 minutes inside the 105-head cluster. Two things break:
1. **Watching humans cannot tell if the process is alive vs hung.** CPU sits near zero (LLM-bound), the output file stays empty, and the only signal is `ps`.
2. **Polling consumers** (a sibling Claude session, a status command) have to reverse-engineer progress from raw `edges` and `bodies` deltas — which only roughly correlate with phase completion (see KNOWN_ISSUES — pairs that return `none` leave no DB trace).

The fix is small: emit progress to one durable place, then surface it through stderr (live) and a `dream-status` CLI (poll).

## Architecture

The feature lives in three places:

1. **`dream_runs` table** — gain five nullable columns describing the active phase and its progress. Heavy.py is the sole writer per run (the existing per-scope lock guarantees it).
2. **`heavy.py`** — wraps each phase in a context that writes phase entry/exit, and threads a throttled progress callback into the long phases (`edge_proposal` is the only N²-heavy one; the others are O(items)). The same callback writes the DB row and emits a stderr line.
3. **New CLI: `bin/dream-status`** — reads the most-recent `dream_runs` row across all scope DBs and prints a one-line summary (or watches it).

No new tables, no new files outside the new CLI, no new lock files, no new external dependencies.

## Data model

Add to `dream_runs` (all nullable, all idempotent):

| Column | Type | Meaning |
|---|---|---|
| `current_phase` | TEXT | One of: `atomize`, `review`, `multi_head`, `edge_proposal`, `contradiction`, `cleanup`. NULL until the first phase begins; left at the last phase's value after `complete_run` flips `status='completed'` (the CLI uses both columns to decide whether to display "running" or "done"). |
| `phase_done` | INTEGER | Items completed in the current phase. Reset to 0 at each phase entry. |
| `phase_total` | INTEGER | Total items the current phase will process. Computed at phase entry. |
| `phase_started_at` | INTEGER | Unix epoch when the current phase began. |
| `last_progress_at` | INTEGER | Unix epoch of the last progress write. Updated at most once per ~5s during a phase. |

Migration follows the existing `schema/NNN_*.sql` versioned pattern (see `schema/002_prune_metadata.sql` and `src/hippo/storage/migrations.py`). New file `schema/003_dream_progress.sql` adds the five columns via plain `ALTER TABLE`; the migration runner's `schema_versions` tracker prevents re-application, so no inspection-based guard is needed.

## Write path

In `heavy.py`, replace the bare phase blocks with a `phase_context(name, total)` helper. On enter it `UPDATE dream_runs SET current_phase=?, phase_done=0, phase_total=?, phase_started_at=now, last_progress_at=now WHERE run_id=?`, plus emits a stderr `phase=<name> total=<total>` line. On exit it emits a `phase=<name> done=N/N (100%) elapsed=Xs` line. (No row clear on exit — the next phase entry overwrites.)

Each phase function gains an optional `progress_cb: Callable[[int], None] | None` argument. The callback is called with the running `done` count after each unit of work:

- `atomize_session`: per session processed (small total, no throttle needed).
- `review_new_atoms` + `review_rolling_slice`: per body reviewed.
- `expand_heads_for_eligible_bodies`: per body expanded.
- `propose_edges`: **per pair iterated** — including pairs skipped because an edge already exists (so the denominator stays meaningful).
- `resolve_contradictions`: per pair judged.

The callback wrapper inside `heavy.py` is throttled: it does an `UPDATE` and a stderr emit at most once every 5 seconds **or** every 100 calls, whichever comes first. The unthrottled fast path is a single integer increment and an `if` — negligible overhead even inside the hot edge-proposal loop.

For `edge_proposal`, the upfront pair total is `Σ C(|cluster|, 2)` for all clusters with `len > 1`. `cluster_active_heads()` is already called once at the top of `propose_edges` — compute the pair total in the same scope and pass it (alongside the cluster list) to the progress callback so the denominator is set before any pair iteration begins.

## Stderr format

One human-readable line per phase boundary and per progress tick. Single timestamp prefix, key=value pairs, fixed-width phase column for grep-friendliness.

```
[12:00:01] phase=atomize          total=1
[12:00:03] phase=atomize          done=1/1     (100%) elapsed=2s
[12:00:03] phase=review           total=16
[12:00:30] phase=review           done=16/16   (100%) elapsed=27s
[12:00:30] phase=edge_proposal    total=5765
[12:00:35] phase=edge_proposal    done=87/5765 (1.5%) rate=9.0/s eta=10m
[12:00:40] phase=edge_proposal    done=178/5765 (3.1%) rate=9.1/s eta=10m
...
[12:14:30] phase=edge_proposal    done=5765/5765 (100%) elapsed=14m
[12:14:32] phase=contradiction    total=8
[12:14:35] phase=contradiction    done=8/8     (100%) elapsed=3s
[12:14:35] phase=cleanup          (instant)
[12:14:35] run                    completed elapsed=14m35s atoms=+16 edges=+312 archived=+4
```

This is stderr (the existing JSON summary keeps stdout clean for shell-piping).

## ETA computation

Rolling rate over the last 60 seconds, not cumulative. The first cluster encountered (often the giant 105-head one) is the slow part of the run; cumulative averaging would underestimate speed by the end and produce ETAs that drift up.

Implementation: keep a `(timestamp, done)` snapshot from approximately 60s ago. On each progress tick, compute `rate = (now_done - then_done) / max(now_time - then_time, 1)`; `eta = (total - now_done) / rate`. Roll the snapshot forward when the recorded one is older than 60s. Round ETA to the nearest minute (or "<1m" / ">99m"); display rate to one decimal.

If `rate == 0` (no progress in window), display `eta=?`.

## `dream-status` CLI

New `src/hippo/cli/dream_status.py` + thin `bin/dream-status` wrapper (matches existing pattern).

```
$ ~/code/hippo/bin/dream-status
running: kaleon run_id=8 phase=edge_proposal 2341/5765 (40.6%) rate=0.9/s eta=63m elapsed=44m
```

Behavior:
- Default: scan all scope DBs (global + every project under `~/.claude/projects/<name>/memory`), find the most recent row with `status='running'`, print one line. If none running, print the most recent `completed` or `failed` row.
- `--scope <name>` — restrict to one scope.
- `--watch` — re-print every 5s until interrupted (uses the same row, no extra writes).
- `--json` — machine-readable single line for scripting.

Reads only — never writes. No lock contention with the dream itself.

## Testing

- `tests/test_dream_progress.py`
  - Throttle: with a fake clock, callback fired 1000× in < 5s produces ≤ 11 emits (1 entry + ≤ 10 ticks).
  - ETA math: known done/total/elapsed yields expected eta string; zero rate yields `?`.
  - End-to-end: run a heavy dream against a tiny in-memory fixture, assert phase rows in `dream_runs` show the right transitions.
- `tests/test_dream_status_cli.py`
  - Output format with a running row (default + `--json`).
  - Fallback to most-recent completed row when nothing is running.
  - `--scope` filter.

Migration test: open a store on a DB without the new columns, assert columns appear; open it again, assert no error (idempotent).

## Files touched

- `schema/003_dream_progress.sql` — versioned migration adding the five columns.
- `src/hippo/storage/dream_runs.py` — `start_phase()`, `update_progress()`, getters used by the CLI; extend `DreamRunRecord` with the new columns.
- `src/hippo/dream/heavy.py` — `phase_context` manager + throttled callback wiring.
- `src/hippo/dream/edge_proposal.py` — accept `progress_cb`, call per pair.
- `src/hippo/dream/atomize.py`, `review.py`, `multi_head.py`, `contradiction.py` — accept `progress_cb`, call per item.
- `src/hippo/cli/dream_status.py` — new CLI.
- `bin/dream-status` — new wrapper script.
- `tests/test_dream_progress.py`, `tests/test_dream_status_cli.py` — new.

## Risks and non-goals

- **Concurrent writers within a scope:** dreams already hold `HEAVY_LOCK_FILENAME`, so only one writer per run.
- **Concurrent dreams across scopes:** safe — each writes its own scope's DB.
- **Throttle misses fast phases:** acceptable. A phase that takes < 5s shows up as one entry + one exit line; the user doesn't care about granular progress at that scale.
- **Stderr noise during nightly launchd runs:** launchd already captures stderr to a log file; the new lines are short and useful for postmortem.
- **Not in scope:** progress for the *light* dream (`light.py` is fast enough), tqdm-style bars (single-line key=value is sufficient and pipes/greps cleanly), push notifications, web dashboards, sub-cluster progress detail.
