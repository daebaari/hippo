# Known Issues

## Raw-turn retrieval not wired into the pipeline

**Symptom:** new captured turns aren't retrievable in fresh sessions until the
next heavy dream atomizes them. Saying "I prefer X" in one session does NOT
make X surface in the next session's `<memory>` block — only after heavy dream
runs (nightly 3am, or manual `bin/dream-heavy --force`).

**Root cause:** `src/hippo/retrieval/vector_search.py` queries only
`head_embeddings`. The Stop hook does embed every captured turn into
`turn_embeddings_vec` (per spec), but the retrieval pipeline ignores that
table. The original spec called for raw-turn retrieval at 0.6× weight; Plan 3
shipped without that step.

**Workaround:** run `bin/dream-heavy --force --project <name>` after a turn
you want immediately retrievable. Takes 5-15 minutes per run.

**Fix:** ~50-100 line addition to `vector_search_all_scopes` and the pipeline
to also query `turn_embeddings_vec`, project a synthetic head-shaped record
from each result (using `turn_embeddings.summary`), and apply 0.6× score
weighting so atoms still rank above raw turns. Worth a small dedicated plan.

## Edge-proposal scales N² within each cosine cluster

**Symptom:** heavy dream wall-time grows quadratically with cluster size and
gets worse over time. Observed runs on the kaleon project store:

| Run | Captures processed | New atoms | New edges | Wall time |
|---|---|---|---|---|
| Bootstrap (~30 files) | 31 files | 55 | 596 | **~110 min** |
| Heavy dream #1 | 4 captures | 4 | 5 | **~26 min** |
| Heavy dream #2 (in progress) | 15 captures | 7+ | ~3 in 16 min | **30-50 min projected** |

The dominant cost is the within-cluster LLM judgment loop. A cluster of N
heads expands to N(N−1)/2 pairs; mxbai-embed-large produces fairly large
clusters (the kaleon corpus had a 22-head cluster → 231 pairs alone).

The behavior compounds: each successful dream grows existing clusters
slightly, increasing the next run's pair count. And most pair evaluations
return `"none"` (the heads weren't actually meaningfully related despite
being similar enough to cluster) — those calls still cost full LLM time
without producing edges.

**Root cause:** `src/hippo/dream/edge_proposal.py` calls the LLM on every
unique pair within each `cluster_active_heads` cluster, no cap. The cluster
algorithm (single-link, cosine ≥ 0.7) is also generous about what counts as
"in the same cluster."

**Workaround:** none today; runs run as long as they need. Don't trigger
manual heavy dreams during work hours unless you can spare ~30 min.

**Fix options:**

1. **Per-cluster pair cap.** For clusters > K heads, only propose edges between
   the K most-central nodes by current degree (or a random sample). 30-50
   lines + tests. Bounds runtime to roughly K² × cluster_count regardless
   of corpus growth.
2. **Skip pairs with low cosine** even within the same cluster. Cluster
   membership is single-link transitive; two members of the same cluster can
   be far apart. Add a per-pair similarity gate (e.g., > 0.75) before LLM
   call. 10-line change, much higher hit rate.
3. **Cheap pre-classifier**. Before the expensive LLM call, run a fast prompt
   on a smaller model (or even a string-distance heuristic) to short-circuit
   "obviously unrelated" pairs. More invasive.

(1) and (2) compose naturally and would together likely cut wall time 5-10×.

## Atomize-prompt noise leakage

**Symptom:** occasional "noise atoms" land in the store from session-debug
turns ("status", "again", "i sent it", etc.) despite the atomize prompt
explicitly telling the LLM to skip in-the-moment chatter.

**Root cause:** LLM doesn't always follow the skip instruction; some short
non-durable turns get atomized anyway.

**Workaround:** soft-archive obvious noise atoms when spotted:

```bash
bin/memory-archive <head_id> --reason "atomize-noise"
```

**Fix:** tighten the atomize prompt with stronger negative examples, OR add a
post-atomize "dream-noise filter" pass that flags atoms with short bodies +
no semantic structure. Future improvement.

## Lock leak between `acquire_lock` and `start_run`

**Symptom:** if `start_run()` raises (e.g., DB I/O error) immediately after
the heavy or light dream takes its lock, the lock is not released by the
finally clause (which only covers `release_lock` after `complete_run`/`fail_run`).
The next dream run is blocked until the SessionStart cleanup hook
(`cleanup-stale-consolidate-locks.sh` in `~/.claude/`) clears it via dead-PID
or 1h-mtime check.

**Affects:** `src/hippo/dream/heavy.py` (`run_heavy_dream_for_scope`),
`src/hippo/dream/light.py` (`run_light_dream`).

**Workaround:** the SessionStart cleanup hook self-heals at next session
start; impact is bounded.

**Fix:** wider try/finally around `start_run` so the lock is released even on
audit-log failure. Trivial change, deferred for cosmetic-fix scope.

## install.sh / uninstall.sh end-to-end smoke deferred

The Plan 8 smoke (install.sh → cross-session prompt → uninstall.sh) was
deliberately not run during release because the development machine was
already in the post-install state. Scripts were validated by syntax check
and byte-match to spec, but a fresh-machine install has not been exercised
end-to-end. First real test will happen when this is installed on a clean
environment; expect to find and fix issues there.

