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

**Symptom:** heavy dream wall-time grows quadratically with cluster size.
A 22-head cluster expands to 231 LLM-judged pairs per run. The kaleon
bootstrap of ~30 files took ~110 minutes; subsequent dreams over 4 captures
that join existing big clusters can take 10-30 min.

**Root cause:** `src/hippo/dream/edge_proposal.py` calls the LLM on every
unique pair within each `cluster_active_heads` cluster, no cap.

**Workaround:** none today; runs run as long as they need.

**Fix:** add a per-cluster pair cap (e.g., for clusters >K heads, only propose
edges between the top-K most-central nodes by current degree, or a random
sample). Probably 30-50 lines + tests.

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

## Gemini backend exists as an unmerged stash

A Plan 7 implementer overreach landed a partial dual-backend implementation
(LocalLLM + GeminiLLM via `HIPPO_LLM_BACKEND` env var, with `thinking_level`
threaded through `LLMProto`) without a design or review. Code currently sits
in a git stash:

```bash
git stash list
# stash@{0}: On main: gemini-backend-wip-from-plan7-overreach
```

Don't `git stash apply` blindly — the `thinking_level` cross-cut into the
LLM Protocol is a contract change that deserves a real spec. Future Plan 9
should either rewrite from scratch or apply selectively after a proper
design review.
