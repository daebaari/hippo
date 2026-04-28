# Operations Guide

A short reference for keeping a running Hippo install healthy. Assumes you ran
`scripts/install.sh` and the launchd agents are loaded.

Throughout this doc, `<user>` is the local macOS username (the value of
`whoami`). Plist labels include it so multiple users on the same machine don't
collide.

## 1. Daemon health

The memory daemon runs as a launchd user agent and stays up across reboots.

```bash
launchctl list | grep memory-daemon
```

A line means it's loaded. The PID column shows whether it's currently running
(a number) or only resident (`-`). To force a restart:

```bash
launchctl kickstart -k gui/$UID/com.<user>.memory-daemon
```

If it won't load at all, check the launchd log path in
`launchd/memory-daemon.plist.template` (rendered into
`~/Library/LaunchAgents/`).

## 2. Manual heavy dream

`bin/dream-heavy` is what the launchd agent runs nightly. It checks AC power
and skips if the laptop is on battery. To force a run regardless:

```bash
~/code/hippo/bin/dream-heavy --force
```

Useful right after a long working session, after `bin/dream-bootstrap`, or
when iterating on dream phases.

## 3. Inspect storage

The CLI prints body / head / edge counts and recent dream runs per scope:

```bash
uv run memory-stats                     # global
uv run memory-stats --project kaleon    # project
```

For ad-hoc queries, open the SQLite file directly:

```bash
sqlite3 ~/.claude/memory/memory.db
```

(Project stores live at `~/.claude/projects/*/memory/memory.db`.)

## 4. Clear stuck locks

Heavy dreams take an exclusive lock so two of them can't run concurrently. The
daemon sweeps stale locks at SessionStart, so this almost never needs manual
intervention. If you do need to force-clear:

```bash
rm ~/.claude/memory/.heavy-lock
```

Only do this if you're certain no heavy dream is actually running.

## 5. Re-embed all heads

If you change the embedder model, existing head embeddings need to be
regenerated. A `bin/memory-reembed` tool will land in a follow-up plan; until
then, the safe path is to drop the `head_vec` table and let the daemon rebuild
on next dream.

## 6. Backup

Memory data lives entirely under:

- `~/.claude/memory/`              (global scope)
- `~/.claude/projects/*/memory/`   (per-project scopes)

Time Machine on those paths is sufficient for most users. A `bin/memory-export`
tool for portable snapshots is planned as a follow-up.

The DBs are SQLite — `cp` while the daemon is idle is also fine. For a hot
copy, use `sqlite3 ... ".backup"` instead of a raw cp.

## 7. Logs

When something looks wrong, the launchd agents write to:

- `~/.claude/debug/memory-daemon.log`
- `~/.claude/debug/dream-heavy.log`

Hook output is captured by Claude Code itself. If a hook is misbehaving, also
check:

- `~/.claude/debug/userprompt-hook.log`
- `~/.claude/debug/stop-hook.log`

(These are written by the hook scripts when stderr is redirected by the
caller. If they're absent, the hook is exiting silently — the
`bin/userprompt-retrieve` and `bin/stop-capture` entry points trap all
exceptions to stderr so Claude Code surfaces them.)
