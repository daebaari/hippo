"""File-based lock with PID liveness and mtime-age recovery.

Pattern mirrors ~/.claude/hooks/cleanup-stale-consolidate-locks.sh:
- Lock file contains the holding PID
- Lock is "stale" if PID is dead OR mtime > STALE_LOCK_AGE_SECONDS
- Sweeps clear stale locks before acquisition
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from hippo.config import STALE_LOCK_AGE_SECONDS


class LockHeldError(Exception):
    """Raised when attempting to acquire a lock already held by a live, fresh PID."""


@dataclass(frozen=True)
class LockHandle:
    path: Path
    pid: int


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours; treat as alive
    return True


def _is_stale(lock_path: Path) -> bool:
    """Lock is stale if PID is dead or mtime is older than the threshold."""
    try:
        content = lock_path.read_text().strip()
        pid = int(content)
    except (FileNotFoundError, ValueError):
        return True
    if not _pid_alive(pid):
        return True
    age = time.time() - lock_path.stat().st_mtime
    return age > STALE_LOCK_AGE_SECONDS


def acquire_lock(lock_path: Path) -> LockHandle:
    """Acquire a lock, sweeping stale locks first. Raises LockHeldError if a fresh
    live-PID lock blocks us."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        if _is_stale(lock_path):
            lock_path.unlink(missing_ok=True)
        else:
            raise LockHeldError(f"lock at {lock_path} held by live PID")
    pid = os.getpid()
    lock_path.write_text(str(pid))
    return LockHandle(path=lock_path, pid=pid)


def release_lock(handle: LockHandle) -> None:
    """Release a lock if still held by us."""
    try:
        content = handle.path.read_text().strip()
        if content == str(handle.pid):
            handle.path.unlink()
    except FileNotFoundError:
        pass


def sweep_stale_locks(lock_paths: list[Path]) -> list[Path]:
    """Clear stale locks. Returns the list of paths that were swept."""
    swept: list[Path] = []
    for lock in lock_paths:
        if lock.exists() and _is_stale(lock):
            lock.unlink(missing_ok=True)
            swept.append(lock)
    return swept
