"""Tests for file-based lock with stale-lock recovery."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from hippo.lock import LockHeld, acquire_lock, release_lock, sweep_stale_locks


def test_acquire_then_release(tmp_path: Path) -> None:
    lock_path = tmp_path / ".test-lock"
    handle = acquire_lock(lock_path)
    assert lock_path.exists()
    release_lock(handle)
    assert not lock_path.exists()


def test_acquire_when_held_raises(tmp_path: Path) -> None:
    lock_path = tmp_path / ".test-lock"
    handle = acquire_lock(lock_path)
    try:
        with pytest.raises(LockHeld):
            acquire_lock(lock_path)
    finally:
        release_lock(handle)


def test_stale_lock_with_dead_pid_swept(tmp_path: Path) -> None:
    lock_path = tmp_path / ".test-lock"
    # Write a lock file with a PID that does not exist
    lock_path.write_text("999999")  # almost certainly dead
    swept = sweep_stale_locks([lock_path])
    assert lock_path in swept
    assert not lock_path.exists()


def test_old_lock_with_live_pid_still_swept(tmp_path: Path) -> None:
    lock_path = tmp_path / ".test-lock"
    # Spawn a long-running process whose PID is alive
    proc = subprocess.Popen(["sleep", "60"])
    try:
        lock_path.write_text(str(proc.pid))
        # Backdate mtime to simulate old lock
        old = time.time() - 7200  # 2 hours ago
        os.utime(lock_path, (old, old))
        swept = sweep_stale_locks([lock_path])
        assert lock_path in swept
        assert not lock_path.exists()
    finally:
        proc.terminate()
        proc.wait()


def test_recent_lock_with_live_pid_not_swept(tmp_path: Path) -> None:
    lock_path = tmp_path / ".test-lock"
    proc = subprocess.Popen(["sleep", "60"])
    try:
        lock_path.write_text(str(proc.pid))
        # mtime is fresh by default
        swept = sweep_stale_locks([lock_path])
        assert swept == []
        assert lock_path.exists()
    finally:
        proc.terminate()
        proc.wait()
        if lock_path.exists():
            lock_path.unlink()
