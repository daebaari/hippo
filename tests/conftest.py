"""Shared pytest fixtures."""
from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlite_vec

from hippo.daemon.server import DaemonServer


@pytest.fixture
def temp_memory_dir(tmp_path: Path) -> Path:
    """Empty memory dir suitable for one store (global or project)."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "bodies").mkdir()
    return memory_dir


@pytest.fixture
def sqlite_conn(temp_memory_dir: Path) -> sqlite3.Connection:
    """Raw sqlite3 connection with sqlite-vec loaded."""
    conn = sqlite3.connect(temp_memory_dir / "memory.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.row_factory = sqlite3.Row
    return conn


@pytest.fixture(scope="session")
def daemon_socket(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Start the daemon on a background thread; yield socket path; tear down."""
    sock_path = tmp_path_factory.mktemp("daemon") / "memory.sock"
    server = DaemonServer.load()
    loop = asyncio.new_event_loop()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()

    serve_future = asyncio.run_coroutine_threadsafe(server.serve(sock_path), loop)

    # Wait for the listening socket to appear.
    for _ in range(50):
        if sock_path.exists():
            break
        time.sleep(0.1)
    else:
        raise TimeoutError("daemon never started")

    yield sock_path

    async def _shutdown() -> None:
        serve_future.cancel()
        try:
            await asyncio.wrap_future(serve_future)
        except (asyncio.CancelledError, Exception):
            pass

    asyncio.run_coroutine_threadsafe(_shutdown(), loop).result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()
