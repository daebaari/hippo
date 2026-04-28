"""Test the sync daemon client by reusing the in-process daemon fixture."""
from __future__ import annotations

from pathlib import Path

from hippo.config import EMBEDDING_DIM
from hippo.daemon.client import DaemonClient


def test_client_ping(daemon_socket: Path) -> None:
    c = DaemonClient(socket_path=daemon_socket)
    assert c.ping() is True


def test_client_embed(daemon_socket: Path) -> None:
    c = DaemonClient(socket_path=daemon_socket)
    vecs = c.embed(["hello", "world"])
    assert len(vecs) == 2
    assert len(vecs[0]) == EMBEDDING_DIM


def test_client_rerank(daemon_socket: Path) -> None:
    c = DaemonClient(socket_path=daemon_socket)
    scores = c.rerank([("apples", "fruit"), ("apples", "asphalt")])
    assert scores[0] > scores[1]
