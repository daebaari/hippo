"""Integration test for the daemon (real models, real socket).

The fixture runs the asyncio loop in a background daemon thread so the
synchronous ``_send()`` helper on the main thread can interact with the
server while the loop pumps events. Without that, blocking socket I/O
on the main thread would starve the loop and requests would hang.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path

from hippo.config import EMBEDDING_DIM
from hippo.daemon.protocol import (
    EmbedRequest,
    PingRequest,
    RerankRequest,
)


def _send(sock_path: Path, line: str) -> str:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(str(sock_path))
    s.sendall(line.encode())
    chunks: list[bytes] = []
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        if chunks[-1].endswith(b"\n"):
            break
    s.close()
    return b"".join(chunks).decode()


def test_ping(daemon_socket: Path) -> None:
    out = _send(daemon_socket, PingRequest().to_json())
    assert '"kind": "ping_response"' in out
    assert '"ok": true' in out


def test_embed_returns_correct_shape(daemon_socket: Path) -> None:
    out = _send(daemon_socket, EmbedRequest(texts=["hello", "world"]).to_json())
    data = json.loads(out)
    assert data["kind"] == "embed_response"
    assert len(data["embeddings"]) == 2
    assert len(data["embeddings"][0]) == EMBEDDING_DIM


def test_rerank_returns_scores(daemon_socket: Path) -> None:
    out = _send(
        daemon_socket,
        RerankRequest(pairs=[("apples", "fruit"), ("apples", "asphalt")]).to_json(),
    )
    data = json.loads(out)
    assert data["kind"] == "rerank_response"
    assert len(data["scores"]) == 2
    assert data["scores"][0] > data["scores"][1]
