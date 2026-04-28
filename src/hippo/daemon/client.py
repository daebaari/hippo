"""Sync daemon client used by hooks. No asyncio in callers."""
from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hippo.daemon.protocol import (
    EmbedRequest,
    PingRequest,
    RerankRequest,
)


@dataclass
class DaemonClient:
    socket_path: Path

    def _round_trip(self, line: str) -> dict[str, Any]:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(str(self.socket_path))
            s.sendall(line.encode())
            buf = bytearray()
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                if b"\n" in buf:
                    break
        finally:
            s.close()
        result: dict[str, Any] = json.loads(buf.decode())
        return result

    def ping(self) -> bool:
        out = self._round_trip(PingRequest().to_json())
        return out.get("kind") == "ping_response" and out.get("ok") is True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out = self._round_trip(EmbedRequest(texts=texts).to_json())
        if out.get("kind") == "error":
            raise RuntimeError(f"daemon error: {out.get('message')}")
        return [list(map(float, vec)) for vec in out["embeddings"]]

    def rerank(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        out = self._round_trip(RerankRequest(pairs=pairs).to_json())
        if out.get("kind") == "error":
            raise RuntimeError(f"daemon error: {out.get('message')}")
        return [float(s) for s in out["scores"]]
