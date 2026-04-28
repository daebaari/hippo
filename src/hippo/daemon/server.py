"""asyncio Unix-socket daemon holding the embedder + reranker resident.

Single-threaded by design — model inference happens inline on the event
loop. For our use case (one request at a time from a hook) this is fine
and avoids GIL/concurrency complexity.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from hippo.daemon.protocol import (
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    PingRequest,
    PingResponse,
    RerankRequest,
    RerankResponse,
    Response,
    decode_request,
    encode_response,
)
from hippo.models.embedder import Embedder
from hippo.models.reranker import Reranker


@dataclass
class DaemonServer:
    embedder: Embedder
    reranker: Reranker

    @staticmethod
    def load() -> DaemonServer:
        return DaemonServer(embedder=Embedder.load(), reranker=Reranker.load())

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            try:
                line = await reader.readline()
                if not line:
                    return
                req = decode_request(line.decode())
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                writer.write(encode_response(ErrorResponse(message=str(e))).encode())
                await writer.drain()
                return

            resp: Response
            try:
                if isinstance(req, PingRequest):
                    resp = PingResponse()
                elif isinstance(req, EmbedRequest):
                    vecs = self.embedder.embed_batch(req.texts)
                    resp = EmbedResponse(embeddings=vecs)
                elif isinstance(req, RerankRequest):
                    scores = self.reranker.rerank(req.pairs)
                    resp = RerankResponse(scores=scores)
                else:
                    resp = ErrorResponse(message=f"unhandled request: {req}")
            # Operational boundary: convert model failures to wire-protocol errors.
            except Exception as e:
                resp = ErrorResponse(message=f"{type(e).__name__}: {e}")

            writer.write(encode_response(resp).encode())
            await writer.drain()
        finally:
            writer.close()

    async def serve(self, sock_path: Path) -> None:
        if sock_path.exists():
            sock_path.unlink()
        sock_path.parent.mkdir(parents=True, exist_ok=True)
        server = await asyncio.start_unix_server(self._handle, path=str(sock_path))
        try:
            async with server:
                await server.serve_forever()
        finally:
            try:
                sock_path.unlink()
            except FileNotFoundError:
                pass
