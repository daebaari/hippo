"""Daemon JSON-line protocol.

Each request/response is one JSON object terminated by a single \\n.
Requests have a 'kind' tag dispatching to the right handler.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class EmbedRequest:
    texts: list[str]
    kind: str = "embed"

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, "texts": self.texts}) + "\n"


@dataclass
class RerankRequest:
    pairs: list[tuple[str, str]]
    kind: str = "rerank"

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, "pairs": list(self.pairs)}) + "\n"


@dataclass
class PingRequest:
    kind: str = "ping"

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind}) + "\n"


@dataclass
class EmbedResponse:
    embeddings: list[list[float]]
    kind: str = "embed_response"


@dataclass
class RerankResponse:
    scores: list[float]
    kind: str = "rerank_response"


@dataclass
class PingResponse:
    ok: bool = True
    kind: str = "ping_response"


@dataclass
class ErrorResponse:
    message: str
    kind: str = "error"


Request = EmbedRequest | RerankRequest | PingRequest
Response = EmbedResponse | RerankResponse | PingResponse | ErrorResponse


def decode_request(line: str) -> Request:
    obj = json.loads(line)
    kind = obj.get("kind")
    if kind == "embed":
        return EmbedRequest(texts=list(obj["texts"]))
    if kind == "rerank":
        return RerankRequest(pairs=[(str(p[0]), str(p[1])) for p in obj["pairs"]])
    if kind == "ping":
        return PingRequest()
    raise ValueError(f"unknown request kind: {kind}")


def encode_response(resp: Response) -> str:
    return json.dumps(resp.__dict__) + "\n"
