"""Tests for daemon JSON-line protocol."""
from __future__ import annotations

from hippo.daemon.protocol import (
    EmbedRequest,
    EmbedResponse,
    PingRequest,
    RerankRequest,
    decode_request,
    encode_response,
)


def test_embed_request_roundtrip() -> None:
    req = EmbedRequest(texts=["hello", "world"])
    encoded = req.to_json()
    decoded = decode_request(encoded)
    assert isinstance(decoded, EmbedRequest)
    assert decoded.texts == ["hello", "world"]


def test_rerank_request_roundtrip() -> None:
    req = RerankRequest(pairs=[("query", "doc")])
    encoded = req.to_json()
    decoded = decode_request(encoded)
    assert isinstance(decoded, RerankRequest)
    assert decoded.pairs == [("query", "doc")]


def test_ping_request_roundtrip() -> None:
    req = PingRequest()
    encoded = req.to_json()
    decoded = decode_request(encoded)
    assert isinstance(decoded, PingRequest)


def test_responses_encode_to_json_line() -> None:
    resp = EmbedResponse(embeddings=[[0.1, 0.2]])
    line = encode_response(resp)
    assert line.endswith("\n")
    assert '"kind": "embed_response"' in line
