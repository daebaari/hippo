"""Integration test for mxbai-rerank-large-v1."""
from __future__ import annotations

import pytest

from hippo.models.reranker import Reranker


@pytest.fixture(scope="module")
def reranker() -> Reranker:
    return Reranker.load()


def test_reranker_returns_score_per_pair(reranker: Reranker) -> None:
    pairs = [
        ("kalshi taker fee", "Taker fee on Kalshi is 2c per contract at typical prices."),
        ("kalshi taker fee", "Tomorrow's weather forecast looks rainy."),
    ]
    scores = reranker.rerank(pairs)
    assert len(scores) == 2
    # First (relevant) pair should outscore the second
    assert scores[0] > scores[1]


def test_reranker_empty_input(reranker: Reranker) -> None:
    assert reranker.rerank([]) == []
