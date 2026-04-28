"""Integration test for mxbai-embed-large-v1 via MLX/sentence-transformers."""
from __future__ import annotations

import pytest

from hippo.config import EMBEDDING_DIM
from hippo.models.embedder import Embedder


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Loads the model once for the whole module — first call downloads weights."""
    return Embedder.load()


def test_embedding_has_expected_dim(embedder: Embedder) -> None:
    vec = embedder.embed("hello world")
    assert len(vec) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in vec[:5])


def test_similar_texts_have_higher_cosine(embedder: Embedder) -> None:
    import math
    a = embedder.embed("cats sleeping in the sun")
    b = embedder.embed("kittens napping in sunlight")
    c = embedder.embed("quarterly revenue forecast")

    def cos(x: list[float], y: list[float]) -> float:
        dot = sum(xi * yi for xi, yi in zip(x, y, strict=True))
        nx = math.sqrt(sum(xi * xi for xi in x))
        ny = math.sqrt(sum(yi * yi for yi in y))
        return dot / (nx * ny)

    assert cos(a, b) > cos(a, c)


def test_batch_embed(embedder: Embedder) -> None:
    vecs = embedder.embed_batch(["one", "two", "three"])
    assert len(vecs) == 3
    assert all(len(v) == EMBEDDING_DIM for v in vecs)
