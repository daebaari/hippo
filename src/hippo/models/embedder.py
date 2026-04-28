"""mxbai-embed-large-v1 wrapper for similarity search.

Uses sentence-transformers under the hood (which itself wraps the HF model).
On Apple Silicon, this uses MPS via PyTorch by default. We keep this layer
thin: load model once, expose embed() and embed_batch().
"""
from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

EMBEDDER_MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"


@dataclass
class Embedder:
    model: SentenceTransformer

    @staticmethod
    def load() -> "Embedder":
        model = SentenceTransformer(EMBEDDER_MODEL_ID)
        return Embedder(model=model)

    def embed(self, text: str) -> list[float]:
        v = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return [float(x) for x in v.tolist()]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        arr = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [[float(x) for x in row.tolist()] for row in arr]
