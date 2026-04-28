"""mxbai-rerank-large-v1 cross-encoder wrapper.

Cross-encoders take (query, document) pairs and produce a single relevance
score per pair. They're slower per-pair than bi-encoders but much sharper
on actual relevance. We use this as the precision step after vector search
+ graph expansion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from sentence_transformers import CrossEncoder

RERANKER_MODEL_ID = "mixedbread-ai/mxbai-rerank-large-v1"


@dataclass
class Reranker:
    model: CrossEncoder

    @staticmethod
    def load() -> Reranker:
        return Reranker(model=CrossEncoder(RERANKER_MODEL_ID))

    def rerank(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self.model.predict(cast(list[Any], pairs), convert_to_numpy=True)
        return [float(s) for s in scores.tolist()]
