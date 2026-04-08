"""
Dense vector retriever — semantic similarity via local embeddings.

Uses fastembed to run a small embedding model locally with no API key required.
Unlike BM25 and TF-IDF, this captures meaning: "layoffs" and "headcount
reduction" will be close in vector space.

Requires: pip install fastembed
"""

import math
from ..corpus import Chunk

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], model_name: str = DEFAULT_MODEL):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for DenseRetriever.\n"
                "Install it with: pip install fastembed"
            ) from exc

        self.chunks = chunks
        self.model = TextEmbedding(model_name=model_name)
        texts = [c.text for c in chunks]
        self.embeddings = list(self.model.embed(texts))

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        query_vec = list(self.model.embed([query]))[0]
        scores = [
            (chunk, self._cosine(query_vec, self.embeddings[i]))
            for i, chunk in enumerate(self.chunks)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _cosine(self, a, b) -> float:
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        norm_a = math.sqrt(sum(float(x) ** 2 for x in a))
        norm_b = math.sqrt(sum(float(x) ** 2 for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
