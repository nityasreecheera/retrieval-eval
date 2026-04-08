"""
Dense vector retriever — semantic similarity via embeddings.

Uses any OpenAI-compatible embeddings API to encode documents and queries
into dense vectors, then ranks by cosine similarity.

Unlike BM25 and TF-IDF, this captures meaning: "layoffs" and "headcount
reduction" will be close in vector space.

Requires: pip install openai
          OPENAI_API_KEY environment variable (or pass api_key directly)
"""

import math
import os
from ..corpus import Chunk

DEFAULT_MODEL = "text-embedding-3-small"


class DenseRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "DenseRetriever requires an API key.\n"
                "Set OPENAI_API_KEY or pass api_key= to DenseRetriever."
            )

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=key)
        except ImportError:
            raise ImportError("openai package is required. Run: pip install openai")

        self.chunks = chunks
        self.model = model
        self.embeddings = self._embed([c.text for c in chunks])

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # Batch in groups of 100 to stay within API limits
        results = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]
            response = self._client.embeddings.create(model=self.model, input=batch)
            results.extend([item.embedding for item in response.data])
        return results

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        query_vec = self._embed([query])[0]
        scores = [
            (chunk, self._cosine(query_vec, self.embeddings[i]))
            for i, chunk in enumerate(self.chunks)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
