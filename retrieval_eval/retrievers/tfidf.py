"""
TF-IDF vector retriever — semantic similarity via cosine distance.

Represents documents and queries as TF-IDF vectors and ranks by cosine
similarity. Better than BM25 at synonym matching and paraphrased queries.
"""

import math
import re
from collections import Counter
from ..corpus import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())


class TFIDFRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._build_index()

    def _build_index(self):
        N = len(self.chunks)
        self.tokenized = [_tokenize(c.text) for c in self.chunks]

        # Build vocabulary and document frequency
        df = Counter()
        for tokens in self.tokenized:
            for term in set(tokens):
                df[term] += 1

        self.vocab = sorted(df.keys())
        self.term_to_idx = {t: i for i, t in enumerate(self.vocab)}
        self.N = N

        # IDF for each term
        self.idf = {
            term: math.log((N + 1) / (df[term] + 1)) + 1
            for term in self.vocab
        }

        # Precompute TF-IDF vectors for all documents
        self.doc_vectors = [self._vectorize(tokens) for tokens in self.tokenized]

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        total = max(len(tokens), 1)
        vec = {}
        for term, count in tf.items():
            if term in self.idf:
                vec[term] = (count / total) * self.idf[term]
        return vec

    def _cosine(self, vec_a: dict, vec_b: dict) -> float:
        common = set(vec_a) & set(vec_b)
        dot = sum(vec_a[t] * vec_b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        query_tokens = _tokenize(query)
        query_vec = self._vectorize(query_tokens)
        scores = [
            (chunk, self._cosine(query_vec, self.doc_vectors[i]))
            for i, chunk in enumerate(self.chunks)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
