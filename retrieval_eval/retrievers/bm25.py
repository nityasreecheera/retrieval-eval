"""
BM25 retriever — keyword-based ranking.

BM25 scores documents based on term frequency and inverse document frequency,
with length normalization. Good at exact keyword matches.
"""

import math
import re
from collections import Counter
from ..corpus import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._build_index()

    def _build_index(self):
        self.tokenized = [_tokenize(c.text) for c in self.chunks]
        self.doc_freqs = []
        self.avgdl = sum(len(t) for t in self.tokenized) / len(self.tokenized)
        self.N = len(self.chunks)

        # Document frequency: how many docs contain each term
        df = Counter()
        for tokens in self.tokenized:
            for term in set(tokens):
                df[term] += 1
        self.df = df

        # Term frequencies per document
        self.tf = [Counter(tokens) for tokens in self.tokenized]

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        query_terms = _tokenize(query)
        scores = []

        for i, chunk in enumerate(self.chunks):
            dl = len(self.tokenized[i])
            score = 0.0
            for term in query_terms:
                tf = self.tf[i].get(term, 0)
                idf = self._idf(term)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (numerator / denominator)
            scores.append((chunk, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
