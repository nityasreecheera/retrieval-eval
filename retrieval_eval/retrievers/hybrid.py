"""
Hybrid retriever — combines BM25 and TF-IDF via Reciprocal Rank Fusion (RRF).

RRF merges ranked lists from multiple retrievers without needing to normalize
their scores. Each document's final score is the sum of 1/(k + rank) across
all retrievers, where k=60 is a smoothing constant.

This usually outperforms either retriever alone because:
- BM25 handles exact keyword matches well
- TF-IDF handles semantic paraphrasing better
- RRF captures documents that rank highly in either
"""

from ..corpus import Chunk
from .bm25 import BM25Retriever
from .tfidf import TFIDFRetriever

RRF_K = 60


class HybridRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.tfidf = TFIDFRetriever(chunks)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        bm25_results = self.bm25.search(query, k=len(self.chunks))
        tfidf_results = self.tfidf.search(query, k=len(self.chunks))

        # Build rank lookup: chunk_id -> rank (1-indexed)
        bm25_ranks = {chunk.id: rank + 1 for rank, (chunk, _) in enumerate(bm25_results)}
        tfidf_ranks = {chunk.id: rank + 1 for rank, (chunk, _) in enumerate(tfidf_results)}

        # RRF score for each chunk
        rrf_scores: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {c.id: c for c in self.chunks}

        for chunk_id in chunk_by_id:
            score = 0.0
            if chunk_id in bm25_ranks:
                score += 1.0 / (RRF_K + bm25_ranks[chunk_id])
            if chunk_id in tfidf_ranks:
                score += 1.0 / (RRF_K + tfidf_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_by_id[cid], score) for cid, score in ranked[:k]]
