"""
Cross-encoder re-ranker — two-stage retrieval using a transformer.

Stage 1: Dense retriever fetches top-N candidates quickly.
Stage 2: Cross-encoder reads (query, document) together and re-ranks.

Unlike dense/sparse retrievers that score query and document separately,
a cross-encoder reads both as a single input and outputs a joint relevance
score — much more accurate but too slow to run over all documents.
"""

from ..corpus import Chunk
from .dense import DenseRetriever

DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
CANDIDATE_POOL = 20


class CrossEncoderReranker:
    def __init__(self, chunks: list[Chunk], first_stage_retriever=None):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:
            raise ImportError(
                "fastembed is required. Install with: pip install fastembed"
            ) from exc

        self.chunks = chunks
        self.retriever = first_stage_retriever or DenseRetriever(chunks)
        self.reranker = TextCrossEncoder(model_name=DEFAULT_MODEL)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        # Stage 1: fetch candidates with fast retriever
        candidates = self.retriever.search(query, k=min(CANDIDATE_POOL, len(self.chunks)))
        candidate_chunks = [chunk for chunk, _ in candidates]

        # Stage 2: re-rank with cross-encoder
        texts = [chunk.text for chunk in candidate_chunks]
        scores = list(self.reranker.rerank(query, texts))

        ranked = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in ranked[:k]]
