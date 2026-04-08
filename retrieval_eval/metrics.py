"""
Step 4: Evaluate retrieval quality.

Metrics:
  Recall@k  — was the correct chunk in the top-k results?
  MRR       — mean reciprocal rank (how high up was the correct chunk?)
  Latency   — average time per query in milliseconds
"""

import time
from .corpus import Chunk


def recall_at_k(results: list[tuple[Chunk, float]], correct_chunk_id: str, k: int) -> float:
    """1.0 if correct chunk appears in top-k results, else 0.0"""
    top_k_ids = [chunk.id for chunk, _ in results[:k]]
    return 1.0 if correct_chunk_id in top_k_ids else 0.0


def reciprocal_rank(results: list[tuple[Chunk, float]], correct_chunk_id: str) -> float:
    """1/rank of the first correct result, or 0 if not found."""
    for rank, (chunk, _) in enumerate(results, start=1):
        if chunk.id == correct_chunk_id:
            return 1.0 / rank
    return 0.0


def evaluate(
    retriever,
    qa_pairs: list[dict],
    k_values: list[int] = (1, 3, 5),
) -> dict:
    """
    Run all QA pairs through a retriever and compute aggregate metrics.

    Returns:
      {
        "recall@1": float, "recall@3": float, "recall@5": float,
        "mrr": float,
        "latency_ms": float,
        "failures": [{"question": str, "expected_source": str, "top_retrieved_source": str}]
      }
    """
    recall_totals = {k: 0.0 for k in k_values}
    mrr_total = 0.0
    latency_total = 0.0
    failures = []
    n = len(qa_pairs)

    for qa in qa_pairs:
        query = qa["question"]
        correct_id = qa["chunk_id"]

        t0 = time.perf_counter()
        results = retriever.search(query, k=max(k_values))
        latency_total += (time.perf_counter() - t0) * 1000  # ms

        for k in k_values:
            recall_totals[k] += recall_at_k(results, correct_id, k)

        mrr_total += reciprocal_rank(results, correct_id)

        if recall_at_k(results, correct_id, k=1) == 0:
            top_result = results[0][0].source if results else "none"
            failures.append({
                "question": query,
                "expected_source": qa["source"],
                "top_retrieved_source": top_result,
            })

    metrics = {f"recall@{k}": round(recall_totals[k] / n, 3) for k in k_values}
    metrics["mrr"] = round(mrr_total / n, 3)
    metrics["latency_ms"] = round(latency_total / n, 1)
    metrics["failures"] = failures
    return metrics
