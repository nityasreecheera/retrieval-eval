"""
Step 4: Evaluate retrieval quality.

Metrics:
  Recall@k  — was the correct chunk in the top-k results?
  MRR       — mean reciprocal rank (how high up was the correct chunk?)
  Precision@1 — was the very first result correct?
"""

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
        "failures": [{"question": str, "expected": str, "got": str}, ...]
      }
    """
    recall_totals = {k: 0.0 for k in k_values}
    mrr_total = 0.0
    failures = []
    n = len(qa_pairs)

    for qa in qa_pairs:
        query = qa["question"]
        correct_id = qa["chunk_id"]
        results = retriever.search(query, k=max(k_values))

        for k in k_values:
            recall_totals[k] += recall_at_k(results, correct_id, k)

        rr = reciprocal_rank(results, correct_id)
        mrr_total += rr

        # Track failures (not in top-1)
        if recall_at_k(results, correct_id, k=1) == 0:
            top_result = results[0][0].source if results else "none"
            failures.append({
                "question": query,
                "expected_source": qa["source"],
                "top_retrieved_source": top_result,
            })

    metrics = {f"recall@{k}": round(recall_totals[k] / n, 3) for k in k_values}
    metrics["mrr"] = round(mrr_total / n, 3)
    metrics["failures"] = failures
    return metrics
