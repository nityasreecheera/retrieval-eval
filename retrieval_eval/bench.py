"""
Main benchmark orchestrator.

Usage:
    from retrieval_eval.bench import run_bench
    run_bench("data/")
"""

from .corpus import load_corpus
from .synthetic import generate_questions
from .retrievers.bm25 import BM25Retriever
from .retrievers.tfidf import TFIDFRetriever
from .retrievers.hybrid import HybridRetriever
from .metrics import evaluate

try:
    from .retrievers.dense import DenseRetriever
    _DENSE_AVAILABLE = True
except ImportError:
    _DENSE_AVAILABLE = False

try:
    from .retrievers.reranker import CrossEncoderReranker
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False


def run_bench(data_dir, chunk_size=400, overlap=50, n_questions_per_chunk=2,
              k_values=(1, 3, 5), model="gpt-4o-mini"):
    """Run the full retrieval eval pipeline and print a report."""
    print("=" * 60)
    print("  RETRIEVAL EVAL BENCH")
    print("=" * 60)

    print(f"\n[1/4] Loading corpus from '{data_dir}'...")
    chunks = load_corpus(data_dir, chunk_size=chunk_size, overlap=overlap)
    n_docs = len(set(c.source for c in chunks))
    print(f"      {len(chunks)} chunks from {n_docs} documents")

    print("\n[2/4] Generating test questions...")
    qa_pairs = generate_questions(
        chunks, n_per_chunk=n_questions_per_chunk, model=model,
    )
    print(f"      {len(qa_pairs)} questions generated")

    print("\n[3/4] Building retrieval indexes...")
    retrievers = {
        "BM25": BM25Retriever(chunks),
        "TF-IDF": TFIDFRetriever(chunks),
        "Hybrid (RRF)": HybridRetriever(chunks),
    }

    dense = None
    if _DENSE_AVAILABLE:
        print("      Loading dense embedding model (first run downloads ~25MB)...")
        dense = DenseRetriever(chunks)
        retrievers["Dense (Semantic)"] = dense
    else:
        print("      Dense retriever unavailable (run: pip install fastembed)")

    if _RERANKER_AVAILABLE and dense is not None:
        print("      Loading cross-encoder re-ranker (first run downloads ~85MB)...")
        retrievers["Cross-Encoder (Transformer)"] = CrossEncoderReranker(chunks, first_stage_retriever=dense)
    elif _RERANKER_AVAILABLE:
        print("      Cross-encoder skipped (requires dense retriever)")

    print("\n[4/4] Running evaluation...\n")
    results = {}
    for name, retriever in retrievers.items():
        results[name] = evaluate(retriever, qa_pairs, k_values=list(k_values))

    _print_report(results, qa_pairs, k_values)
    return results


def _best_per_metric(results, k_values):
    best = {f"recall@{k}": max(r[f"recall@{k}"] for r in results.values()) for k in k_values}
    best["mrr"] = max(r["mrr"] for r in results.values())
    return best


def _print_report(results, qa_pairs, k_values):
    """Print a formatted results table and failure analysis."""
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)

    k_headers = "  ".join(f"R@{k:<4}" for k in k_values)
    print(f"\n{'Strategy':<28}  {k_headers}  {'MRR':<6}  {'Latency':<10}")
    print("-" * 72)

    best = _best_per_metric(results, k_values)

    for name, metrics in results.items():
        row = f"{name:<28}  "
        for k in k_values:
            val = metrics[f"recall@{k}"]
            marker = "*" if val == best[f"recall@{k}"] else " "
            row += f"{val:.3f}{marker}  "
        mrr = metrics["mrr"]
        marker = "*" if mrr == best["mrr"] else " "
        row += f"{mrr:.3f}{marker}  "
        row += f"{metrics['latency_ms']:.2f}ms"
        print(row)

    print("\n  * = best in column\n")
    _print_failures(results)

    print("=" * 60)
    print(f"  Total questions: {len(qa_pairs)}")
    print("=" * 60)


def _print_failures(results):
    """Print failure analysis for questions where top-1 retrieval was wrong."""
    print("=" * 60)
    print("  FAILURE ANALYSIS  (questions where top-1 was wrong)")
    print("=" * 60)

    all_failures = {}
    for name, metrics in results.items():
        for failure in metrics["failures"]:
            key = failure["question"]
            if key not in all_failures:
                all_failures[key] = {
                    "question": failure["question"],
                    "expected": failure["expected_source"],
                    "top_retrieved": failure["top_retrieved_source"],
                    "failed_in": [],
                }
            all_failures[key]["failed_in"].append(name)

    if not all_failures:
        print("\n  All strategies retrieved the correct chunk at rank 1 for every question.")
        return

    hard = [v for v in all_failures.values() if len(v["failed_in"]) == len(results)]
    partial = [v for v in all_failures.values() if len(v["failed_in"]) < len(results)]

    if hard:
        print("\n  Hard failures (missed by ALL strategies):")
        for f in hard:
            print(f"    Q: {f['question']}")
            print(f"       Expected: {f['expected']}  |  Got: {f['top_retrieved']}\n")

    if partial:
        print("  Partial failures (missed by some strategies):")
        for f in partial:
            missed = ", ".join(f["failed_in"])
            print(f"    Q: {f['question']}")
            print(f"       Missed by: {missed}\n")
