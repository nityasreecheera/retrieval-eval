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


def run_bench(
    data_dir: str,
    chunk_size: int = 400,
    overlap: int = 50,
    n_questions_per_chunk: int = 2,
    k_values: list[int] = (1, 3, 5),
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
):
    print("=" * 60)
    print("  RETRIEVAL EVAL BENCH")
    print("=" * 60)

    # Step 1: Load corpus
    print(f"\n[1/4] Loading corpus from '{data_dir}'...")
    chunks = load_corpus(data_dir, chunk_size=chunk_size, overlap=overlap)
    print(f"      {len(chunks)} chunks from {len(set(c.source for c in chunks))} documents")

    # Step 2: Generate test questions
    print(f"\n[2/4] Generating test questions...")
    qa_pairs = generate_questions(
        chunks,
        n_per_chunk=n_questions_per_chunk,
        model=model,
        api_key=api_key,
    )
    print(f"      {len(qa_pairs)} questions generated")

    # Step 3: Build retrievers
    print(f"\n[3/4] Building retrieval indexes...")
    retrievers = {
        "BM25": BM25Retriever(chunks),
        "TF-IDF": TFIDFRetriever(chunks),
        "Hybrid (RRF)": HybridRetriever(chunks),
    }

    # Step 4: Evaluate
    print(f"\n[4/4] Running evaluation...\n")
    results = {}
    for name, retriever in retrievers.items():
        results[name] = evaluate(retriever, qa_pairs, k_values=list(k_values))

    _print_report(results, qa_pairs, k_values)
    return results


def _print_report(results: dict, qa_pairs: list[dict], k_values):
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)

    # Header
    col_w = 14
    k_headers = "  ".join(f"R@{k:<4}" for k in k_values)
    print(f"\n{'Strategy':<18}  {k_headers}  {'MRR':<6}")
    print("-" * 55)

    # Scores
    best = {}
    for k in k_values:
        best[f"recall@{k}"] = max(r[f"recall@{k}"] for r in results.values())
    best["mrr"] = max(r["mrr"] for r in results.values())

    for name, metrics in results.items():
        row = f"{name:<18}  "
        for k in k_values:
            val = metrics[f"recall@{k}"]
            marker = "*" if val == best[f"recall@{k}"] else " "
            row += f"{val:.3f}{marker}  "
        mrr = metrics["mrr"]
        marker = "*" if mrr == best["mrr"] else " "
        row += f"{mrr:.3f}{marker}"
        print(row)

    print("\n  * = best in column\n")

    # Failure analysis
    print("=" * 60)
    print("  FAILURE ANALYSIS  (questions where top-1 was wrong)")
    print("=" * 60)

    all_failures = {}
    for name, metrics in results.items():
        for f in metrics["failures"]:
            key = f["question"]
            if key not in all_failures:
                all_failures[key] = {
                    "question": f["question"],
                    "expected": f["expected_source"],
                    "failed_in": [],
                }
            all_failures[key]["failed_in"].append(name)
            all_failures[key]["top_retrieved"] = f["top_retrieved_source"]

    if not all_failures:
        print("\n  All strategies retrieved the correct chunk at rank 1 for every question.")
    else:
        hard = [v for v in all_failures.values() if len(v["failed_in"]) == len(results)]
        easy_miss = [v for v in all_failures.values() if len(v["failed_in"]) < len(results)]

        if hard:
            print(f"\n  Hard failures (missed by ALL strategies):")
            for f in hard:
                print(f"    Q: {f['question']}")
                print(f"       Expected: {f['expected']}  |  Got: {f['top_retrieved']}\n")

        if easy_miss:
            print(f"  Partial failures (missed by some strategies):")
            for f in easy_miss:
                missed = ", ".join(f["failed_in"])
                print(f"    Q: {f['question']}")
                print(f"       Missed by: {missed}\n")

    print("=" * 60)
    print(f"  Total questions: {len(qa_pairs)}")
    print("=" * 60)
