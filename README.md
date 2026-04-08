# retrieval-eval

A lightweight eval harness for retrieval pipelines — no labeled data required.

## The problem

When building RAG or knowledge systems over organizational data, you rarely have labeled query-document pairs. Without ground truth, you can't measure whether your retrieval is actually working or compare strategies objectively.

## What this does

1. **Ingests** a folder of documents and splits them into chunks
2. **Generates** realistic test questions from the corpus using an LLM (or uses built-in questions for the sample dataset)
3. **Runs** three retrieval strategies against every question:
   - BM25 (keyword matching)
   - TF-IDF vectors (cosine similarity)
   - Hybrid RRF (combines both via Reciprocal Rank Fusion)
4. **Scores** each strategy on Recall@1, Recall@3, Recall@5, and MRR
5. **Reports** which strategy wins and which questions every strategy fails on

## Quickstart

```bash
# No install needed for core functionality
python examples/demo.py
```

To use LLM-generated questions instead of built-in samples:
```bash
OPENAI_API_KEY=sk-... python examples/demo.py
```

## Use on your own data

```python
from retrieval_eval.bench import run_bench

run_bench(
    data_dir="path/to/your/documents/",
    n_questions_per_chunk=3,
    k_values=[1, 3, 5],
)
```

Supports `.txt` and `.md` files. No configuration required.

## Output

```
============================================================
  RESULTS
============================================================

Strategy            R@1    R@3    R@5    MRR
-------------------------------------------------------
BM25                0.800  0.933  1.000  0.867
TF-IDF              0.733  0.867  0.933  0.800
Hybrid (RRF)        0.867* 0.933  1.000  0.900*

  * = best in column

============================================================
  FAILURE ANALYSIS  (questions where top-1 was wrong)
============================================================
...
```

## Design

- **Zero external dependencies** for retrieval and eval — pure Python math
- **Model-agnostic** — bring any OpenAI-compatible API for question generation
- **Retrieval is pluggable** — implement `.search(query, k)` to add your own strategy
- BM25 and TF-IDF are implemented from scratch (no libraries) for transparency

## Extending

Add a new retriever by implementing this interface:

```python
class MyRetriever:
    def __init__(self, chunks: list[Chunk]):
        ...

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        # Return top-k (chunk, score) pairs, sorted by score descending
        ...
```

Then pass it to `evaluate()` in `metrics.py`.
