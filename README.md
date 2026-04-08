# retrieval-eval

A lightweight eval harness for retrieval pipelines — no labeled data required.

## The problem

When building RAG or knowledge systems over organizational data, you rarely have labeled query-document pairs. Without ground truth, you can't measure whether your retrieval is actually working or compare strategies objectively.

## What this does

1. **Ingests** a folder of documents and splits them into chunks
2. **Generates** realistic test questions from the corpus using an LLM (or uses built-in questions for the sample dataset)
3. **Runs** five retrieval strategies against every question:
   - BM25 (keyword matching)
   - TF-IDF (cosine similarity)
   - Hybrid RRF (combines BM25 + TF-IDF via Reciprocal Rank Fusion)
   - Dense (local semantic embeddings via fastembed)
   - Cross-Encoder (transformer-based re-ranking — most accurate)
4. **Scores** each strategy on Recall@k, MRR, and latency per query
5. **Reports** which strategy wins and which questions every strategy fails on

## Quickstart

```bash
pip install fastembed
python examples/demo.py
```

No API key required. The dense and cross-encoder models download automatically on first run (~110MB total).

## Results on sample dataset

```
Strategy                      R@1     R@3     R@5     MRR     Latency
------------------------------------------------------------------------
BM25                          0.867   1.000   1.000   0.933   0.0ms
TF-IDF                        0.867   1.000   1.000   0.933   0.1ms
Hybrid (RRF)                  0.867   1.000   1.000   0.933   0.1ms
Dense (Semantic)              0.867   1.000   1.000   0.933   4.5ms
Cross-Encoder (Transformer)   1.000*  1.000*  1.000*  1.000*  203.4ms
```

The cross-encoder achieves perfect retrieval at the cost of latency — 200x slower than BM25. In production, the right architecture is a two-stage pipeline: a fast retriever (BM25 or Dense) narrows to top-20 candidates, then the cross-encoder re-ranks them. The top result goes to the LLM.

The failure analysis reveals *why* strategies fail differently:
- BM25/Hybrid miss semantically framed questions ("what should I expect on my first day?")
- Dense misses exact keyword matches ("what was the main bottleneck for mobile performance?")
- Cross-encoder gets both right by reading query and document together

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

To use LLM-generated questions instead of built-in samples:
```bash
OPENAI_API_KEY=sk-... python examples/demo.py
```

## Pipeline

```
Documents → Chunking → LLM Question Generation (synthetic ground truth)
                                    ↓
                          BM25 / TF-IDF / Hybrid / Dense / Cross-Encoder
                                    ↓
                        Recall@k  |  MRR  |  Latency  |  Failure Analysis
```

## Design

- **Zero external dependencies** for BM25, TF-IDF, and Hybrid — pure Python math
- **No API key required** — Dense and Cross-Encoder run locally via fastembed
- **Pluggable** — add any retriever that implements `.search(query, k)`
- **Model-agnostic** — bring any OpenAI-compatible API for question generation

## Adding a custom retriever

```python
class MyRetriever:
    def __init__(self, chunks: list[Chunk]):
        ...

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        # Return top-k (chunk, score) pairs, sorted by score descending
        ...
```

Pass it to `evaluate()` in `metrics.py`.
