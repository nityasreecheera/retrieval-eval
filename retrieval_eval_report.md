# Retrieval Evaluation for Organizational Knowledge Systems

**Nitya Sree Cheera**
nityasree_cheera@berkeley.edu | +15105142290

---

## Motivation

AI systems are most useful when they can work with your own data — not just the public internet. Enterprise knowledge systems need to retrieve relevant information from internal documents, emails, and wikis to answer questions grounded in real organizational context.

The core challenge: how do you know if your retrieval is actually working? Standard benchmarks require manually labeled query-document pairs, which real organizations don't have. This project builds a framework to evaluate retrieval quality without any manual labeling.

---

## What I Built

A retrieval evaluation harness that automatically generates test questions from a document corpus and benchmarks multiple retrieval strategies against those questions.

**Pipeline:**
1. Documents are loaded and split into chunks
2. An LLM generates realistic questions from each chunk — this becomes the synthetic ground truth
3. Each question is run through every retrieval strategy
4. Results are scored and compared across strategies

This makes it possible to evaluate and compare retrieval approaches on any document corpus, with no manual annotation required.

---

## Retrieval Strategies

**BM25 (Keyword)**
Ranks documents by term frequency and inverse document frequency. Fast and effective for exact keyword matches, but misses synonyms and paraphrased queries.

**Dense (Semantic)**
Encodes documents and queries as vectors using a local embedding model (BAAI/bge-small-en-v1.5 via fastembed). Captures meaning rather than just word overlap. Runs locally with no API key required.

**Cross-Encoder (Transformer)**
A two-stage approach: Dense retrieves the top 20 candidates, then a transformer-based cross-encoder reads each (query, document) pair jointly and re-ranks them. Most accurate, but significantly slower.

---

## Results

Evaluated on 15 questions generated from 5 internal company documents (meeting notes, engineering RFCs, HR policies, sales updates, onboarding guides).

```
Strategy                      R@1     Latency
----------------------------------------------
BM25 (Keyword)                0.867   0.02ms
Dense (Semantic)              0.867   4.19ms
Cross-Encoder (Transformer)   1.000   201.55ms
```

**R@1** — fraction of questions where the correct document was the top result.

---

## Key Findings

**No single strategy dominates.** BM25 and Dense scored identically overall but failed on completely different questions:

- BM25 missed semantically framed questions like *"what should I expect on my first day?"* — the words don't overlap with the onboarding document even though the meaning does.
- Dense missed exact keyword matches like *"what was the main bottleneck for mobile performance?"* — the word "bottleneck" appears literally in the document, which BM25 handles better.

This shows the strategies have complementary weaknesses, not just different accuracy levels. You can't know this without measuring it.

**The cross-encoder fixes both failure types** by reading the query and document together, achieving perfect R@1. But it comes at a cost: 200x slower than BM25 per query.

---

## Tradeoffs

| | Accuracy | Latency | Dependencies |
|---|---|---|---|
| BM25 | 0.867 | 0.02ms | None |
| Dense | 0.867 | 4.19ms | fastembed |
| Cross-Encoder | 1.000 | 201.55ms | fastembed |

Running a cross-encoder over thousands of documents per query is too slow for production. The practical architecture is a two-stage pipeline: a fast retriever (BM25 or Dense) narrows to the top 20–50 candidates, and the cross-encoder re-ranks those. The real engineering challenge is balancing accuracy, latency, and cost across the full pipeline.

---

## How to Run

```bash
pip install fastembed
python examples/demo.py
```

No API key required. Models download automatically on first run (~110MB).

To use LLM-generated questions on your own corpus:
```bash
OPENAI_API_KEY=sk-... python examples/demo.py
```

Source code: https://github.com/nityasreecheera/retrieval-eval

---

## What I Would Do Next

- Fine-tune embeddings on domain-specific data for better semantic retrieval
- Add latency-aware evaluation to surface the accuracy/speed tradeoff at scale
- Scale to larger corpora to show where strategies diverge more meaningfully
- Experiment with hybrid retrieval combining BM25 and Dense before re-ranking
