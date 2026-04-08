"""
End-to-end demo using the sample company documents in data/.

Run from the project root:
    python examples/demo.py

To use an LLM for question generation instead of the built-in sample questions:
    OPENAI_API_KEY=sk-... python examples/demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval_eval.bench import run_bench

if __name__ == "__main__":
    run_bench(
        data_dir="data/",
        chunk_size=400,
        overlap=50,
        n_questions_per_chunk=3,
        k_values=[1, 3, 5],
    )
