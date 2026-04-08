"""
Step 2: Generate synthetic (question, answer, source_chunk_id) triples.

Uses any OpenAI-compatible LLM to produce realistic questions from each chunk.
The LLM is only used here — everything else in the pipeline is pure Python.

If no API key is set, falls back to a small set of hand-written questions
for the sample dataset so the demo still runs.
"""

import json
import os
import re
from .corpus import Chunk


PROMPT_TEMPLATE = """You are building a retrieval evaluation dataset.

Given the document chunk below, generate {n} realistic questions that:
1. Can ONLY be answered using information in this specific chunk
2. Sound like questions a real employee would ask
3. Vary in phrasing — don't just copy phrases from the text

For each question also provide a short expected answer (1-2 sentences).

Return a JSON array with this format:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Return only the JSON array, no other text.

CHUNK:
{chunk_text}
"""


def generate_questions(
    chunks: list[Chunk],
    n_per_chunk: int = 2,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> list[dict]:
    """
    Generate synthetic QA pairs from chunks.

    Returns a list of dicts:
      [{"question": str, "answer": str, "chunk_id": str, "source": str}, ...]

    Falls back to hand-written questions if no API key is available.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")

    if not key:
        print("  No OPENAI_API_KEY found — using built-in sample questions.")
        return _fallback_questions(chunks)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
    except ImportError:
        print("  openai package not installed — using built-in sample questions.")
        return _fallback_questions(chunks)

    qa_pairs = []
    for chunk in chunks:
        prompt = PROMPT_TEMPLATE.format(n=n_per_chunk, chunk_text=chunk.text)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            pairs = json.loads(raw)
            for pair in pairs:
                qa_pairs.append({
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "chunk_id": chunk.id,
                    "source": chunk.source,
                })
        except Exception as e:
            print(f"  Warning: failed to generate questions for {chunk.id}: {e}")

    return qa_pairs


def _fallback_questions(chunks: list[Chunk]) -> list[dict]:
    """
    Hand-written questions for the sample dataset.
    Each question maps to the chunk it should be retrieved from.
    """
    chunk_map = {c.source: c for c in chunks}

    questions = []

    if "q3_product_review.txt" in chunk_map:
        c = chunk_map["q3_product_review.txt"]
        questions += [
            {"question": "What was the outcome of the RBAC refactor in Q3?",
             "answer": "Daniel's team completed the RBAC refactor. Document-level permissions now propagate through the inheritance chain.",
             "chunk_id": c.id, "source": c.source},
            {"question": "What is the Q4 priority related to search?",
             "answer": "Building smart empty states for search, owned by Sarah, due October 15.",
             "chunk_id": c.id, "source": c.source},
            {"question": "By how much did mobile load time improve in Q3?",
             "answer": "Load times dropped from 4.2s to 1.8s after lazy loading changes shipped in September.",
             "chunk_id": c.id, "source": c.source},
        ]

    if "engineering_rfc_auth.txt" in chunk_map:
        c = chunk_map["engineering_rfc_auth.txt"]
        questions += [
            {"question": "What encryption method is used for caching tokens on mobile?",
             "answer": "AES-GCM 256-bit via Web Crypto API for browsers, Android Keystore / iOS Secure Enclave on mobile.",
             "chunk_id": c.id, "source": c.source},
            {"question": "What is the target P50 cold load time after the auth RFC is implemented?",
             "answer": "Under 1.0 seconds on mobile.",
             "chunk_id": c.id, "source": c.source},
            {"question": "Why was service worker caching rejected as an alternative?",
             "answer": "It adds complexity and has poor support on iOS Safari.",
             "chunk_id": c.id, "source": c.source},
        ]

    if "hr_policy_pto.txt" in chunk_map:
        c = chunk_map["hr_policy_pto.txt"]
        questions += [
            {"question": "How many weeks of parental leave does a primary caregiver get?",
             "answer": "16 weeks of fully paid parental leave.",
             "chunk_id": c.id, "source": c.source},
            {"question": "Does unused PTO roll over at the end of the year?",
             "answer": "No, unused PTO is forfeited on December 31.",
             "chunk_id": c.id, "source": c.source},
            {"question": "How far in advance do I need to request more than 3 days off?",
             "answer": "At least 5 business days in advance, submitted in Rippling.",
             "chunk_id": c.id, "source": c.source},
        ]

    if "sales_pipeline_oct.txt" in chunk_map:
        c = chunk_map["sales_pipeline_oct.txt"]
        questions += [
            {"question": "What is Meridian Financial Group's main concern about the deal?",
             "answer": "Data residency — they require US-only storage.",
             "chunk_id": c.id, "source": c.source},
            {"question": "Why did we lose Thornfield Manufacturing?",
             "answer": "We were 40% more expensive and couldn't justify the price difference to a cost-focused buyer.",
             "chunk_id": c.id, "source": c.source},
            {"question": "What document does Lakeview Health System need by October 18?",
             "answer": "The SOC 2 Type II report.",
             "chunk_id": c.id, "source": c.source},
        ]

    if "onboarding_guide.txt" in chunk_map:
        c = chunk_map["onboarding_guide.txt"]
        questions += [
            {"question": "What laptop do new employees receive?",
             "answer": "A MacBook Pro M3 with 16GB RAM.",
             "chunk_id": c.id, "source": c.source},
            {"question": "How long do I have to enroll in health benefits?",
             "answer": "30 days from your start date.",
             "chunk_id": c.id, "source": c.source},
            {"question": "Which tools do engineers use for issue tracking?",
             "answer": "Linear is used for project and issue tracking by engineering and product roles.",
             "chunk_id": c.id, "source": c.source},
        ]

    return questions
