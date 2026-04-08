"""
Step 1: Ingest documents and split into chunks.

Each chunk is a dict with:
  - id: unique identifier
  - text: the chunk text
  - source: filename it came from
"""

import os
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.text[:80].replace("\n", " ")
        return f"Chunk(id={self.id!r}, source={self.source!r}, text={preview!r}...)"


def load_corpus(data_dir: str, chunk_size: int = 400, overlap: int = 50) -> list[Chunk]:
    """
    Load all .txt and .md files from data_dir and split into overlapping chunks.

    Args:
        data_dir:   Path to folder containing documents.
        chunk_size: Approximate number of words per chunk.
        overlap:    Number of words to overlap between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    chunks = []

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith((".txt", ".md")):
            continue

        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        file_chunks = _split_into_chunks(text, chunk_size, overlap)

        for i, chunk_text in enumerate(file_chunks):
            chunk_id = f"{filename}::chunk_{i}"
            chunks.append(Chunk(id=chunk_id, text=chunk_text, source=filename))

    return chunks


def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into chunks by word count with overlap.
    Tries to split on paragraph boundaries when possible.
    """
    # Split into paragraphs first to avoid cutting mid-sentence
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current_words = []

    for para in paragraphs:
        para_words = para.split()

        # If adding this paragraph exceeds chunk_size, flush current buffer
        if current_words and len(current_words) + len(para_words) > chunk_size:
            chunks.append(" ".join(current_words))
            # Keep overlap words for context continuity
            current_words = current_words[-overlap:] if overlap > 0 else []

        current_words.extend(para_words)

        # If a single paragraph is larger than chunk_size, split it hard
        while len(current_words) > chunk_size:
            chunks.append(" ".join(current_words[:chunk_size]))
            current_words = current_words[chunk_size - overlap:]

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks
