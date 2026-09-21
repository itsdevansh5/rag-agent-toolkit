"""
chunking.py — splits raw documents into overlapping chunks for embedding.

WHY CHUNK AT ALL, AND WHY OVERLAP:

An LLM's context window is finite, and retrieval quality degrades if you
embed an entire document as one vector -- a single embedding for a 5-page
doc averages away the specific paragraph a query actually needs, so
semantically distinct sections wash each other out. Chunking keeps each
embedded unit focused enough that similarity search can find the *right
paragraph*, not just the right document.

Overlap (a few sentences shared between consecutive chunks) exists because
naive non-overlapping splits can cut a key sentence in half across a chunk
boundary, so the fact you need is split between two chunks and neither one
scores well against the query. A small overlap trades a bit of redundant
storage for not losing information at chunk edges -- this is a real,
commonly-tuned parameter in production RAG systems, not an arbitrary choice.

This is a simple fixed-size, word-count-based splitter. Production systems
often chunk by sentence/paragraph boundaries (to avoid cutting mid-sentence)
or by token count (to match the embedding model's actual tokenizer) --
flagged here as the natural next step, not implemented, to keep this
dependency-free and offline-runnable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    start_word: int


def chunk_text(doc_id: str, text: str, chunk_size: int = 120, overlap: int = 20) -> list[Chunk]:
    """Split `text` into overlapping word-count chunks.

    chunk_size: words per chunk. overlap: words shared between consecutive
    chunks. Both are tunable per corpus -- shorter chunks give more precise
    retrieval but more chunks to search and more redundant context; longer
    chunks give richer context per hit but blunter precision.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size, or chunks never advance")

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}::chunk{idx}",
                text=" ".join(chunk_words),
                start_word=start,
            )
        )
        idx += 1
        if end == len(words):
            break
        start += step

    return chunks
