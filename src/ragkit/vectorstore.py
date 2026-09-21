"""
vectorstore.py — a minimal, real vector store.

WHY BUILD A TOY ONE INSTEAD OF JUST IMPORTING CHROMA/FAISS:

Two reasons, both deliberate. First, this sandbox has no network access to
pull a Chroma/FAISS-backed model or persist to a hosted store, so an
in-memory implementation is what's actually testable here. Second -- and
more important for an interview -- understanding what FAISS/Chroma/Pinecone
are *doing under the hood* (storing vectors, computing similarity, returning
top-k) is exactly the kind of question this JD's "work with vector
databases" line implies you should be able to answer, not just name-drop.
This class implements the same conceptual contract (add + query top-k) that
every real vector DB exposes, using brute-force cosine similarity.

BRUTE FORCE IS THE HONEST BASELINE, NOT THE PRODUCTION ANSWER:
Cosine similarity against every stored vector is O(n) per query -- fine for
thousands of chunks, the wrong approach past roughly 100K+ vectors. Real
vector databases (FAISS, Chroma's HNSW index, Pinecone) use approximate
nearest-neighbor structures (e.g. HNSW graphs, IVF indexes) to answer top-k
queries in sublinear time, trading a small amount of recall for a large
speedup at scale. Knowing *why* you'd reach for an ANN index instead of
brute force -- and that it trades exactness for speed -- is a stronger
answer than just knowing FAISS's API surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float


class SimpleVectorStore:
    def __init__(self):
        self._ids: list[str] = []
        self._doc_ids: list[str] = []
        self._texts: list[str] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunk_id: str, doc_id: str, text: str, vector: np.ndarray) -> None:
        self._ids.append(chunk_id)
        self._doc_ids.append(doc_id)
        self._texts.append(text)
        vec = vector.astype(np.float32).reshape(1, -1)
        self._vectors = vec if self._vectors is None else np.vstack([self._vectors, vec])

    def size(self) -> int:
        return len(self._ids)

    def query(self, query_vector: np.ndarray, top_k: int = 3) -> list[SearchResult]:
        if self._vectors is None or self.size() == 0:
            return []

        q = query_vector.astype(np.float32).reshape(1, -1)

        # Cosine similarity = dot(a, b) / (|a| * |b|). Guard against
        # zero-norm vectors (an all-zero TF-IDF row for a chunk with no
        # vocabulary overlap at all) to avoid a divide-by-zero producing NaN.
        store_norms = np.linalg.norm(self._vectors, axis=1)
        q_norm = np.linalg.norm(q)
        denom = store_norms * q_norm
        denom[denom == 0] = 1e-10

        sims = (self._vectors @ q.T).flatten() / denom

        top_k = min(top_k, self.size())
        top_idx = np.argsort(-sims)[:top_k]

        return [
            SearchResult(
                chunk_id=self._ids[i],
                doc_id=self._doc_ids[i],
                text=self._texts[i],
                score=float(sims[i]),
            )
            for i in top_idx
        ]
