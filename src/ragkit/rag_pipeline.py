"""
rag_pipeline.py — ties chunking + embedding + retrieval + generation together.

THE CORE RAG LOOP, AND WHY EACH STEP EXISTS:

  1. ingest() + build_index(): chunk each document, then fit the embedder on
     the full corpus and embed every chunk in one batch. This is the
     "indexing" half of RAG, separate from the "query" half below. Splitting
     ingest (collect) from build_index (commit) is not incidental -- an
     earlier version of this file embedded each document immediately on
     ingest, and with TF-IDF (whose vector space is exactly the corpus
     vocabulary) that meant document 1 got embedded into a smaller space
     than document 2 once the vocabulary grew, producing vectors of
     different dimensionality in the same store. The test suite caught it
     as a hard crash. Batch indexing is the fix, and it also matches how
     production RAG pipelines are actually structured.
  2. query(): embed the user's question with the SAME embedder/index used at
     ingest time (embedding spaces aren't comparable across different
     models or across an index built before vs. after new documents were
     added -- see above), retrieve the top-k most similar chunks, and stuff
     them into a prompt template alongside the question.
  3. The LLM never sees the whole corpus -- only the handful of chunks
     retrieval judged most relevant. This is *why* RAG lets an LLM answer
     questions about documents far larger than its context window, and why
     retrieval quality is the actual bottleneck of most RAG systems in
     practice: a perfect LLM given the wrong context still gives a wrong
     answer, so the vector store / embedder choice usually matters more
     than the generation model choice.

This module is intentionally provider-agnostic -- it depends only on the
Embedder and LLMClient interfaces, never on a specific model or vector DB.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragkit.chunking import chunk_text
from ragkit.embeddings import Embedder
from ragkit.llm_client import LLMClient
from ragkit.vectorstore import SearchResult, SimpleVectorStore

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context doesn't contain the answer, say so "
    "explicitly instead of guessing -- do not use outside knowledge."
)


@dataclass
class RAGAnswer:
    answer: str
    sources: list[SearchResult]


class RAGPipeline:
    def __init__(self, embedder: Embedder, llm: LLMClient, chunk_size: int = 120, overlap: int = 20):
        self.embedder = embedder
        self.llm = llm
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.store = SimpleVectorStore()
        self._pending_chunks: list = []  # chunked, not yet embedded/indexed
        self._indexed = False

    def ingest(self, doc_id: str, text: str) -> int:
        """Chunk one document and queue it for indexing. Returns chunk count.

        Deliberately does NOT embed or index immediately. TF-IDF's vector
        space is defined by the corpus vocabulary at fit-time -- embedding
        document 1's chunks right away, then growing the vocabulary when
        document 2 arrives, would silently produce two batches of vectors
        with different dimensionality (an earlier version of this file did
        exactly that, and the test suite caught it as a shape-mismatch
        crash when both batches landed in the same vector store). Ingestion
        here is a "collect" step; build_index() is the "commit" step,
        mirroring how batch/offline RAG indexing pipelines are structured
        in practice -- ingest everything, then build the index once.
        """
        chunks = chunk_text(doc_id, text, self.chunk_size, self.overlap)
        self._pending_chunks.extend(chunks)
        self._indexed = False
        return len(chunks)

    def build_index(self) -> int:
        """Fit the embedder on the full accumulated corpus and populate the
        vector store. Must be called after ingest() and before retrieve()/
        query(). Returns the number of chunks indexed."""
        if not self._pending_chunks:
            raise RuntimeError("No documents ingested yet -- call ingest() first")

        texts = [c.text for c in self._pending_chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.embed(texts)

        self.store = SimpleVectorStore()
        for c, vec in zip(self._pending_chunks, vectors):
            self.store.add(c.chunk_id, c.doc_id, c.text, vec)

        self._indexed = True
        return len(self._pending_chunks)

    def retrieve(self, question: str, top_k: int = 3) -> list[SearchResult]:
        if not self._indexed:
            raise RuntimeError("Index not built -- call build_index() after ingest()")
        q_vector = self.embedder.embed([question])[0]
        return self.store.query(q_vector, top_k=top_k)

    def build_prompt(self, question: str, sources: list[SearchResult]) -> str:
        context = "\n\n".join(f"[{s.chunk_id}] {s.text}" for s in sources)
        return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    def query(self, question: str, top_k: int = 3) -> RAGAnswer:
        sources = self.retrieve(question, top_k=top_k)
        prompt = self.build_prompt(question, sources)
        answer = self.llm.complete(SYSTEM_PROMPT, prompt)
        return RAGAnswer(answer=answer, sources=sources)
