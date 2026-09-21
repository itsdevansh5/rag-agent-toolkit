"""
embeddings.py — pluggable text-embedding interface.

WHY A PLUGGABLE INTERFACE AT ALL:

The embedding model is the single most consequential, most frequently
swapped component of a RAG system -- teams routinely change embedding
providers (OpenAI text-embedding-3, Cohere, a local sentence-transformers
model) as cost/quality trade-offs shift, and the rest of the pipeline
(chunking, vector store, retrieval, generation) shouldn't need to change
when that happens. This module defines one small `Embedder` interface and
gives two implementations:

  - TfidfEmbedder: classical, offline, dependency-light (scikit-learn only).
    Runs anywhere with no API key and no model download -- which is exactly
    why it's the one actually exercised by this repo's tests. It captures
    lexical/keyword overlap well but not semantic meaning (it won't know
    "car" and "automobile" are related) -- the honest limitation to state
    if asked.
  - NeuralEmbedder: a thin wrapper showing exactly where a real embedding
    API (OpenAI, Anthropic voyage-compatible endpoints, Cohere) or a local
    sentence-transformers model plugs in. Not exercised in this sandbox
    (no network access to those providers here) -- run it on your own
    machine with an API key, or install sentence-transformers there for a
    fully local neural option.

Swapping TfidfEmbedder for NeuralEmbedder anywhere in this codebase is a
one-line change, by design -- that's the point of the interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(ABC):
    @abstractmethod
    def fit(self, texts: list[str]) -> None:
        """Learn vocabulary/parameters from a corpus. No-op for most hosted
        neural embedding APIs, which are stateless per call -- required for
        TF-IDF, whose vector space is defined by the corpus it's fit on."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) float32 matrix of embeddings."""


class TfidfEmbedder(Embedder):
    """Classical bag-of-words embedding, offline and dependency-light.

    Every dimension corresponds to a vocabulary term; a document's value on
    that dimension is high if the term is frequent in the document but rare
    across the corpus (term-frequency * inverse-document-frequency). This
    means retrieval quality depends entirely on query/document *word
    overlap* -- it will not match a query about "physician" against a
    document that only says "doctor". A neural embedder (see NeuralEmbedder
    below) captures that kind of semantic similarity; TF-IDF does not.
    That's a real, worth-stating trade-off, not a bug.
    """

    def __init__(self, max_features: int = 4096):
        self._vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed()")
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32)


class NeuralEmbedder(Embedder):
    """Template for a real hosted/neural embedding provider.

    NOT exercised in this sandbox (no network access to embedding APIs or
    HuggingFace model downloads here). On your own machine:

      - OpenAI: `pip install openai`, set OPENAI_API_KEY, use
        client.embeddings.create(model="text-embedding-3-small", input=texts)
      - Local, no API key: `pip install sentence-transformers`, use
        SentenceTransformer("all-MiniLM-L6-v2").encode(texts)

    This class documents the call shape without executing it, so swapping
    it in is copy-editing, not a redesign.
    """

    def __init__(self, provider: str = "openai", model: str = "text-embedding-3-small"):
        self.provider = provider
        self.model = model

    def fit(self, texts: list[str]) -> None:
        return  # stateless per call for hosted embedding APIs

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(
            "Plug in a real client here, e.g.:\n"
            "  from openai import OpenAI\n"
            "  client = OpenAI()\n"
            "  resp = client.embeddings.create(model=self.model, input=texts)\n"
            "  return np.array([d.embedding for d in resp.data], dtype=np.float32)\n"
            "Requires OPENAI_API_KEY and network access this sandbox doesn't have."
        )
