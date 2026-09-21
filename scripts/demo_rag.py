#!/usr/bin/env python3
"""
Demo: ingest the sample policy documents, run retrieval, and report an
evaluation hit-rate. Runs fully offline (TfidfEmbedder + EchoLLMClient).

To use a real LLM instead of the offline stub: set OPENAI_API_KEY or
ANTHROPIC_API_KEY, install this on a machine with network access, and swap
EchoLLMClient() below for OpenAIClient() or AnthropicClient().
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragkit.embeddings import TfidfEmbedder
from ragkit.eval import EvalCase, evaluate_retrieval
from ragkit.llm_client import EchoLLMClient
from ragkit.rag_pipeline import RAGPipeline

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def main():
    pipeline = RAGPipeline(embedder=TfidfEmbedder(), llm=EchoLLMClient())

    print("Ingesting documents...")
    for path in sorted(DATA_DIR.glob("*.txt")):
        n_chunks = pipeline.ingest(doc_id=path.stem, text=path.read_text())
        print(f"  {path.name}: {n_chunks} chunk(s)")

    n_indexed = pipeline.build_index()
    print(f"Index built: {n_indexed} chunks total, {len(list(DATA_DIR.glob('*.txt')))} documents\n")

    queries = [
        "How many days do I have to return an unopened product?",
        "Is shipping free and how long does it take?",
        "Does the warranty cover accidental drops?",
        "Can I return Gadget Max after opening it?",
    ]

    print("=" * 70)
    print("RAG QUERIES")
    print("=" * 70)
    for q in queries:
        result = pipeline.query(q, top_k=2)
        print(f"\nQ: {q}")
        print(f"Top source: [{result.sources[0].doc_id}] (score={result.sources[0].score:.3f})")
        print(f"A: {result.answer}")

    print("\n" + "=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)
    eval_cases = [
        EvalCase(query="return policy for unopened items", expected_doc_id="returns_policy"),
        EvalCase(query="cost and speed of shipping", expected_doc_id="shipping_policy"),
        EvalCase(query="what does the warranty not cover", expected_doc_id="warranty_policy"),
        EvalCase(query="extended warranty on premium products", expected_doc_id="warranty_policy"),
    ]
    report = evaluate_retrieval(pipeline, eval_cases, top_k=2)
    print(f"Hit rate @ top-2: {report.hits}/{report.total} = {report.hit_rate:.0%}")
    if report.misses:
        print("Missed queries:")
        for m in report.misses:
            print(f"  - '{m.query}' (expected {m.expected_doc_id})")


if __name__ == "__main__":
    main()
