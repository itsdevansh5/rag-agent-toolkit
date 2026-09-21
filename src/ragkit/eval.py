"""
eval.py — retrieval evaluation harness.

WHY EVALUATE RETRIEVAL SEPARATELY FROM GENERATION:
A RAG system's failures decompose into two independent questions: (1) did
retrieval find the right source material, and (2) given that material, did
the LLM generate a correct answer from it. Conflating them makes debugging
much harder -- if the final answer is wrong, you first need to know whether
retrieval handed the model garbage (a retrieval problem, fixed by chunking/
embedding/index changes) or handed it the right material and the model
still got it wrong (a generation/prompting problem, fixed by prompt or model
changes). This module evaluates retrieval alone, using labeled test cases,
so that question can be answered without needing a real LLM call at all --
which is also why it's the one piece of "evaluation" from this JD's
requirements that's fully exercised and tested in this offline sandbox.

METRIC: hit-rate@k -- for each test query, was the expected document among
the top-k retrieved chunks? This is the simplest meaningful retrieval
metric and a reasonable first thing to report; production systems often add
MRR (mean reciprocal rank, rewarding the right answer appearing *earlier*)
and NDCG once there's a large enough labeled query set to make the extra
resolution worthwhile.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragkit.rag_pipeline import RAGPipeline


@dataclass
class EvalCase:
    query: str
    expected_doc_id: str


@dataclass
class EvalReport:
    total: int
    hits: int
    hit_rate: float
    misses: list[EvalCase]


def evaluate_retrieval(pipeline: RAGPipeline, cases: list[EvalCase], top_k: int = 3) -> EvalReport:
    hits = 0
    misses: list[EvalCase] = []

    for case in cases:
        results = pipeline.retrieve(case.query, top_k=top_k)
        retrieved_doc_ids = {r.doc_id for r in results}
        if case.expected_doc_id in retrieved_doc_ids:
            hits += 1
        else:
            misses.append(case)

    total = len(cases)
    return EvalReport(
        total=total,
        hits=hits,
        hit_rate=(hits / total) if total else 0.0,
        misses=misses,
    )
