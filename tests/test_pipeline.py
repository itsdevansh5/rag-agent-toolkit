"""
Offline correctness tests. Every test here runs with TfidfEmbedder and
EchoLLMClient -- no network, no API key, no external model download.
This is deliberate: it proves the plumbing (chunking boundaries, retrieval
ranking, prompt assembly, agent parsing/tool-calling) is correct
independent of which LLM/embedding provider eventually sits behind the
interface. Swapping in NeuralEmbedder/OpenAIClient later doesn't change
what these tests assert.
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragkit.agent import Agent
from ragkit.chunking import chunk_text
from ragkit.embeddings import TfidfEmbedder
from ragkit.eval import EvalCase, evaluate_retrieval
from ragkit.llm_client import EchoLLMClient
from ragkit.rag_pipeline import RAGPipeline
from ragkit.tools import build_demo_database, make_search_tool, make_sql_tool
from ragkit.vectorstore import SimpleVectorStore


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text("doc1", text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    # every chunk except the last should be exactly chunk_size words
    for c in chunks[:-1]:
        assert len(c.text.split()) == 100
    # consecutive chunks should share the overlap region
    first_tail = chunks[0].text.split()[-20:]
    second_head = chunks[1].text.split()[:20]
    assert first_tail == second_head


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("doc1", "") == []


def test_chunk_text_rejects_overlap_ge_chunk_size():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        chunk_text("doc1", "a b c", chunk_size=10, overlap=10)


# --------------------------------------------------------------------------
# vector store
# --------------------------------------------------------------------------

def test_vectorstore_returns_closest_match_first():
    import numpy as np

    store = SimpleVectorStore()
    store.add("a", "doc1", "about cats", np.array([1.0, 0.0]))
    store.add("b", "doc1", "about dogs", np.array([0.0, 1.0]))
    store.add("c", "doc1", "also about cats", np.array([0.9, 0.1]))

    results = store.query(np.array([1.0, 0.0]), top_k=2)
    assert results[0].chunk_id == "a"
    assert results[1].chunk_id == "c"  # closer to the cat vector than "dogs"


def test_vectorstore_handles_zero_vector_without_nan():
    import math

    import numpy as np

    store = SimpleVectorStore()
    store.add("a", "doc1", "empty", np.zeros(3))
    results = store.query(np.array([1.0, 0.0, 0.0]), top_k=1)
    assert not math.isnan(results[0].score)


# --------------------------------------------------------------------------
# end-to-end RAG retrieval (real TF-IDF, offline)
# --------------------------------------------------------------------------

def _sample_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline(embedder=TfidfEmbedder(), llm=EchoLLMClient())
    pipeline.ingest(
        "returns",
        "Customers may return any unopened product within 30 days for a full refund. "
        "Opened electronics may be returned within 14 days with original packaging.",
    )
    pipeline.ingest(
        "shipping",
        "Standard shipping takes 5 to 7 business days and is free on orders over 50 dollars. "
        "Expedited shipping is available for an additional fee.",
    )
    pipeline.ingest(
        "warranty",
        "All products come with a 1 year limited warranty covering manufacturing defects. "
        "The warranty does not cover accidental drops or water damage.",
    )
    pipeline.build_index()
    return pipeline


def test_retrieval_finds_relevant_document_for_returns_query():
    pipeline = _sample_pipeline()
    results = pipeline.retrieve("how many days do I have to return an unopened item", top_k=1)
    assert results[0].doc_id == "returns"


def test_retrieval_finds_relevant_document_for_shipping_query():
    pipeline = _sample_pipeline()
    results = pipeline.retrieve("is shipping free and how long does it take", top_k=1)
    assert results[0].doc_id == "shipping"


def test_query_end_to_end_includes_retrieved_context_in_prompt_to_llm():
    pipeline = _sample_pipeline()
    answer = pipeline.query("what does the warranty cover", top_k=1)
    assert answer.sources[0].doc_id == "warranty"
    # EchoLLMClient echoes back a snippet of the context it was given --
    # this assertion proves retrieval output actually reached the prompt,
    # not just that retrieval ran.
    assert "warranty" in answer.answer.lower() or "manufacturing" in answer.answer.lower()


# --------------------------------------------------------------------------
# retrieval evaluation harness
# --------------------------------------------------------------------------

def test_evaluate_retrieval_reports_hit_rate():
    pipeline = _sample_pipeline()
    cases = [
        EvalCase(query="return policy for unopened items", expected_doc_id="returns"),
        EvalCase(query="how much does shipping cost", expected_doc_id="shipping"),
        EvalCase(query="what voids the warranty", expected_doc_id="warranty"),
    ]
    report = evaluate_retrieval(pipeline, cases, top_k=1)
    assert report.total == 3
    assert report.hit_rate >= 0.66  # allow for TF-IDF's known lexical-overlap limits


# --------------------------------------------------------------------------
# agent loop (parsing/tool-routing logic, offline via EchoLLMClient)
# --------------------------------------------------------------------------

def test_agent_parses_action_and_calls_the_right_tool():
    pipeline = _sample_pipeline()
    search_tool = make_search_tool(pipeline)

    canned = "ACTION: search_knowledge_base\nACTION_INPUT: warranty coverage"
    agent = Agent(llm=EchoLLMClient(canned_tool_call=canned), tools=[search_tool], max_steps=1)

    result = agent.run("What does the warranty cover?")
    # With max_steps=1 and a canned ACTION response, the loop should hit the
    # step cap after one tool call rather than crash -- confirms the
    # ACTION -> tool dispatch -> observation path works end to end.
    assert result.hit_max_steps
    assert result.steps[0].action == "search_knowledge_base"
    assert result.steps[0].observation is not None
    assert "warranty" in result.steps[0].observation.lower()


def test_agent_parses_final_answer_and_stops():
    tool = make_search_tool(_sample_pipeline())
    canned = "FINAL_ANSWER: Shipping is free over $50."
    agent = Agent(llm=EchoLLMClient(canned_tool_call=canned), tools=[tool], max_steps=5)

    result = agent.run("Is shipping free?")
    assert result.final_answer == "Shipping is free over $50."
    assert not result.hit_max_steps
    assert len(result.steps) == 1  # stopped immediately, didn't waste steps


def test_sql_tool_rejects_non_select_statements():
    conn = build_demo_database()
    tool = make_sql_tool(conn)
    result = tool.func("DROP TABLE products")
    assert "only SELECT" in result

    # confirm the table really is still there
    cursor = conn.execute("SELECT COUNT(*) FROM products")
    assert cursor.fetchone()[0] == 4


def test_sql_tool_executes_valid_select():
    conn = build_demo_database()
    tool = make_sql_tool(conn)
    result = tool.func("SELECT name, stock FROM products WHERE stock = 0")
    assert "Gadget Mini" in result
