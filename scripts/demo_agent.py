#!/usr/bin/env python3
"""
Demo: a tool-calling agent choosing between two tools -- a RAG search over
policy documents, and a SQL query tool over a toy 'enterprise' database.

IMPORTANT HONESTY NOTE: this script uses EchoLLMClient with a pre-scripted
sequence of canned responses to drive the agent loop, because this sandbox
has no access to a real LLM API. This demonstrates that the agent's
PARSING, DISPATCH, and OBSERVATION-FEEDBACK machinery works correctly --
it does NOT demonstrate an LLM actually deciding which tool to use, since
no real model is in the loop here. That's the one piece of this repo that
needs a real API key and network access to see for real: swap
EchoLLMClient for OpenAIClient()/AnthropicClient() on your own machine and
the exact same Agent/Tool code will work with real model-driven decisions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragkit.agent import Agent
from ragkit.embeddings import TfidfEmbedder
from ragkit.llm_client import EchoLLMClient, LLMClient
from ragkit.rag_pipeline import RAGPipeline
from ragkit.tools import build_demo_database, make_search_tool, make_sql_tool

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


class ScriptedLLMClient(LLMClient):
    """Returns a pre-set sequence of responses, one per call -- simulates a
    multi-step agent trajectory without a real model, purely to demonstrate
    the loop mechanics end to end. See module docstring."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._i = 0

    def complete(self, system: str, prompt: str, max_tokens: int = 500) -> str:
        if self._i >= len(self._responses):
            return "FINAL_ANSWER: (scripted responses exhausted)"
        resp = self._responses[self._i]
        self._i += 1
        return resp


def build_pipeline() -> RAGPipeline:
    pipeline = RAGPipeline(embedder=TfidfEmbedder(), llm=EchoLLMClient())
    for path in sorted(DATA_DIR.glob("*.txt")):
        pipeline.ingest(doc_id=path.stem, text=path.read_text())
    pipeline.build_index()
    return pipeline


def main():
    pipeline = build_pipeline()
    conn = build_demo_database()
    tools = [make_search_tool(pipeline), make_sql_tool(conn)]

    print("=" * 70)
    print("SCENARIO 1: question answerable from the SQL tool")
    print("=" * 70)
    scripted = ScriptedLLMClient([
        "ACTION: query_database\nACTION_INPUT: SELECT name, stock FROM products WHERE stock = 0",
        "FINAL_ANSWER: Gadget Mini is currently out of stock.",
    ])
    agent = Agent(llm=scripted, tools=tools, max_steps=4)
    result = agent.run("Which product is out of stock?")
    for i, step in enumerate(result.steps, 1):
        print(f"Step {i}: action={step.action!r} observation={step.observation!r}")
    print(f"Final answer: {result.final_answer}\n")

    print("=" * 70)
    print("SCENARIO 2: question requiring BOTH tools (RAG then SQL)")
    print("=" * 70)
    scripted2 = ScriptedLLMClient([
        "ACTION: search_knowledge_base\nACTION_INPUT: warranty length by product",
        "ACTION: query_database\nACTION_INPUT: SELECT name, price FROM products WHERE name = 'Gadget Max'",
        "FINAL_ANSWER: Gadget Max has an extended 2-year warranty and is priced at $129.99.",
    ])
    agent2 = Agent(llm=scripted2, tools=tools, max_steps=4)
    result2 = agent2.run("What's the warranty and price on Gadget Max?")
    for i, step in enumerate(result2.steps, 1):
        print(f"Step {i}: action={step.action!r}")
        print(f"         observation={step.observation!r}")
    print(f"Final answer: {result2.final_answer}\n")

    print("=" * 70)
    print("SCENARIO 3: the SQL tool refuses an unsafe instruction")
    print("=" * 70)
    scripted3 = ScriptedLLMClient([
        "ACTION: query_database\nACTION_INPUT: DROP TABLE products",
        "FINAL_ANSWER: I was not able to modify the database -- that action is restricted.",
    ])
    agent3 = Agent(llm=scripted3, tools=tools, max_steps=4)
    result3 = agent3.run("Delete the products table.")
    for i, step in enumerate(result3.steps, 1):
        print(f"Step {i}: action={step.action!r} observation={step.observation!r}")
    print(f"Final answer: {result3.final_answer}")


if __name__ == "__main__":
    main()
