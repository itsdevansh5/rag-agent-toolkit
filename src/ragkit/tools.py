"""
tools.py — concrete tools an agent can call.

WHY TOOLS ARE JUST PLAIN PYTHON CALLABLES WITH A NAME + DESCRIPTION:
This is deliberately framework-agnostic. LangChain, LlamaIndex, and custom
agent loops all converge on the same underlying shape: a tool is a name, a
natural-language description (which the LLM reads to decide *when* to use
it), and a function taking a string input and returning a string
observation. Once you understand this minimal shape, picking up any
specific agent framework's tool-decorator syntax is a small step -- the
concept transfers, the framework doesn't.

Two tools here, matching this JD's "integrate LLMs with APIs, databases, and
enterprise systems":
  - SearchKnowledgeBaseTool: wraps a RAGPipeline's retrieval step.
  - QueryDatabaseTool: wraps a toy SQLite database -- deliberately restricted
    to SELECT-only queries (see note below), the same "least privilege"
    instinct that showed up in the DevSecOps research paper's approach to
    tooling, applied here to keep an LLM-driven agent from being able to
    issue a destructive query against a real database by mistake.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable

from ragkit.rag_pipeline import RAGPipeline


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


def make_search_tool(pipeline: RAGPipeline, top_k: int = 3) -> Tool:
    def _run(query: str) -> str:
        results = pipeline.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant documents found."
        return "\n".join(f"[{r.chunk_id}] (score={r.score:.3f}) {r.text}" for r in results)

    return Tool(
        name="search_knowledge_base",
        description=(
            "Search the ingested document collection for information relevant "
            "to a natural-language query. Input: a search query string. "
            "Returns the most relevant text passages with their source IDs."
        ),
        func=_run,
    )


def make_sql_tool(conn: sqlite3.Connection) -> Tool:
    def _run(query: str) -> str:
        stripped = query.strip().rstrip(";")
        # Deliberately restrict this tool to SELECT-only queries. An
        # LLM-driven agent choosing its own SQL is a real prompt-injection /
        # accidental-mutation surface -- a malicious or malformed document
        # in the RAG corpus could otherwise trick the agent into emitting a
        # DROP TABLE. Least-privilege at the tool layer is cheap insurance.
        if not stripped.upper().startswith("SELECT"):
            return "Error: only SELECT queries are permitted through this tool."
        try:
            cursor = conn.execute(stripped)
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
            if not rows:
                return "Query returned no rows."
            lines = [", ".join(cols)]
            lines += [", ".join(str(v) for v in row) for row in rows]
            return "\n".join(lines)
        except sqlite3.Error as e:
            return f"SQL error: {e}"

    return Tool(
        name="query_database",
        description=(
            "Run a read-only SQL SELECT query against the enterprise database "
            "(table: products(id, name, price, stock)). Input: a valid SQL "
            "SELECT statement. Returns the result rows as text."
        ),
        func=_run,
    )


def build_demo_database() -> sqlite3.Connection:
    """An in-memory SQLite DB standing in for an 'enterprise system' the
    agent can query -- swap for a real connection (Postgres, a data
    warehouse, etc.) in a production setting; the tool interface above
    doesn't change."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL, stock INTEGER)")
    conn.executemany(
        "INSERT INTO products (name, price, stock) VALUES (?, ?, ?)",
        [
            ("Widget Pro", 49.99, 120),
            ("Widget Lite", 19.99, 340),
            ("Gadget Max", 129.99, 15),
            ("Gadget Mini", 59.99, 0),
        ],
    )
    conn.commit()
    return conn
