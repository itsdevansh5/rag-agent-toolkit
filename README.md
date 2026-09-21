# RAG + Tool-Calling Agent Toolkit (Python)

A small, real RAG pipeline and a minimal ReAct-style tool-calling agent,
built to demonstrate the actual mechanics behind "RAG pipelines," "vector
databases," "multi-agent orchestration," "LLM integration with APIs and
databases," and "prompt evaluation" — not to be a production system.

```
Documents ──chunk──► Chunks ──embed──► SimpleVectorStore
                                              │
User question ──embed──► query vector ───────┤
                                              ▼
                                     top-k similar chunks
                                              │
                                              ▼
                              prompt = context + question
                                              │
                                              ▼
                                       LLMClient.complete()
                                              │
                                              ▼
                                           Answer

Agent loop:  question → [tools available] → LLM picks ACTION or FINAL_ANSWER
             → tool runs → observation fed back → repeat until final answer
             Tools here: search_knowledge_base (wraps the RAG pipeline above)
                          query_database (SELECT-only SQLite, an "enterprise system")
```

## Honesty note, up front

This was built in a sandboxed environment **with no network access to
OpenAI/Anthropic APIs or HuggingFace model downloads, and no API keys
configured.** So the design deliberately separates what's provider-agnostic
plumbing (chunking, retrieval ranking, the agent's parse/dispatch loop,
evaluation) from what needs a real model:

- **Fully implemented, tested, and run in this sandbox:** chunking, TF-IDF
  embedding + cosine-similarity retrieval, the vector store, the RAG
  prompt-assembly pipeline, the agent's ACTION/FINAL_ANSWER parsing and
  tool-dispatch loop, the SQL safety guard, and a retrieval-quality
  evaluation harness (100% hit-rate on the demo query set — see below).
- **Written correctly, using the real `openai` and `anthropic` SDKs, but
  NOT executed here** — `OpenAIClient` / `AnthropicClient` in
  `llm_client.py` need `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` and network
  access. Run these on your own machine to see real model-driven answers
  and real agent tool-selection (vs. this sandbox's scripted/echo stand-ins).

Don't claim "built a production RAG system with GPT-4" from this — claim
"designed and implemented a RAG pipeline and tool-calling agent with a
provider-agnostic architecture, verified via an offline test suite, ready to
run against a real LLM." That's accurate, and it's still a real, defensible
answer to "tell me about a RAG project you've built."

## Run it

```bash
pip install -r requirements.txt

# Fully offline — runs right now, no API key needed
PYTHONPATH=src python3 -m pytest tests/ -v      # 13 tests, all offline
PYTHONPATH=src python3 scripts/demo_rag.py      # RAG retrieval + eval demo
PYTHONPATH=src python3 scripts/demo_agent.py    # agent tool-routing demo

# Needs a real key + network (not available in this sandbox):
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY
# then swap EchoLLMClient()/ScriptedLLMClient() for AnthropicClient()/OpenAIClient()
# in demo_rag.py / demo_agent.py
```

## A real bug this caught, worth knowing for an interview

The first version of `RAGPipeline.ingest()` embedded each document
immediately as it arrived. With TF-IDF, whose vector dimensionality **is**
the corpus vocabulary size, that meant document 1 got embedded into a
smaller space than document 2 once the vocabulary grew — and the vector
store crashed trying to stack vectors of different width. The test suite
caught it immediately as a shape-mismatch error. The fix: split `ingest()`
(collect + chunk) from `build_index()` (fit the embedder on the *whole*
corpus, then embed everything in one batch) — which is also how real batch
RAG indexing pipelines are structured, not just a sandbox workaround. This
is a genuinely good story to tell in an interview: it shows you test your
own retrieval pipeline and understand *why* embedding-space consistency
matters, not just that you called an embedding API.

## What's simplified, and the honest "next step" for each

| This project | Production version |
|---|---|
| TF-IDF (lexical/keyword-overlap embeddings) | Neural embeddings (OpenAI `text-embedding-3`, or local `sentence-transformers`) — captures semantic similarity, not just word overlap. See `NeuralEmbedder` in `embeddings.py` for the exact swap-in point. |
| `SimpleVectorStore` (brute-force cosine similarity, O(n) per query) | FAISS/Chroma/Pinecone with an ANN index (HNSW/IVF) — sublinear query time at scale, trading a little recall for a lot of speed. |
| Fixed word-count chunking | Sentence/paragraph-boundary-aware chunking, or token-count-based chunking matched to the actual embedding model's tokenizer. |
| Agent: single loop, one model, strict text-format parsing | A real framework (LangGraph, AutoGen, CrewAI) for multi-agent graphs, structured function-calling (JSON schema, not text parsing), and built-in retry/guardrails. |
| Retrieval-only evaluation (hit-rate@k) | Add generation-quality eval (LLM-as-judge or human-labeled answer correctness) once a real LLM is in the loop — retrieval and generation are evaluated separately here on purpose (see `eval.py` docstring for why). |

## JD-to-code mapping

- *"Build and optimize RAG pipelines"* → `rag_pipeline.py`, `chunking.py`,
  the chunk-size/overlap trade-off discussion in the docstrings, and the
  real bug + fix above.
- *"Work with vector databases (FAISS, Pinecone, Chroma)"* → `vectorstore.py`
  implements the same add/query-top-k contract those systems expose, plus
  the brute-force-vs-ANN-index trade-off explained in its docstring.
- *"Develop multi-agent systems and orchestration workflows"* →
  `agent.py`'s ReAct loop is the single-agent building block those
  frameworks compose; `demo_agent.py` shows multi-step, multi-tool chaining.
- *"Integrate LLMs with APIs, databases, and enterprise systems"* →
  `tools.py`'s `query_database` tool (a real SQLite connection, SELECT-only
  for safety) and `llm_client.py`'s real OpenAI/Anthropic SDK wrappers.
- *"Implement prompt engineering, evaluation, and optimization techniques"*
  → `rag_pipeline.py`'s `SYSTEM_PROMPT`/`build_prompt()`, and `eval.py`'s
  hit-rate@k harness with a real, run, 100%-on-demo-data result.
