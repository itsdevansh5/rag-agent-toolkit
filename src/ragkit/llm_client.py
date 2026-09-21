"""
llm_client.py — pluggable LLM interface.

WHY THIS EXISTS: the rest of this codebase (rag_pipeline.py, agent.py) is
written against the small `LLMClient` interface below, never against a
specific provider's SDK directly. That's what makes "continuously experiment
with new models, frameworks, and architectures" (this JD's words) cheap:
swapping GPT-4o for Claude for a local model is a one-line change at the
call site, not a rewrite.

THREE IMPLEMENTATIONS:

  - OpenAIClient / AnthropicClient: real SDK calls (openai, anthropic
    packages, both installed in this repo's environment). They need
    OPENAI_API_KEY / ANTHROPIC_API_KEY and network access to the provider --
    neither of which this sandbox has, so they're written correctly but not
    exercised here. Run them on your own machine with a real key.

  - EchoLLMClient: a deterministic, offline stand-in used by every test and
    demo script in this repo. It does NOT simulate intelligence -- it
    returns a templated response built from the prompt it was given, so
    tests can assert on structure (did the retrieved context actually reach
    the prompt? did the agent's tool-call format parse?) without needing a
    real model. This is a legitimate, common testing pattern for LLM-backed
    systems: you test the *plumbing* deterministically and reserve real API
    calls for a smaller set of end-to-end smoke tests run separately.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, prompt: str, max_tokens: int = 500) -> str:
        """Return the model's text completion for a system+user prompt pair."""


class OpenAIClient(LLMClient):
    """Real OpenAI SDK call. Needs OPENAI_API_KEY and network access."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def complete(self, system: str, prompt: str, max_tokens: int = 500) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    """Real Anthropic SDK call. Needs ANTHROPIC_API_KEY and network access."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def complete(self, system: str, prompt: str, max_tokens: int = 500) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))


class EchoLLMClient(LLMClient):
    """Deterministic offline stub -- exercised by every test/demo in this repo.

    Behavior, by design, is simple and inspectable:
      - If the prompt contains a recognizable tool-call request pattern
        (used by agent.py), it returns a canned tool-call in the expected
        format so the agent loop's PARSING logic can be tested without a
        real model deciding anything.
      - Otherwise, it returns a short extract of the context it was given,
        prefixed with a disclosure that this is a stub -- so it's never
        mistaken for a real generated answer in a demo transcript.
    """

    def __init__(self, canned_tool_call: str | None = None):
        self._canned_tool_call = canned_tool_call

    def complete(self, system: str, prompt: str, max_tokens: int = 500) -> str:
        if self._canned_tool_call is not None:
            return self._canned_tool_call

        context_match = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.DOTALL)
        context_snippet = context_match.group(1)[:200] if context_match else "(no context found)"
        return (
            "[EchoLLMClient stub -- not a real generated answer] "
            f"Based on retrieved context: {context_snippet.strip()}..."
        )
