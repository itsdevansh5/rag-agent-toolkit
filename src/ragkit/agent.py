"""
agent.py — a minimal ReAct-style tool-calling agent loop.

WHAT "MULTI-AGENT ORCHESTRATION" ACTUALLY REDUCES TO, AT THE FOUNDATION:
Before multi-agent, there's single-agent: a loop where an LLM is given a
question, a list of tools it may call, and repeatedly asked "what do you
want to do next" until it decides it has enough information to answer.
This is the ReAct pattern (Reason + Act), and it's the building block every
multi-agent framework (LangGraph, AutoGen, CrewAI) composes into graphs of
multiple such loops talking to each other. Understanding this single-agent
loop cold -- the prompt format, the parse step, the observation feedback --
is what makes a multi-agent framework's abstractions legible instead of
magic.

THE LOOP, CONCRETELY:
  1. Build a prompt: the question, the tools available (name + description),
     and the running transcript of past (action, observation) pairs.
  2. Ask the LLM to respond in a strict, parseable format: either
     "ACTION: <tool_name>\nACTION_INPUT: <input>" or "FINAL_ANSWER: <text>".
  3. Parse the response. If it's an action, call the named tool with the
     given input, append the tool's output as an "Observation" to the
     transcript, and go back to step 1. If it's a final answer, stop.
  4. Cap iterations (max_steps) -- an ungrounded model can loop forever
     without one; this is a real production concern, not a toy detail.

WHY THE PARSER IS STRICT, WITH A GRACEFUL FALLBACK:
An LLM asked to "call a tool" doesn't literally call a function -- it emits
text, and something has to parse that text back into a real function call.
If parsing fails (the model didn't follow the format), this implementation
falls back to treating the raw response as the final answer rather than
crashing -- a deliberate reliability choice: a slightly-off answer beats an
unhandled exception in a system meant to run against untrusted, imperfect
model output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragkit.llm_client import LLMClient
from ragkit.tools import Tool

AGENT_SYSTEM_PROMPT = """You are an assistant that can use tools to answer questions.

Available tools:
{tool_descriptions}

Respond in EXACTLY one of these two formats, nothing else:

ACTION: <tool name>
ACTION_INPUT: <input to the tool>

OR, once you have enough information to answer:

FINAL_ANSWER: <your answer to the user>
"""

_ACTION_RE = re.compile(r"ACTION:\s*(.+?)\s*\nACTION_INPUT:\s*(.+?)\s*$", re.DOTALL)
_FINAL_RE = re.compile(r"FINAL_ANSWER:\s*(.+)$", re.DOTALL)


@dataclass
class AgentStep:
    thought_raw: str
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


@dataclass
class AgentResult:
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)
    hit_max_steps: bool = False


class Agent:
    def __init__(self, llm: LLMClient, tools: list[Tool], max_steps: int = 5):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def _tool_descriptions(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())

    def run(self, question: str) -> AgentResult:
        system = AGENT_SYSTEM_PROMPT.format(tool_descriptions=self._tool_descriptions())
        transcript = f"Question: {question}\n"
        steps: list[AgentStep] = []

        for _ in range(self.max_steps):
            response = self.llm.complete(system, transcript)
            step = AgentStep(thought_raw=response)

            final_match = _FINAL_RE.search(response)
            if final_match:
                answer = final_match.group(1).strip()
                steps.append(step)
                return AgentResult(final_answer=answer, steps=steps)

            action_match = _ACTION_RE.search(response)
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_input = action_match.group(2).strip()
                step.action = tool_name
                step.action_input = tool_input

                if tool_name not in self.tools:
                    observation = (
                        f"Error: unknown tool '{tool_name}'. "
                        f"Available tools: {', '.join(self.tools)}"
                    )
                else:
                    observation = self.tools[tool_name].func(tool_input)

                step.observation = observation
                steps.append(step)
                transcript += (
                    f"\nACTION: {tool_name}\nACTION_INPUT: {tool_input}\n"
                    f"Observation: {observation}\n"
                )
                continue

            # Parsing failed -- graceful fallback, not a crash. See module
            # docstring for why this is the deliberate choice.
            steps.append(step)
            return AgentResult(final_answer=response.strip(), steps=steps)

        steps_summary = steps[-1].observation if steps else "(no steps taken)"
        return AgentResult(
            final_answer=f"Stopped after {self.max_steps} steps without a final answer. "
            f"Last observation: {steps_summary}",
            steps=steps,
            hit_max_steps=True,
        )
