"""LLM access with a mock fallback.

If ANTHROPIC_API_KEY is set, the agent reasons with Claude (claude-opus-4-8). If not,
a deterministic mock stands in so the whole graph still runs end-to-end for free — you
plug in a key when you want real reasoning. Same graph, same tools, either way.
"""
from __future__ import annotations

import os

MODEL = "claude-opus-4-8"


def get_mode() -> str:
    return "claude" if os.getenv("ANTHROPIC_API_KEY") else "mock"


def get_llm():
    """Return a LangChain chat model, or None in mock mode."""
    if get_mode() == "claude":
        from langchain_anthropic import ChatAnthropic

        # Adaptive thinking on; modest cap since each node is a focused step.
        return ChatAnthropic(model=MODEL, max_tokens=1500, timeout=60)
    return None


def complete(llm, system: str, user: str) -> str:
    """Single-shot completion. Only called when llm is not None (Claude mode)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content if isinstance(resp.content, str) else str(resp.content)
