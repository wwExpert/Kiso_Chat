"""Message Correction and Optimization utilities."""

from __future__ import annotations

from langchain_core.messages import HumanMessage


PROMPT_TEMPLATE = (
    "Improve the following user message for a conversation with a large "
    "language model. Correct grammar and rephrase concisely."
)


def apply_mco(llm: object | None, message: str) -> str:
    """Return an optimized version of *message* using *llm* if available."""
    if not llm or not message:
        return message

    try:
        prompt = f"{PROMPT_TEMPLATE}\nUser: {message}"
        result = llm.invoke([HumanMessage(content=prompt)])
        return result.content.strip() if hasattr(result, "content") else message
    except Exception:
        return message
