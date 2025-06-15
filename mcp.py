"""Message Cleaning and Polishing utilities."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langchain.agents import Tool

MCP_PROMPT = (
    "Rewrite the following text to be clear, concise and professional. "
    "Fix grammar and typos. Return only the revised text."
)


def apply_mcp(llm: Any | None, message: str) -> str:
    """Return a polished version of *message* using *llm* if provided."""
    if not llm or not message:
        return message

    try:
        prompt = f"{MCP_PROMPT}\nUser: {message}"
        result = llm.invoke([HumanMessage(content=prompt)])
        if hasattr(result, "content") and result.content:
            return str(result.content).strip()
    except Exception:
        pass
    return message


def get_mcp_tool(llm: Any) -> Tool:
    """Return a LangChain tool wrapping :func:`apply_mcp`."""

    def _run(text: str) -> str:
        return apply_mcp(llm, text)

    return Tool(
        name="mcp_tool",
        func=_run,
        description="Clean and polish text using MCP",
    )
