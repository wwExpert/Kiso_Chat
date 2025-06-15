import types
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mcp import apply_mcp


class DummyLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def invoke(self, messages):
        return types.SimpleNamespace(content=self._response)


def test_apply_mcp_success():
    llm = DummyLLM("clean")
    assert apply_mcp(llm, "test") == "clean"


def test_apply_mcp_without_llm():
    assert apply_mcp(None, "test") == "test"
