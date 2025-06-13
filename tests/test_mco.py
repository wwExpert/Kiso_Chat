import types
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mco import apply_mco


class DummyLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    def invoke(self, messages):
        return types.SimpleNamespace(content=self._response)


def test_apply_mco_success():
    llm = DummyLLM("better")
    assert apply_mco(llm, "test") == "better"


def test_apply_mco_without_llm():
    assert apply_mco(None, "test") == "test"
