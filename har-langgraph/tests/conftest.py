"""Shared test doubles for HAR node tests.

DeepSeek-R1/Qwen are reached only through the OpenAI-compatible API, so unit
tests never call a live endpoint: FakeLLM stands in for the ChatOpenAI
instance `get_llm()` returns.
"""
from types import SimpleNamespace


class FakeLLM:
    """Returns scripted replies in call order and records every prompt sent.

    Raises if a node makes more LLM calls than the test scripted, so tests
    double as an assertion on exactly how many LLM round-trips a code path
    takes (e.g. that the Coordinator short-circuits after the first failing
    check instead of always running all three).
    """

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[str] = []

    def invoke(self, msg):
        self.calls.append(msg)
        if not self._replies:
            raise AssertionError(
                f"FakeLLM received an unscripted call #{len(self.calls)}: {msg[:200]!r}"
            )
        return SimpleNamespace(content=self._replies.pop(0))
