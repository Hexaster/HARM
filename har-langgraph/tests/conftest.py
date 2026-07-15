"""Shared fixtures for the HAR node/graph test suite.

The paper's worked example (p.8: a 68-year-old woman with stress
incontinence + overactive bladder) is the anchor case used throughout
Tasks 3-7. Its structured fields (y1-y9) come straight from the knowledge
base entry that test_kb.py already verifies against the paper; the raw
patient narrative below is a first-person reconstruction of those same
facts, since the paper shows the resulting note rather than the verbatim
intake question fed to the Information Collection Agent.
"""
import os

import dotenv
import pytest

dotenv.load_dotenv()

from har.knowledge import standardized_note

PAPER_CASE = standardized_note("stress incontinence")

PATIENT_QUESTION = (
    "I'm a 68-year-old woman. Lately I've been getting up three times a night "
    "to urinate, though it's normal during the day. I also leak urine when I "
    "cough, sneeze, or feel a sudden urge to go. I took Mirabegron for a month "
    "and it helped, but my symptoms came back within two days of stopping it. "
    "I had a coronary intervention two months ago. A urine test around that "
    "time showed a white blood cell count of 27.7 per high-power field, but a "
    "more recent test came back at 2.1."
)


def has_llm_credentials() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("OPENAI_BASE_URL"))


requires_llm = pytest.mark.skipif(
    not has_llm_credentials(),
    reason="OPENAI_API_KEY / OPENAI_BASE_URL not configured - skipping live LLM test",
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    """Stands in for har.llm.get_llm() so reflection-loop tests are
    deterministic and need no network access or API key. Returns each
    reply in order, then repeats the last one for any extra calls."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)

    def invoke(self, *_args, **_kwargs) -> _FakeMessage:
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        return _FakeMessage(reply)
