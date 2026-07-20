"""RJUA-QA data loader (plan Task 8).

The RJUA-QA release ships one JSON object per line with the fields
id / question / context / answer / disease / advice. Only the question and
the gold disease label matter for the diagnostic F1 in Task 9; `context` is
the retrieval passage the dataset authors supply, which HAR does not use
because it retrieves from its own knowledge base instead.

A record's `disease` field names one or more conditions separated by the
Chinese enumeration comma, e.g. "睾丸炎、睾丸扭转，脓毒血症", so gold is a set.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rjua_qa"

_SPLIT_FILES = {
    "train": "chat_train.json",
    "valid": "chat_valid.json",
    "test": "chat_test.json",
}

# The gold label separates co-occurring diseases with Chinese or ASCII commas.
_DISEASE_SEPARATORS = re.compile(r"[、,，;；]+")


def split_diseases(label: str) -> list[str]:
    """Split a gold `disease` field into individual disease names."""
    return [part.strip() for part in _DISEASE_SEPARATORS.split(label) if part.strip()]


def load_rjua(split: str = "test", n: int | None = None) -> Iterator[dict]:
    """Yield `{"id", "question", "gold_diseases"}` for one RJUA-QA split.

    `n` caps how many records are produced, since a full sweep runs the whole
    multi-agent graph once per question.
    """
    try:
        filename = _SPLIT_FILES[split]
    except KeyError:
        raise ValueError(
            f"unknown split {split!r}; expected one of {sorted(_SPLIT_FILES)}"
        ) from None

    path = _DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Download the RJUA-QA release and place the "
            f"three chat_*.json splits under {_DATA_DIR}."
        )

    produced = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if n is not None and produced >= n:
                return
            if not line.strip():
                continue
            record = json.loads(line)
            produced += 1
            yield {
                "id": record["id"],
                "question": record["question"],
                "gold_diseases": split_diseases(record["disease"]),
            }
