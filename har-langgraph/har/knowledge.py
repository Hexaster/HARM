"""Knowledge base loaders for the HAR pipeline: per-disease key inquiry
points, the differential-diagnosis graph, and standardized clinical notes
(paper Sec. 3.1 / 3.5, Fig. 3), covering the 17 urological diseases used
by the RJUA-QA evaluation.
"""
import difflib
import json
from functools import lru_cache
from pathlib import Path

_KB_DIR = Path(__file__).resolve().parent.parent / "kb"


def _load(filename: str) -> dict:
    with open(_KB_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _diseases() -> dict:
    return _load("diseases.json")


@lru_cache(maxsize=1)
def _differential_graph() -> dict:
    return _load("differential_graph.json")


@lru_cache(maxsize=1)
def _standardized_notes() -> dict:
    return _load("standardized_notes.json")


def all_diseases() -> list[str]:
    """All disease names known to the knowledge base."""
    return list(_diseases())


def _normalize(name: str) -> str:
    return " ".join(name.lower().split())


def _match_disease(name: str, known: dict) -> str:
    """Resolve a possibly loosely-phrased disease name (as an agent might
    write it) to the exact key used in the knowledge base."""
    target = _normalize(name)
    normalized_to_key = {_normalize(k): k for k in known}

    if target in normalized_to_key:
        return normalized_to_key[target]

    contains = [k for k in normalized_to_key if target in k or k in target]
    if contains:
        return normalized_to_key[max(contains, key=len)]

    close = difflib.get_close_matches(target, normalized_to_key.keys(), n=1, cutoff=0.6)
    if close:
        return normalized_to_key[close[0]]

    raise ValueError(f"No disease in the knowledge base matches {name!r}")


def key_points(disease: str) -> list[str]:
    """Diagnostic key inquiry points for a disease (paper Sec. 3.1)."""
    entries = _diseases()
    return entries[_match_disease(disease, entries)]["key_inquiry_points"]


def differentials(disease: str) -> list[str]:
    """Diseases requiring differential-diagnosis exclusion, from the
    knowledge graph (paper Fig. 3)."""
    graph = _differential_graph()
    return graph[_match_disease(disease, graph)]


def standardized_note(disease: str) -> dict:
    """The standardized clinical note template the Coordinator Agent
    compares a raw generated note against (paper Sec. 3.5)."""
    notes = _standardized_notes()
    return notes[_match_disease(disease, notes)]
