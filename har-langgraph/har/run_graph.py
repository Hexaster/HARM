"""Command-line entry point for running the HAR clinical-note graph."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from har.graph import build_har_graph


def run_graph(question: str) -> dict[str, Any]:
    """Run the HAR pipeline for one patient question and return its state."""
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    graph = build_har_graph()
    return dict(graph.invoke({"question": question}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a structured clinical note with the HAR pipeline."
    )
    parser.add_argument(
        "question",
        nargs="+",
        help="patient narrative or clinical question",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    args = _parser().parse_args(argv)
    result = run_graph(" ".join(args.question))

    try:
        clinical_note = result["clinical_note"]
    except KeyError as exc:
        raise RuntimeError("HAR graph did not produce a clinical note") from exc

    print(clinical_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
