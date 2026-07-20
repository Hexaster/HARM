"""Automated note-quality scoring (plan Task 10; paper Fig. 5).

An LLM grades each of the eight note sections out of 5, giving a total out
of 40. The paper uses this to filter generated notes before they are used as
training data, keeping those at or above NOTE_SCORE_THRESHOLD; it reports an
expert average of roughly 25.3 for reference.

This is independent of the diagnostic F1 in eval_rjua.py: F1 asks whether
HAR reached the right disease, this asks whether the note it wrote is well
formed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import field_validator

from har.config import Config
from har.graph import build_har_graph
from har.llm import LLMResponse, get_llm, parse_json
from har.rjua import load_rjua

# The eight graded sections. The nine-field note collapses diseases_list and
# differential_diagnosis_process into one "differential diagnosis" judgement,
# matching the paper's 8 x 5 = 40 point scale.
SECTIONS = (
    "medical_history",
    "physical_examination",
    "auxiliary_examination",
    "clinical_features",
    "initial_diagnosis",
    "diagnostic_basis",
    "differential_diagnosis",
    "final_diagnosis",
)

MAX_TOTAL = 5 * len(SECTIONS)

SCORE_NOTE_PROMPT = """You are an experienced clinical records reviewer.
Score the clinical note below section by section on a 0-5 integer scale,
where 0 means absent or wrong and 5 means complete, accurate, and clinically
well reasoned. Judge only what the note contains; do not reward speculation.

Output only one JSON object with exactly these integer fields:
{{
  "medical_history": Int,
  "physical_examination": Int,
  "auxiliary_examination": Int,
  "clinical_features": Int,
  "initial_diagnosis": Int,
  "diagnostic_basis": Int,
  "differential_diagnosis": Int,
  "final_diagnosis": Int
}}

Below is the clinical note:
{clinical_note}
"""


class NoteScores(LLMResponse):
    medical_history: int
    physical_examination: int
    auxiliary_examination: int
    clinical_features: int
    initial_diagnosis: int
    diagnostic_basis: int
    differential_diagnosis: int
    final_diagnosis: int

    @field_validator("*", mode="before")
    @classmethod
    def _clamp(cls, value):
        try:
            return max(0, min(5, int(float(value))))
        except (TypeError, ValueError):
            return 0


def score_note(clinical_note: str) -> dict:
    """Grade one assembled note, returning per-section scores and the total."""
    reply = get_llm().invoke(
        SCORE_NOTE_PROMPT.format(clinical_note=clinical_note)
    ).content
    scores = parse_json(NoteScores, reply).model_dump()
    return {"sections": scores, "total": sum(scores.values())}


def _score_one(record: dict) -> dict:
    try:
        state = build_har_graph().invoke({"question": record["question"]})
        note = state.get("clinical_note") or ""
        if not note:
            raise ValueError("graph produced no clinical note")
        result = score_note(note)
        error = None
    except Exception as exc:
        result, error = {"sections": {}, "total": 0}, f"{type(exc).__name__}: {exc}"

    return {
        "id": record["id"],
        "gold": record["gold_diseases"],
        "error": error,
        "kept": error is None and result["total"] >= Config.NOTE_SCORE_THRESHOLD,
        **result,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate notes and score their quality out of 40."
    )
    parser.add_argument("--n", type=int, default=20, help="notes to generate and score")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("note_scores.json"))
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    records = list(load_rjua(args.split, args.n))
    if not records:
        print("no records to score", file=sys.stderr)
        return 1

    print(f"Scoring {len(records)} generated notes (max {MAX_TOTAL})...", flush=True)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_score_one, record) for record in records]
        for done, future in enumerate(as_completed(futures), start=1):
            item = future.result()
            results.append(item)
            flag = "!" if item["error"] else " "
            print(
                f"{flag}[{done}/{len(records)}] id={item['id']} total={item['total']}/{MAX_TOTAL}",
                flush=True,
            )

    scored = [r for r in results if not r["error"]]
    totals = [r["total"] for r in scored]
    summary = {
        "n": len(results),
        "scored": len(scored),
        "failed": len(results) - len(scored),
        "threshold": Config.NOTE_SCORE_THRESHOLD,
        "kept": sum(1 for r in results if r["kept"]),
        "mean_total": round(statistics.mean(totals), 2) if totals else 0.0,
        "median_total": round(statistics.median(totals), 2) if totals else 0.0,
        "min_total": min(totals) if totals else 0,
        "max_total": max(totals) if totals else 0,
    }
    args.out.write_text(
        json.dumps({"summary": summary, "notes": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n=== note quality over {summary['scored']} notes (max {MAX_TOTAL}) ===")
    print(
        f"mean {summary['mean_total']}  median {summary['median_total']}  "
        f"range {summary['min_total']}-{summary['max_total']}  "
        f"(paper's expert average is about 25.3)"
    )
    print(f"kept at threshold {summary['threshold']}: {summary['kept']}/{summary['n']}")
    if summary["failed"]:
        print(f"{summary['failed']} note(s) errored; see {args.out}")
    print(f"scores written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
