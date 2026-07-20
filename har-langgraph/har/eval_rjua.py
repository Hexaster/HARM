"""RJUA-QA diagnostic F1 evaluation (plan Task 9 - the paper's Table 1 metric).

Runs the compiled HAR graph over an RJUA-QA subset, reduces each generated
clinical note to the set of diseases it names, and scores that against the
gold disease set.

Predictions are compared against the gold disease strings directly, so the
score does not depend on which diseases the knowledge base happens to hold.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from har.graph import build_har_graph
from har.llm import get_llm
from har.rjua import load_rjua, split_diseases

_BASELINE_PROMPT = """You are an experienced clinical diagnosis expert.
Read the patient's question and name the diagnosed disease or diseases.
Answer with the disease names only, separated by commas, and nothing else.

Below is the patient's question:
{question}
"""

# Qualifiers that carry no diagnostic content on their own, stripped before
# comparing a predicted name against a gold one.
_NOISE = re.compile(
    r"(可能|考虑|疑似|待排|不排除|建议|initial|final|diagnosis|suspected|probable|possible)"
)


def f1_set(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _normalize(term: str) -> str:
    return _NOISE.sub("", term).strip().lower()


def disease_terms(text: str) -> list[str]:
    """Reduce a free-text diagnosis to the individual disease names in it."""
    if not text:
        return []
    terms = (_normalize(t) for t in split_diseases(str(text)))
    return [t for t in terms if t]


def _same_disease(gold: str, pred: str, overlap: float = 0.6) -> bool:
    """Whether a predicted disease name denotes the gold one.

    Clinical free text qualifies a diagnosis rather than repeating the gold
    label verbatim: "右输尿管结石" comes back as "右侧输尿管结石伴梗阻性肾病",
    which shares the whole stem but is not a substring either way. Matching on
    the longest shared run of characters, as a fraction of the shorter name,
    accepts that while still rejecting unrelated diseases.

    This is deliberately a lexical test, so it under-counts diagnoses that are
    equivalent but share no characters ("急性肾损伤" for "肾功能不全"). Scores
    from it are therefore a lower bound on true diagnostic agreement.
    """
    if not gold or not pred:
        return False
    if gold in pred or pred in gold:
        return True
    shared = difflib.SequenceMatcher(None, gold, pred).find_longest_match(
        0, len(gold), 0, len(pred)
    ).size
    return shared >= 2 and shared / min(len(gold), len(pred)) >= overlap


def score_item(pred_text: str, gold_diseases: list[str]) -> tuple[float, float, float]:
    """Set P/R/F1 for one question."""
    preds = set(disease_terms(pred_text))
    golds = {_normalize(g) for g in gold_diseases} - {""}
    if not preds or not golds:
        return 0.0, 0.0, 0.0

    matched_gold = {g for g in golds if any(_same_disease(g, p) for p in preds)}
    matched_pred = {p for p in preds if any(_same_disease(g, p) for g in golds)}

    p = len(matched_pred) / len(preds)
    r = len(matched_gold) / len(golds)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _run_har(question: str) -> str:
    result = build_har_graph().invoke({"question": question})
    return result.get("final_diagnosis") or ""


def _run_baseline(question: str) -> str:
    return get_llm().invoke(_BASELINE_PROMPT.format(question=question)).content


def _evaluate_one(record: dict, baseline: bool) -> dict:
    runner = _run_baseline if baseline else _run_har
    started = time.time()
    try:
        prediction = runner(record["question"])
        error = None
    except Exception as exc:  # one bad question must not abort the sweep
        prediction, error = "", f"{type(exc).__name__}: {exc}"

    p, r, f = score_item(prediction, record["gold_diseases"])
    return {
        "id": record["id"],
        "question": record["question"],
        "gold": record["gold_diseases"],
        "prediction": prediction,
        "precision": p,
        "recall": r,
        "f1": f,
        "seconds": round(time.time() - started, 1),
        "error": error,
    }


def _aggregate(items: list[dict]) -> dict:
    """Micro- (pooled over all diseases) and macro- (mean over items) scores."""
    if not items:
        return {}
    tp = sum(i["recall"] * len({_normalize(g) for g in i["gold"]}) for i in items)
    n_gold = sum(len({_normalize(g) for g in i["gold"]}) for i in items)
    n_pred = sum(len(set(disease_terms(i["prediction"]))) for i in items)
    micro_p = tp / n_pred if n_pred else 0.0
    micro_r = tp / n_gold if n_gold else 0.0
    micro_f = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    )
    n = len(items)
    return {
        "n": n,
        "micro": {
            "precision": round(micro_p, 4),
            "recall": round(micro_r, 4),
            "f1": round(micro_f, 4),
        },
        "macro": {
            "precision": round(sum(i["precision"] for i in items) / n, 4),
            "recall": round(sum(i["recall"] for i in items) / n, 4),
            "f1": round(sum(i["f1"] for i in items) / n, 4),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score HAR on an RJUA-QA subset with set-based P/R/F1."
    )
    parser.add_argument("--n", type=int, default=50, help="questions to evaluate")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="single direct LLM call instead of the HAR graph (see plan section 7)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="questions evaluated concurrently"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval_rjua_predictions.json"),
        help="where to write the per-question prediction log",
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)

    records = list(load_rjua(args.split, args.n))
    if not records:
        print("no records to evaluate", file=sys.stderr)
        return 1

    label = "baseline" if args.baseline else "HAR"
    print(f"Evaluating {label} on {len(records)} {args.split} questions...", flush=True)

    order = {record["id"]: index for index, record in enumerate(records)}
    items: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_evaluate_one, record, args.baseline) for record in records]
        for done, future in enumerate(as_completed(futures), start=1):
            item = future.result()
            items.append(item)
            flag = "!" if item["error"] else " "
            print(
                f"{flag}[{done}/{len(records)}] id={item['id']} "
                f"F1={item['f1']:.2f} {item['seconds']}s "
                f"gold={'/'.join(item['gold'])[:40]}",
                flush=True,
            )

    items.sort(key=lambda item: order[item["id"]])
    summary = _aggregate(items)
    failures = [i for i in items if i["error"]]

    report = {
        "config": {"split": args.split, "n": len(items), "mode": label},
        "overall": summary,
        "failures": len(failures),
        "items": items,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    micro, macro = summary["micro"], summary["macro"]
    print(f"\n=== {label} on {len(items)} {args.split} questions ===")
    print(
        f"micro P/R/F1 = {micro['precision']:.3f}/{micro['recall']:.3f}/{micro['f1']:.3f}   "
        f"macro P/R/F1 = {macro['precision']:.3f}/{macro['recall']:.3f}/{macro['f1']:.3f}"
    )
    if failures:
        print(f"{len(failures)} question(s) errored; see {args.out}")
    print(f"predictions written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
