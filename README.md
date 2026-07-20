# HARM — Hierarchical Agent Reflection

A reimplementation of the multi-agent clinical-note generation pipeline from:

> Wang, X., Li, X., Zhao, D., Feng, K., Liang, L., Zhang, Z., Ding, K., Chen, H., Wan, B., & Zhang, Q.
> **"Hierarchical agent reflection for aligning LLM reasoning with clinical diagnostic processes."**
> *Health Information Science and Systems* 14:21 (2026). https://doi.org/10.1007/s13755-025-00410-1

The paper (`s13755-025-00410-1.pdf`, included in this repo) proposes **Hierarchical Agent
Reflection (HAR)**: instead of asking an LLM to produce a diagnosis in one shot, HAR
decomposes the diagnostic process into stages handled by specialized agents, each of
which iteratively drafts and self-reflects on its output, with a supervisory agent
checking the assembled note against a standardized template and routing corrections
back to whichever stage introduced the error. The resulting structured clinical notes
are intended for use as training data so a downstream LLM learns to reason the way a
clinician does, rather than just emit a final answer.

## Architecture

HAR is implemented here as a [LangGraph](https://github.com/langchain-ai/langgraph)
state machine (`har-langgraph/har/graph.py`) with four agents, matching the paper's
Fig. 2 framework:

1. **ICA — Information Collection Agent** (`har/nodes/ica.py`)
   Extracts medical history, physical examination, and auxiliary examination from the
   raw patient question, then summarizes them into clinical features.
2. **PDA — Preliminary Diagnosis Agent** (`har/nodes/pda.py`)
   Iteratively drafts and reflects on an initial diagnostic hypothesis with supporting
   basis, up to `Config.PDA_MAX_ITERS` rounds.
3. **DDA — Differential Diagnosis Agent** (`har/nodes/dda.py`)
   Retrieves candidate diseases and differential-diagnosis criteria from the knowledge
   base, then iteratively refines the differential diagnosis and reflects on it, up to
   `Config.DDA_MAX_ITERS` rounds.
4. **Coordinator Agent** (`har/nodes/coordinator.py`)
   The supervisory agent. Matches the final diagnosis against a standardized note
   template (`kb/standardized_notes.json`), checks the assembled note for
   inconsistencies attributable to ICA, PDA, or DDA, and routes the graph back to the
   offending stage (up to `Config.CA_MAX_ITERS` times) before assembling the final
   clinical note.

State flows through a single `ClinicalNoteState` (`har/state.py`) that accumulates the
nine note fields (`y1`–`y9` in the paper's notation) plus iteration counters and
feedback fields used for the reflection/routing loops.

Supporting modules:

- `har/knowledge.py` — loads the knowledge base (`kb/diseases.json`,
  `kb/differential_graph.json`, `kb/standardized_notes.json`) covering 17 urological
  diseases, with fuzzy disease-name matching.
- `har/llm.py` — LLM client wrapper (OpenAI-compatible API) and structured-output
  parsing helpers.
- `har/prompts.py` — the prompt templates for each agent stage.
- `har/run_graph.py` — CLI entry point to run the pipeline on a single patient
  question.
- `har/eval_rjua.py` — reproduces the paper's diagnostic-F1 evaluation (Table 1) on
  the RJUA-QA benchmark.
- `har/score_notes.py` — reproduces the paper's note-quality scoring (Fig. 5), an LLM
  grading each note section out of 5 (40 total) for use as a training-data filter.
- `har/rjua.py` — RJUA-QA dataset loader.

## Repository layout

```
har-langgraph/
├── har/                  # library + CLI code (see above)
├── kb/                   # knowledge base (diseases, differential graph, standardized notes)
├── data/rjua_qa/         # RJUA-QA benchmark splits (chat_train/valid/test.json)
├── tests/                # pytest suite for each agent and the end-to-end graph
└── pyproject.toml
```

## Setup

```bash
cd har-langgraph
pip install -e ".[dev]"
```

Configure LLM access via environment variables (see `.env.example`):

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...   # OpenAI-compatible endpoint, e.g. DeepSeek-R1 or a Qwen instruct model
```

Model and iteration-budget defaults live in `har/config.py` (`Config.MODEL_ID`,
`PDA_MAX_ITERS`, `DDA_MAX_ITERS`, `CA_MAX_ITERS`, `NOTE_SCORE_THRESHOLD`).

## Usage

Generate a clinical note for one patient question:

```bash
python -m har.run_graph "患者主诉..."
```

Run the RJUA-QA diagnostic-F1 evaluation:

```bash
python -m har.eval_rjua --split test --n 50
```

Score generated notes for quality:

```bash
python -m har.score_notes
```

Run the test suite:

```bash
pytest
```

## Status

This is a from-scratch reimplementation of the HAR framework as described in the
paper, built around LangGraph for orchestration and evaluated against the same
RJUA-QA benchmark the paper uses. It is not the authors' original code.
