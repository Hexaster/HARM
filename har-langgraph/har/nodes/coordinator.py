"""Coordinator Agent: match the final diagnosis to its standardized note,
compare the raw note against it section by section, and route control back
to the offending agent (ICA/PDA/DDA) with feedback -- the hierarchical
supervisory correction on top of the two inner reflection loops (paper
Sec. 3.2 / 3.5, Algorithm 1).
"""
from dataclasses import dataclass

from pydantic import BaseModel, Field

from har import config, prompts
from har.knowledge import standardized_note
from har.llm import get_llm, parse_json

_NOTE_FIELD_LABELS = [
    ("medical_history", "Medical history"),
    ("physical_examination", "Physical examination"),
    ("auxiliary_examination", "Auxiliary examination"),
    ("clinical_features", "Case characteristics"),
    ("initial_diagnosis", "Initial diagnosis"),
    ("diagnostic_basis", "Diagnostic basis"),
    ("differential_process", "Differential diagnosis process"),
    ("final_diagnosis", "Final diagnosis"),
]

_ROUTE_TARGETS = {"ica": "ica_summarize", "pda": "pda_diagnose", "dda": "dda_differentiate"}


class ICACheck(BaseModel):
    model_config = {"populate_by_name": True}
    flag: bool
    ica_error: str | None = Field(default=None, alias="ICA_error")


class PDACheck(BaseModel):
    model_config = {"populate_by_name": True}
    flag: bool
    pda_error: str | None = Field(default=None, alias="PDA_error")


class DDACheck(BaseModel):
    model_config = {"populate_by_name": True}
    flag: bool
    dda_error: str | None = Field(default=None, alias="DDA_error")


@dataclass
class CAResult:
    flag: bool
    target: str | None
    feedback: str | None


def format_note(fields: dict) -> str:
    """Render clinical-note fields using the paper's template headers
    (verbatim in GENERATE_RAW_CLINICAL_NOTE_PROMPT), for the raw note
    assembled from graph state, the standardized note from the knowledge
    base, and the final assembled note."""
    return "\n\n".join(
        f"{label}:\n{fields.get(key) or 'None'}" for key, label in _NOTE_FIELD_LABELS
    )


def ca_match(state: dict) -> dict:
    return {"standardized_note": standardized_note(state["final_diagnosis"])}


def _run_ca_checks(raw_note: str, std_note_text: str) -> CAResult:
    ica_reply = get_llm().invoke(
        prompts.REFLECT_AND_CORRECT_ICA_PROMPT.format(
            raw_clinical_note=raw_note, standardized_clinical_note=std_note_text
        )
    ).content
    ica_check = parse_json(ICACheck, ica_reply)
    if not ica_check.flag:
        return CAResult(flag=False, target="ica", feedback=ica_check.ica_error or "unspecified")

    pda_reply = get_llm().invoke(
        prompts.REFLECT_AND_CORRECT_PDA_PROMPT.format(
            raw_clinical_note=raw_note, standardized_clinical_note=std_note_text
        )
    ).content
    pda_check = parse_json(PDACheck, pda_reply)
    if not pda_check.flag:
        return CAResult(flag=False, target="pda", feedback=pda_check.pda_error or "unspecified")

    dda_reply = get_llm().invoke(
        prompts.REFLECT_AND_CORRECT_DDA_PROMPT.format(
            raw_clinical_note=raw_note, standardized_clinical_note=std_note_text
        )
    ).content
    dda_check = parse_json(DDACheck, dda_reply)
    if not dda_check.flag:
        return CAResult(flag=False, target="dda", feedback=dda_check.dda_error or "unspecified")

    return CAResult(flag=True, target=None, feedback=None)


def ca_reflect(state: dict) -> dict:
    raw_note = format_note(state)
    std_note_text = format_note(state["standardized_note"])
    result = _run_ca_checks(raw_note, std_note_text)

    updates = {"ca_iters": state.get("ca_iters", 0) + 1, "ca_target": result.target}
    if result.flag:
        return updates

    updates["ca_feedback"] = result.feedback
    if result.target == "pda":
        updates["pda_error"] = result.feedback
        updates["pda_iters"] = 0
    elif result.target == "dda":
        updates["dda_error"] = result.feedback
        updates["dda_iters"] = 0
    return updates


def route_after_ca(state: dict) -> str:
    if state.get("ca_target") is None or state.get("ca_iters", 0) >= config.Config.ca_max_iters:
        return "assemble"
    return _ROUTE_TARGETS[state["ca_target"]]


def assemble(state: dict) -> dict:
    return {"clinical_note": format_note(state)}
