import json

from pydantic import field_validator

from har import prompts
from har.config import Config
from har.knowledge import standardized_note
from har.llm import LLMResponse, get_llm, parse_json, text_from_llm


NOTE_FIELDS = (
    "medical_history",
    "physical_examination",
    "auxiliary_examination",
    "clinical_features",
    "initial_diagnosis",
    "diagnostic_basis",
    "diseases_list",
    "differential_diagnosis_process",
    "final_diagnosis",
)


class CAReflection(LLMResponse):
    flag: bool
    ica_error: str | None = None
    pda_error: str | None = None
    dda_error: str | None = None

    _normalize_text = field_validator(
        "ica_error", "pda_error", "dda_error", mode="before"
    )(text_from_llm)


def ca_match(state):
    return {"_std_note": standardized_note(state["final_diagnosis"])}


def ca_reflect(state):
    raw = _assemble_raw_note(state)
    std = state["_std_note"]
    reflection = _run_ca_checks(raw, std)

    updates = {
        "ca_iters": state.get("ca_iters", 0) + 1,
        "ca_target": None,
        "ca_feedback": None,
    }
    if reflection.flag:
        return updates
    if reflection.ica_error:
        updates.update(
            ca_target="ica",
            ca_feedback=reflection.ica_error,
        )
    elif reflection.pda_error:
        updates.update(
            ca_target="pda",
            pda_error=reflection.pda_error,
            pda_iters=0,
        )
    elif reflection.dda_error:
        updates.update(
            ca_target="dda",
            dda_error=reflection.dda_error,
            dda_iters=0,
        )
    return updates


def route_after_ca(state):
    if (
        state.get("ca_target") is None
        or state.get("ca_iters", 0) >= Config.CA_MAX_ITERS
    ):
        return "assemble"
    return {
        "ica": "ica_summarize",
        "pda": "pda_diagnose",
        "dda": "dda_differentiate",
    }[state["ca_target"]]


def assemble(state):
    return {"clinical_note": _format_note(state)}


def _assemble_raw_note(state):
    return {field: state.get(field) for field in NOTE_FIELDS}


def _run_ca_checks(raw, std):
    llm = get_llm()
    raw_text = json.dumps(raw, ensure_ascii=False)
    std_text = json.dumps(std, ensure_ascii=False)
    checks = (
        (prompts.REFLECT_AND_CORRECT_ICA_PROMPT, "ica_error"),
        (prompts.REFLECT_AND_CORRECT_PDA_PROMPT, "pda_error"),
        (prompts.REFLECT_AND_CORRECT_DDA_PROMPT, "dda_error"),
    )

    for prompt, error_field in checks:
        message = prompt.format(
            raw_clinical_note=raw_text,
            standardized_clinical_note=std_text,
        )
        try:
            reflection = parse_json(CAReflection, llm.invoke(message).content)
        except ValueError:
            # A malformed reflection must not trap the graph in a retry loop.
            continue

        if not reflection.flag:
            error = getattr(reflection, error_field) or "unspecified"
            return CAReflection(flag=False, **{error_field: error})

    return CAReflection(flag=True)


def _format_note(state):
    labels = {
        "medical_history": "Medical History",
        "physical_examination": "Physical Examination",
        "auxiliary_examination": "Auxiliary Examination",
        "clinical_features": "Clinical Features",
        "initial_diagnosis": "Initial Diagnosis",
        "diagnostic_basis": "Diagnostic Basis",
        "diseases_list": "Diseases for Differential Diagnosis",
        "differential_diagnosis_process": "Differential Diagnosis Process",
        "final_diagnosis": "Final Diagnosis",
    }
    sections = []
    for field in NOTE_FIELDS:
        value = state.get(field, "None")
        if isinstance(value, list):
            value = ", ".join(value)
        sections.append(f"{labels[field]}:\n{value}")
    return "\n\n".join(sections)
