import re

from pydantic import field_validator

from har.llm import LLMResponse, get_llm, parse_json, text_from_llm
from har import prompts

_SECTION_PATTERN = re.compile(
    r"medical history:(.*?)physical examination:(.*?)auxiliary examination:(.*)",
    re.IGNORECASE | re.DOTALL,
)


class ExtractedPatientInformation(LLMResponse):
    medical_history: str
    physical_examination: str
    auxiliary_examination: str

    _normalize_text = field_validator(
        "medical_history",
        "physical_examination",
        "auxiliary_examination",
        mode="before",
    )(text_from_llm)


class ClinicalFeatures(LLMResponse):
    clinical_features: str

    _normalize_text = field_validator("clinical_features", mode="before")(
        text_from_llm
    )

def ica_extract(state):
    msg = prompts.PATIENT_INFORMATION_EXTRACTION_PROMPT.format(
        question=state["question"],
    )
    llm = get_llm()
    out = llm.invoke(msg).content

    try:
        return parse_json(ExtractedPatientInformation, out).model_dump()
    except ValueError:
        # Accept the labeled format used by earlier prompt versions.
        medical_history, physical_examination, auxiliary_examination = (
            _split_three_sections(out)
        )
        return {
            "medical_history": medical_history,
            "physical_examination": physical_examination,
            "auxiliary_examination": auxiliary_examination,
        }

def ica_analysis_and_summarize(state):
    msg = prompts.ANALYSIS_AND_SUMMARIZE_PROMPT.format(
        medical_history=state["medical_history"],
        physical_examination=state["physical_examination"],
        auxiliary_examination=state["auxiliary_examination"],
        question=state["question"],
    )
    
    out = get_llm().invoke(msg).content
    try:
        return parse_json(ClinicalFeatures, out).model_dump()
    except ValueError:
        # Accept plain text produced by earlier prompt versions.
        return {"clinical_features": out.strip()}

def _split_three_sections(text):
    match = _SECTION_PATTERN.search(text)
    if not match:
        raise ValueError(f"Could not find the three labeled sections in LLM output: {text!r}")

    medical_history, physical_examination, auxiliary_examination = (
        group.strip() for group in match.groups()
    )

    return medical_history, physical_examination, auxiliary_examination
