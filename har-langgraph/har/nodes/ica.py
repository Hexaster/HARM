"""Information Collection Agent: extract history/exam/aux from the patient's
question, then summarize the itemized clinical features (paper Sec. 3.2,
fields y1-y4).
"""
from har import prompts
from har.llm import get_llm
from har.nodes import split_labeled_sections

_SECTION_PATTERNS = {
    "medical_history": r"medical\s*history\s*:?",
    "physical_examination": r"physical\s*examination\s*:?",
    "auxiliary_examination": r"auxiliary\s*examination\s*:?",
}


def _split_three_sections(text: str) -> dict[str, str]:
    sections = split_labeled_sections(text, _SECTION_PATTERNS)
    return {field: (value or "None") for field, value in sections.items()}


def ica_extract(state: dict) -> dict:
    msg = prompts.PATIENT_INFORMATION_EXTRACTION_PROMPT.format(question=state["question"])
    text = get_llm().invoke(msg).content
    sections = _split_three_sections(text)
    return {
        "medical_history": sections["medical_history"],
        "physical_examination": sections["physical_examination"],
        "auxiliary_examination": sections["auxiliary_examination"],
    }


def ica_summarize(state: dict) -> dict:
    msg = prompts.ANALYSIS_AND_SUMMARIZE_PROMPT.format(
        medical_history=state["medical_history"],
        physical_examination=state["physical_examination"],
        auxiliary_examination=state["auxiliary_examination"],
        question=state["question"],
    )
    return {"clinical_features": get_llm().invoke(msg).content}
