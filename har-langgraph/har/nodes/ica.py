import re

from har.llm import get_llm
from har import prompts

_SECTION_PATTERN = re.compile(
    r"medical history:(.*?)physical examination:(.*?)auxiliary examination:(.*)",
    re.IGNORECASE | re.DOTALL,
)

def ica_extract(state):
    msg = prompts.PATIENT_INFORMATION_EXTRACTION_PROMPT.format(
        question=state["question"],
    )
    llm = get_llm()
    out = llm.invoke(msg).content

    medical_history, physical_examination, auxiliary_examination = _split_three_sections(out)

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
    
    return {"clinical_features": get_llm().invoke(msg).content}

def _split_three_sections(text):
    match = _SECTION_PATTERN.search(text)
    if not match:
        raise ValueError(f"Could not find the three labeled sections in LLM output: {text!r}")

    medical_history, physical_examination, auxiliary_examination = (
        group.strip() for group in match.groups()
    )

    return medical_history, physical_examination, auxiliary_examination