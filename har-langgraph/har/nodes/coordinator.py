from pydantic import field_validator
from har.llm import LLMResponse, text_from_llm


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
