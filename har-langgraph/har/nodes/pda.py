from pydantic import BaseModel
from har.llm import get_llm, parse_json
from har.knowledge import key_points
from har.config import Config
from har import prompts
import re

_SECTION_PATTERN = re.compile(
    r"initial diagnosis:(.*?)diagnostic basis:(.*)",
    re.IGNORECASE | re.DOTALL,
)


class PDAReflection(BaseModel):
    flag: bool
    diagnosis_error: str | None= None

def pda_diagnose(state):
    msg = prompts.MAKE_PRELIMINARY_DIAGNOSIS_PROMPT.format(
        clinical_features=state["clinical_features"],
        question=state["question"],
    )
    
    if state.get("pda_error"):
        msg += f"\n\nPrevious reflection found issues, correct them: {state['pda_error']}"
    
    text = get_llm().invoke(msg).content

    initial_diagnosis, diagnostic_basis = _split_dx_basis(text)

    return {
        "initial_diagnosis": initial_diagnosis,
        "diagnostic_basis": diagnostic_basis,
        "pda_iters": state.get("pda_iters", 0) + 1,
        }

def pda_reflect(state):
    kp = key_points(state["initial_diagnosis"])
    msg = prompts.REFLECT_PRELIMINARY_DIAGNOSIS_PROMPT.format(
        preliminary_diagnosis=state["initial_diagnosis"],
        diagnostic_basis=state["diagnostic_basis"],
        key_inquiry_points=kp,
    )

    text = get_llm().invoke(msg).content

    r = parse_json(PDAReflection, text)

    return {
        "pda_error": None if r.flag else (r.diagnosis_error or "unspecified"),
    }

def route_after_pda_reflect(state):
    if state.get("pda_error") and state.get("pda_iters", 0) < Config.PDA_MAX_ITERS:
        return "pda_diagnose"
    return "dda_retrieve"
        

def _split_dx_basis(text):
    match = _SECTION_PATTERN.search(text)
    if not match:
        raise ValueError(f"Could not find the labeled sections 'initial diagnosis' and 'diagnostic basis' in LLM output: {text!r}")

    initial_diagnosis, diagnostic_basis = (
        group.strip() for group in match.groups()
    )

    return initial_diagnosis, diagnostic_basis