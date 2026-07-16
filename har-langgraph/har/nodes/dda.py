from pydantic import field_validator
from har.knowledge import differentials, key_points
from har.config import Config
from har.llm import LLMResponse, get_llm, parse_json, text_from_llm
from har import prompts

class DifferentialDiagnosisProcess(LLMResponse):
    differential_diagnosis_process: str

    _normalize_text = field_validator(
        "differential_diagnosis_process", mode="before"
    )(text_from_llm)

class DDAReflection(LLMResponse):
    flag: bool
    final_diagnosis: str | None = None
    diff_error: str | None = None

    _normalize_text = field_validator(
        "final_diagnosis", "diff_error", mode="before"
    )(text_from_llm)

def dda_retrieve(state) -> dict:
    """
    Retrieve the list of diseases to be ruled out for differential diagnosis based on the initial diagnosis.
    """
    dl = differentials(state["initial_diagnosis"])
    return {
        "diseases_list": dl
    }

def dda_differentiate(state) -> dict:
    '''
    Differentiate between the diseases in the list based on key inquiry points.
    '''
    kp = {d: key_points(d) for d in state["diseases_list"]}
    msg = prompts.DIFFERENTIAL_DIAGNOSIS_PROMPT.format(
        diseases_list=state["diseases_list"],
        key_inquiry_points=kp,
    )
    if state.get("dda_error"):
        msg += f"\n\nRe-diagnose these: {state['dda_error']}"

    out = get_llm().invoke(msg).content
    proc = parse_json(
        DifferentialDiagnosisProcess, out
    ).differential_diagnosis_process
    return {
        "differential_diagnosis_process": proc,
        "dda_iters": state.get("dda_iters", 0) + 1,
    }

def dda_reflect(state) -> dict:
    '''
    Reflect on the differential diagnosis process and determine if there are any errors.
    '''
    msg = prompts.REFLECT_DIFFERENTIAL_DIAGNOSIS_PROCESS_PROMPT.format(
        diseases_list=state["diseases_list"],
        differential_diagnosis_process=state["differential_diagnosis_process"],
    )

    r = parse_json(DDAReflection, get_llm().invoke(msg).content)

    if not r.flag:
        return {
            "dda_error": r.diff_error or "unspecified"
        }

    return {
        "final_diagnosis": r.final_diagnosis,
        "dda_error": None
    }

def route_after_dda_reflect(state) -> str:
    """
    Determine the next step after reflecting on the differential diagnosis process.
    If there is an error and we haven't reached the iteration cap, loop back to differentiation.
    Otherwise, proceed to the final diagnosis step.
    """
    if state.get("dda_error") and state.get("dda_iters", 0) < Config.DDA_MAX_ITERS:
        return "dda_differentiate"
    return "ca_match"
