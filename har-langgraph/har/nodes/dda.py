"""Differential Diagnosis Agent: retrieve the differential disease list from
the knowledge graph, run a differential diagnosis process against it, then
self-reflect and iterate until satisfied or the cap is hit, producing the
final diagnosis (paper Sec. 3.2, fields y7-y9).
"""
from pydantic import BaseModel, Field

from har import config, prompts
from har.knowledge import differentials, key_points
from har.llm import get_llm, parse_json


class DiffProcess(BaseModel):
    diff_process: str


class DDAReflection(BaseModel):
    model_config = {"populate_by_name": True}

    flag: bool
    final_diagnosis: str | None = Field(default=None, alias="Final_Diagnosis")
    diff_error: str | None = None


def dda_retrieve(state: dict) -> dict:
    return {"disease_list": differentials(state["initial_diagnosis"])}


def dda_differentiate(state: dict) -> dict:
    kp = {d: key_points(d) for d in state["disease_list"]}
    msg = prompts.DIFFERENTIAL_DIAGNOSIS_PROMPT.format(
        diseases_list=state["disease_list"], key_inquiry_points=kp
    )
    if state.get("dda_error"):
        msg += f"\n\nA previous reflection flagged these diseases for re-diagnosis: {state['dda_error']}"

    reply = get_llm().invoke(msg).content
    diff_process = parse_json(DiffProcess, reply).diff_process
    return {
        "differential_process": diff_process,
        "dda_iters": state.get("dda_iters", 0) + 1,
    }


def dda_reflect(state: dict) -> dict:
    msg = prompts.REFLECT_DIFFERENTIAL_DIAGNOSIS_PROCESS_PROMPT.format(
        diseases_list=state["disease_list"],
        differential_diagnosis_process=state["differential_process"],
    )
    reply = get_llm().invoke(msg).content
    reflection = parse_json(DDAReflection, reply)
    if reflection.flag:
        return {"final_diagnosis": reflection.final_diagnosis, "dda_error": None}
    return {"dda_error": reflection.diff_error or "unspecified"}


def route_after_dda_reflect(state: dict) -> str:
    if state.get("dda_error") and state.get("dda_iters", 0) < config.Config.dda_max_iters:
        return "dda_differentiate"
    return "ca_match"
