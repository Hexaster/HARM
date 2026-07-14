"""Preliminary Diagnosis Agent: make an initial diagnosis with basis, then
self-reflect against the disease's key inquiry points and retry until the
reflection is satisfied or the iteration cap is hit (paper Sec. 3.2, fields
y5-y6, reflection loop per Algorithm 1 / Fig. 7).
"""
from pydantic import BaseModel

from har import config, prompts
from har.knowledge import key_points
from har.llm import get_llm, parse_json
from har.nodes import split_labeled_sections

_DX_BASIS_PATTERNS = {
    "initial_diagnosis": r"initial\s*diagnosis\s*:?",
    "diagnostic_basis": r"diagnostic\s*basis\s*:?",
}


class PDAReflection(BaseModel):
    flag: bool
    diagnosis_error: str | None = None


def _split_dx_basis(text: str) -> tuple[str, str]:
    sections = split_labeled_sections(text, _DX_BASIS_PATTERNS)
    if not sections["initial_diagnosis"] and not sections["diagnostic_basis"]:
        return text.strip(), ""
    return sections["initial_diagnosis"], sections["diagnostic_basis"]


def pda_diagnose(state: dict) -> dict:
    msg = prompts.MAKE_PRELIMINARY_DIAGNOSIS_PROMPT.format(
        clinical_features=state["clinical_features"], question=state["question"]
    )
    if state.get("pda_error"):
        msg += f"\n\nA previous reflection found issues with this diagnosis; correct them: {state['pda_error']}"

    text = get_llm().invoke(msg).content
    dx, basis = _split_dx_basis(text)
    return {
        "initial_diagnosis": dx,
        "diagnostic_basis": basis,
        "pda_iters": state.get("pda_iters", 0) + 1,
    }


def pda_reflect(state: dict) -> dict:
    kp = key_points(state["initial_diagnosis"])
    msg = prompts.REFLECT_PRELIMINARY_DIAGNOSIS_PROMPT.format(
        preliminary_diagnosis=state["initial_diagnosis"],
        diagnostic_basis=state["diagnostic_basis"],
        key_inquiry_points=kp,
    )
    reply = get_llm().invoke(msg).content
    reflection = parse_json(PDAReflection, reply)
    return {"pda_error": None if reflection.flag else (reflection.diagnosis_error or "unspecified")}


def route_after_pda_reflect(state: dict) -> str:
    if state.get("pda_error") and state.get("pda_iters", 0) < config.Config.pda_max_iters:
        return "pda_diagnose"
    return "dda_retrieve"
