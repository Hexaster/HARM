"""Tests for the Preliminary Diagnosis Agent and its self-reflection loop
(paper Sec. 3.2; plan Task 4).

pda_diagnose's output-parsing format isn't specified beyond "diagnosis +
basis", so its content is only checked live. pda_reflect and
route_after_pda_reflect have a precisely specified I/O contract (paper's
{"flag": ..., "diagnosis_error": ...} JSON in, {"pda_error": str | None}
out; route on pda_error + pda_iters vs the iteration cap), so those are
tested deterministically with a stubbed LLM - no network or API key needed.
"""
from har import config
from har.nodes.pda import pda_diagnose, pda_reflect, route_after_pda_reflect

from .conftest import FakeLLM, PAPER_CASE, PATIENT_QUESTION, requires_llm


@requires_llm
def test_pda_diagnose_identifies_stress_incontinence():
    state = {"question": PATIENT_QUESTION, "clinical_features": PAPER_CASE["clinical_features"]}
    result = pda_diagnose(state)

    assert result.keys() == {"initial_diagnosis", "diagnostic_basis", "pda_iters"}
    assert "stress incontinence" in result["initial_diagnosis"].lower()
    assert result["diagnostic_basis"].strip()
    assert result["pda_iters"] == 1


def test_pda_reflect_maps_flag_false_to_pda_error(monkeypatch):
    monkeypatch.setattr(
        "har.nodes.pda.get_llm",
        lambda: FakeLLM(['{"flag": false, "diagnosis_error": "basis lacks supporting evidence"}']),
    )
    state = {
        "initial_diagnosis": "stress incontinence",
        "diagnostic_basis": PAPER_CASE["diagnostic_basis"],
    }
    assert pda_reflect(state) == {"pda_error": "basis lacks supporting evidence"}


def test_pda_reflect_maps_flag_true_to_no_error(monkeypatch):
    monkeypatch.setattr("har.nodes.pda.get_llm", lambda: FakeLLM(['{"flag": true}']))
    state = {
        "initial_diagnosis": "stress incontinence",
        "diagnostic_basis": PAPER_CASE["diagnostic_basis"],
    }
    assert pda_reflect(state) == {"pda_error": None}


def test_route_after_pda_reflect_loops_while_error_and_under_cap():
    state = {"pda_error": "needs more evidence", "pda_iters": config.Config.pda_max_iters - 1}
    assert route_after_pda_reflect(state) == "pda_diagnose"


def test_route_after_pda_reflect_stops_at_iteration_cap_even_with_error():
    # This is the loop-termination guarantee: a reflector that keeps
    # returning flag=false must not keep the graph looping forever.
    state = {"pda_error": "still wrong", "pda_iters": config.Config.pda_max_iters}
    assert route_after_pda_reflect(state) == "dda_retrieve"


def test_route_after_pda_reflect_advances_once_reflection_passes():
    state = {"pda_error": None, "pda_iters": 1}
    assert route_after_pda_reflect(state) == "dda_retrieve"
