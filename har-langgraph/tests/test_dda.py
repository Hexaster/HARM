"""Tests for the Differential Diagnosis Agent and its self-reflection loop
(paper Sec. 3.2; plan Task 5).

dda_retrieve is a pure knowledge-base lookup (no LLM). dda_reflect and
route_after_dda_reflect have a precisely specified I/O contract, tested
deterministically with a stubbed LLM. dda_differentiate's output-parsing
format isn't specified beyond "a differential process", and the paper's
own REFLECT_DIFFERENTIAL_DIAGNOSIS_PROCESS_PROMPT template carries no
explicit reference back to the candidate diagnosis being tested, so the
live round-trip test only checks the pipeline runs and returns one of the
two documented shapes - not specific diagnostic content, which is the
job of the full end-to-end gate.
"""
from har import config
from har.knowledge import differentials
from har.nodes.dda import dda_differentiate, dda_reflect, dda_retrieve, route_after_dda_reflect

from .conftest import FakeLLM, requires_llm


def test_dda_retrieve_delegates_to_the_knowledge_graph():
    state = {"initial_diagnosis": "stress incontinence"}
    assert dda_retrieve(state) == {"disease_list": differentials("stress incontinence")}


def test_dda_reflect_maps_flag_true_to_final_diagnosis(monkeypatch):
    monkeypatch.setattr(
        "har.nodes.dda.get_llm",
        lambda: FakeLLM(['{"flag": true, "Final_Diagnosis": "stress incontinence"}']),
    )
    state = {
        "disease_list": ["urge incontinence", "overflow incontinence"],
        "differential_process": "1. Urge incontinence ruled out. 2. Overflow incontinence ruled out.",
    }
    result = dda_reflect(state)
    assert result["dda_error"] is None
    assert result["final_diagnosis"]


def test_dda_reflect_maps_flag_false_to_dda_error(monkeypatch):
    monkeypatch.setattr(
        "har.nodes.dda.get_llm",
        lambda: FakeLLM(['{"flag": false, "diff_error": "overflow incontinence not adequately excluded"}']),
    )
    state = {
        "disease_list": ["urge incontinence", "overflow incontinence"],
        "differential_process": "1. Urge incontinence ruled out.",
    }
    result = dda_reflect(state)
    assert result["dda_error"] == "overflow incontinence not adequately excluded"
    assert result.get("final_diagnosis") is None


def test_route_after_dda_reflect_loops_while_error_and_under_cap():
    state = {"dda_error": "needs another pass", "dda_iters": config.Config.dda_max_iters - 1}
    assert route_after_dda_reflect(state) == "dda_differentiate"


def test_route_after_dda_reflect_stops_at_iteration_cap_even_with_error():
    state = {"dda_error": "still wrong", "dda_iters": config.Config.dda_max_iters}
    assert route_after_dda_reflect(state) == "ca_match"


def test_route_after_dda_reflect_advances_once_reflection_passes():
    state = {"dda_error": None, "dda_iters": 1}
    assert route_after_dda_reflect(state) == "ca_match"


@requires_llm
def test_dda_differentiate_then_reflect_round_trip_on_paper_case():
    disease_list = differentials("stress incontinence")
    state = {"initial_diagnosis": "stress incontinence", "disease_list": disease_list}

    diff_result = dda_differentiate(state)
    assert diff_result.keys() == {"differential_process", "dda_iters"}
    assert diff_result["differential_process"].strip()
    assert diff_result["dda_iters"] == 1

    reflect_state = {**state, **diff_result}
    reflect_result = dda_reflect(reflect_state)
    # Either the reflection passed (final_diagnosis set) or flagged an
    # issue (dda_error set) - exactly one of the two documented shapes.
    assert bool(reflect_result.get("final_diagnosis")) != bool(reflect_result.get("dda_error"))
