from har import config
from har.nodes import dda
from tests.conftest import FakeLLM


def _interleave(a, b):
    out = []
    for x, y in zip(a, b):
        out.extend([x, y])
    return out


def test_dda_retrieve_pulls_the_fig3_differential_list_for_stress_incontinence():
    result = dda.dda_retrieve({"initial_diagnosis": "stress incontinence"})

    assert set(result["disease_list"]) == {
        "urge incontinence", "overflow incontinence",
        "overactive bladder", "lower urinary tract syndrome",
    }


def test_dda_retrieve_resolves_a_fuzzy_or_casing_variant_diagnosis():
    result = dda.dda_retrieve({"initial_diagnosis": "Stress Incontinence"})

    assert "urge incontinence" in result["disease_list"]


def test_dda_differentiate_extracts_diff_process_and_increments_iters(monkeypatch):
    fake = FakeLLM(['{"diff_process": "1. Urge incontinence excluded: no preceding urge."}'])
    monkeypatch.setattr(dda, "get_llm", lambda: fake)

    result = dda.dda_differentiate({
        "disease_list": ["urge incontinence", "overactive bladder"],
    })

    assert "Urge incontinence excluded" in result["differential_process"]
    assert result["dda_iters"] == 1


def test_dda_differentiate_sends_key_points_for_every_listed_disease(monkeypatch):
    fake = FakeLLM(['{"diff_process": "..."}'])
    monkeypatch.setattr(dda, "get_llm", lambda: fake)

    dda.dda_differentiate({"disease_list": ["urge incontinence", "overactive bladder"]})

    sent_prompt = fake.calls[0]
    assert "urge incontinence" in sent_prompt
    assert "overactive bladder" in sent_prompt


def test_dda_differentiate_includes_prior_error_feedback(monkeypatch):
    fake = FakeLLM(['{"diff_process": "revised"}'])
    monkeypatch.setattr(dda, "get_llm", lambda: fake)

    dda.dda_differentiate({
        "disease_list": ["urge incontinence"], "dda_error": "unique-dda-marker-7",
    })

    assert "unique-dda-marker-7" in fake.calls[0]


def test_dda_differentiate_increments_existing_iter_count(monkeypatch):
    fake = FakeLLM(['{"diff_process": "x"}'])
    monkeypatch.setattr(dda, "get_llm", lambda: fake)

    result = dda.dda_differentiate({"disease_list": [], "dda_iters": 2})

    assert result["dda_iters"] == 3


def test_dda_reflect_sets_final_diagnosis_on_the_papers_mixed_case_json_key(monkeypatch):
    # the paper's own prompt requests `"Final_Diagnosis"` (mixed case), not
    # `final_diagnosis` -- this must be aliased correctly or the diagnosis
    # silently comes back as None.
    reply = '{"flag": true, "Final_Diagnosis": "stress incontinence, overactive bladder"}'
    fake = FakeLLM([reply])
    monkeypatch.setattr(dda, "get_llm", lambda: fake)

    result = dda.dda_reflect({
        "disease_list": ["urge incontinence"], "differential_process": "process",
    })

    assert result == {"final_diagnosis": "stress incontinence, overactive bladder", "dda_error": None}


def test_dda_reflect_extracts_diff_error_when_flag_false(monkeypatch):
    reply = (
        "Reasoning about each excluded disease...\n"
        '{"flag": false, "diff_error": "urge incontinence not adequately ruled out"}'
    )
    fake = FakeLLM([reply])
    monkeypatch.setattr(dda, "get_llm", lambda: fake)

    result = dda.dda_reflect({
        "disease_list": ["urge incontinence"], "differential_process": "process",
    })

    assert result == {"dda_error": "urge incontinence not adequately ruled out"}


def test_route_after_dda_reflect_loops_while_error_and_under_cap():
    state = {"dda_error": "bad", "dda_iters": config.Config.dda_max_iters - 1}
    assert dda.route_after_dda_reflect(state) == "dda_differentiate"


def test_route_after_dda_reflect_stops_at_cap_despite_persisting_error():
    state = {"dda_error": "still bad", "dda_iters": config.Config.dda_max_iters}
    assert dda.route_after_dda_reflect(state) == "ca_match"


def test_route_after_dda_reflect_stops_when_error_cleared():
    state = {"dda_error": None, "dda_iters": 1}
    assert dda.route_after_dda_reflect(state) == "ca_match"


def test_dda_reflection_loop_terminates_at_max_iters_with_persistent_failure(monkeypatch):
    """Mirrors the PDA loop-cap test: a reflector that always finds fault
    must not loop forever."""
    n = config.Config.dda_max_iters
    diff_replies = ['{"diff_process": "attempt"}'] * n
    reflect_replies = ['{"flag": false, "diff_error": "still not convincing"}'] * n
    fake = FakeLLM(_interleave(diff_replies, reflect_replies))
    monkeypatch.setattr(dda, "get_llm", lambda: fake)

    state = {"disease_list": ["urge incontinence", "overactive bladder"]}
    iterations = 0
    route = None
    while True:
        state.update(dda.dda_differentiate(state))
        state.update(dda.dda_reflect(state))
        iterations += 1
        route = dda.route_after_dda_reflect(state)
        assert iterations <= n, "loop did not respect the iteration cap"
        if route != "dda_differentiate":
            break

    assert iterations == n
    assert route == "ca_match"
    assert "final_diagnosis" not in state
    assert len(fake.calls) == 2 * n


def test_dda_full_loop_stops_early_once_reflection_passes(monkeypatch):
    fake = FakeLLM([
        '{"diff_process": "first attempt"}',
        '{"flag": false, "diff_error": "overactive bladder not addressed"}',
        '{"diff_process": "revised attempt"}',
        '{"flag": true, "Final_Diagnosis": "stress incontinence, overactive bladder"}',
    ])
    monkeypatch.setattr(dda, "get_llm", lambda: fake)

    state = {"disease_list": ["urge incontinence", "overactive bladder"]}
    iterations = 0
    route = None
    while True:
        state.update(dda.dda_differentiate(state))
        state.update(dda.dda_reflect(state))
        iterations += 1
        route = dda.route_after_dda_reflect(state)
        if route != "dda_differentiate":
            break

    assert iterations == 2
    assert route == "ca_match"
    assert state["final_diagnosis"] == "stress incontinence, overactive bladder"
