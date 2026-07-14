from har import config
from har.nodes import coordinator as ca
from tests.conftest import FakeLLM

FULL_STATE = {
    "medical_history": "Leakage on coughing; Mirabegron partial response.",
    "physical_examination": "None",
    "auxiliary_examination": "WBC 27.7/HPF -> 2.1/HPF.",
    "clinical_features": "Stress-pattern leakage with an overactive component.",
    "initial_diagnosis": "stress incontinence, overactive bladder",
    "diagnostic_basis": "Leakage synchronous with exertion; partial Mirabegron response.",
    "differential_process": "Urge/overflow incontinence and LUTS excluded.",
    "final_diagnosis": "stress incontinence, overactive bladder",
}

FLAG_TRUE = '{"flag": true}'


def test_ca_match_fetches_the_standardized_note_for_the_final_diagnosis():
    result = ca.ca_match({"final_diagnosis": "stress incontinence, overactive bladder"})

    note = result["standardized_note"]
    assert note["reviewed"] is True
    assert "Mirabegron" in note["medical_history"]


def test_ca_reflect_all_checks_pass_clears_target_after_three_calls(monkeypatch):
    fake = FakeLLM([FLAG_TRUE, FLAG_TRUE, FLAG_TRUE])
    monkeypatch.setattr(ca, "get_llm", lambda: fake)

    state = dict(FULL_STATE, standardized_note=FULL_STATE, ca_iters=0)
    result = ca.ca_reflect(state)

    assert result["ca_target"] is None
    assert result["ca_iters"] == 1
    assert "ca_feedback" not in result
    assert len(fake.calls) == 3


def test_ca_reflect_ica_failure_short_circuits_before_pda_and_dda_checks(monkeypatch):
    fake = FakeLLM(['{"flag": false, "ICA_error": "physical examination section is inconsistent"}'])
    monkeypatch.setattr(ca, "get_llm", lambda: fake)

    state = dict(FULL_STATE, standardized_note=FULL_STATE)
    result = ca.ca_reflect(state)

    assert result["ca_target"] == "ica"
    assert result["ca_feedback"] == "physical examination section is inconsistent"
    assert "pda_error" not in result
    assert "dda_error" not in result
    assert len(fake.calls) == 1


def test_ca_reflect_pda_failure_resets_pda_iters_and_stops_before_dda_check(monkeypatch):
    fake = FakeLLM([FLAG_TRUE, '{"flag": false, "PDA_error": "diagnostic basis is too thin"}'])
    monkeypatch.setattr(ca, "get_llm", lambda: fake)

    state = dict(FULL_STATE, standardized_note=FULL_STATE, pda_iters=4)
    result = ca.ca_reflect(state)

    assert result["ca_target"] == "pda"
    assert result["pda_error"] == "diagnostic basis is too thin"
    assert result["pda_iters"] == 0
    assert "dda_error" not in result
    assert len(fake.calls) == 2


def test_ca_reflect_dda_failure_resets_dda_iters(monkeypatch):
    fake = FakeLLM([FLAG_TRUE, FLAG_TRUE, '{"flag": false, "DDA_error": "overflow incontinence not excluded"}'])
    monkeypatch.setattr(ca, "get_llm", lambda: fake)

    state = dict(FULL_STATE, standardized_note=FULL_STATE, dda_iters=3)
    result = ca.ca_reflect(state)

    assert result["ca_target"] == "dda"
    assert result["dda_error"] == "overflow incontinence not excluded"
    assert result["dda_iters"] == 0
    assert len(fake.calls) == 3


def test_ca_reflect_increments_ca_iters_from_existing_value(monkeypatch):
    fake = FakeLLM([FLAG_TRUE, FLAG_TRUE, FLAG_TRUE])
    monkeypatch.setattr(ca, "get_llm", lambda: fake)

    state = dict(FULL_STATE, standardized_note=FULL_STATE, ca_iters=1)
    result = ca.ca_reflect(state)

    assert result["ca_iters"] == 2


def test_ca_reflect_sends_both_raw_and_standardized_note_text(monkeypatch):
    fake = FakeLLM([FLAG_TRUE, FLAG_TRUE, FLAG_TRUE])
    monkeypatch.setattr(ca, "get_llm", lambda: fake)

    state = dict(FULL_STATE, standardized_note=FULL_STATE)
    ca.ca_reflect(state)

    ica_prompt = fake.calls[0]
    assert "Mirabegron" in ica_prompt
    assert ica_prompt.count("Mirabegron") >= 1


def test_route_after_ca_maps_each_target_to_its_agent_entry_node():
    assert ca.route_after_ca({"ca_target": "ica", "ca_iters": 1}) == "ica_summarize"
    assert ca.route_after_ca({"ca_target": "pda", "ca_iters": 1}) == "pda_diagnose"
    assert ca.route_after_ca({"ca_target": "dda", "ca_iters": 1}) == "dda_differentiate"


def test_route_after_ca_assembles_when_target_is_none():
    assert ca.route_after_ca({"ca_target": None, "ca_iters": 1}) == "assemble"


def test_route_after_ca_forces_assemble_at_cap_even_with_a_pending_target():
    state = {"ca_target": "ica", "ca_iters": config.Config.ca_max_iters}
    assert ca.route_after_ca(state) == "assemble"


def test_route_after_ca_allows_route_back_just_under_the_cap():
    state = {"ca_target": "ica", "ca_iters": config.Config.ca_max_iters - 1}
    assert ca.route_after_ca(state) == "ica_summarize"


def test_assemble_produces_a_note_with_all_nine_fields():
    note = ca.assemble(FULL_STATE)["clinical_note"]

    assert "Mirabegron" in note
    assert "stress incontinence" in note
    for label in (
        "Medical history", "Physical examination", "Auxiliary examination",
        "Case characteristics", "Initial diagnosis", "Diagnostic basis",
        "Differential diagnosis process", "Final diagnosis",
    ):
        assert f"{label}:" in note


def test_coordinator_loop_terminates_at_ca_max_iters_with_persistent_ica_failure(monkeypatch):
    """Plan Task 6: 'assert overall stop at CA_MAX_ITERS' even if the
    Coordinator keeps finding the same fault forever."""
    n = config.Config.ca_max_iters
    fake = FakeLLM(['{"flag": false, "ICA_error": "still inconsistent"}'] * n)
    monkeypatch.setattr(ca, "get_llm", lambda: fake)

    state = dict(FULL_STATE, standardized_note=FULL_STATE)
    iterations = 0
    route = None
    while True:
        state.update(ca.ca_reflect(state))
        iterations += 1
        route = ca.route_after_ca(state)
        assert iterations <= n, "coordinator loop did not respect ca_max_iters"
        if route == "assemble":
            break

    assert iterations == n
    assert route == "assemble"
    assert len(fake.calls) == n
