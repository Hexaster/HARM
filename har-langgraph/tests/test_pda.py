from har import config
from har.nodes import pda
from tests.conftest import FakeLLM

CLINICAL_FEATURES = (
    "1. Urinary leakage synchronous with coughing/sneezing/urgency. "
    "2. Partial response to Mirabegron, relapse two days after stopping. "
    "3. Recent coronary intervention. 4. Urinalysis WBC improved 27.7->2.1/HPF."
)

DIAGNOSE_REPLY = """Initial diagnosis:
Stress incontinence, overactive bladder.

Diagnostic basis:
1. Leakage synchronous with coughing/sneezing without a preceding urge suggests a stress component.
2. Partial, non-sustained response to Mirabegron with relapse after stopping suggests an overactive bladder component."""


def _interleave(a, b):
    out = []
    for x, y in zip(a, b):
        out.extend([x, y])
    return out


def test_pda_diagnose_extracts_diagnosis_and_basis(monkeypatch):
    fake = FakeLLM([DIAGNOSE_REPLY])
    monkeypatch.setattr(pda, "get_llm", lambda: fake)

    result = pda.pda_diagnose({"clinical_features": CLINICAL_FEATURES, "question": "q"})

    assert "stress incontinence" in result["initial_diagnosis"].lower()
    assert "Mirabegron" in result["diagnostic_basis"]
    assert result["pda_iters"] == 1


def test_pda_diagnose_falls_back_to_whole_reply_when_headers_are_absent(monkeypatch):
    fake = FakeLLM(["stress incontinence, no clear header structure here"])
    monkeypatch.setattr(pda, "get_llm", lambda: fake)

    result = pda.pda_diagnose({"clinical_features": CLINICAL_FEATURES, "question": "q"})

    assert "stress incontinence" in result["initial_diagnosis"].lower()
    assert result["diagnostic_basis"] == ""


def test_pda_diagnose_increments_existing_iter_count(monkeypatch):
    fake = FakeLLM([DIAGNOSE_REPLY])
    monkeypatch.setattr(pda, "get_llm", lambda: fake)

    result = pda.pda_diagnose({"clinical_features": CLINICAL_FEATURES, "question": "q", "pda_iters": 3})

    assert result["pda_iters"] == 4


def test_pda_diagnose_includes_prior_error_feedback_in_the_prompt(monkeypatch):
    fake = FakeLLM([DIAGNOSE_REPLY])
    monkeypatch.setattr(pda, "get_llm", lambda: fake)

    pda.pda_diagnose({
        "clinical_features": CLINICAL_FEATURES, "question": "q",
        "pda_error": "unique-feedback-marker-42",
    })

    assert "unique-feedback-marker-42" in fake.calls[0]


def test_pda_diagnose_omits_feedback_block_on_first_attempt(monkeypatch):
    fake = FakeLLM([DIAGNOSE_REPLY])
    monkeypatch.setattr(pda, "get_llm", lambda: fake)

    pda.pda_diagnose({"clinical_features": CLINICAL_FEATURES, "question": "q"})

    assert "previous reflection" not in fake.calls[0].lower()


def test_pda_reflect_clears_error_when_flag_true(monkeypatch):
    fake = FakeLLM(['Some reasoning first.\n{"flag": true}'])
    monkeypatch.setattr(pda, "get_llm", lambda: fake)

    result = pda.pda_reflect({"initial_diagnosis": "stress incontinence", "diagnostic_basis": "basis"})

    assert result == {"pda_error": None}


def test_pda_reflect_extracts_error_when_flag_false(monkeypatch):
    reply = (
        "Let me think step by step about whether this fits the key points.\n"
        '{"flag": false, "diagnosis_error": "does not address the urge component"}\n'
        "That is my assessment."
    )
    fake = FakeLLM([reply])
    monkeypatch.setattr(pda, "get_llm", lambda: fake)

    result = pda.pda_reflect({"initial_diagnosis": "stress incontinence", "diagnostic_basis": "basis"})

    assert result == {"pda_error": "does not address the urge component"}


def test_pda_reflect_looks_up_key_points_for_the_diagnosed_disease(monkeypatch):
    fake = FakeLLM(['{"flag": true}'])
    monkeypatch.setattr(pda, "get_llm", lambda: fake)

    pda.pda_reflect({"initial_diagnosis": "Stress Incontinence", "diagnostic_basis": "basis"})

    sent_prompt = fake.calls[0].lower()
    assert "cough stress test" in sent_prompt or "pelvic floor" in sent_prompt


def test_pda_reflect_resolves_a_multi_disease_diagnosis_string(monkeypatch):
    # initial_diagnosis often lists more than one disease ("X, Y"); key_points
    # must still resolve via fuzzy/substring matching rather than raising.
    fake = FakeLLM(['{"flag": true}'])
    monkeypatch.setattr(pda, "get_llm", lambda: fake)

    result = pda.pda_reflect({
        "initial_diagnosis": "stress incontinence, overactive bladder", "diagnostic_basis": "basis",
    })

    assert result == {"pda_error": None}


def test_route_after_pda_reflect_loops_while_error_and_under_cap():
    state = {"pda_error": "bad", "pda_iters": config.Config.pda_max_iters - 1}
    assert pda.route_after_pda_reflect(state) == "pda_diagnose"


def test_route_after_pda_reflect_stops_at_cap_despite_persisting_error():
    state = {"pda_error": "still bad", "pda_iters": config.Config.pda_max_iters}
    assert pda.route_after_pda_reflect(state) == "dda_retrieve"


def test_route_after_pda_reflect_stops_when_error_cleared():
    state = {"pda_error": None, "pda_iters": 1}
    assert pda.route_after_pda_reflect(state) == "dda_retrieve"


def test_route_after_pda_reflect_stops_when_never_run():
    assert pda.route_after_pda_reflect({}) == "dda_retrieve"


def test_pda_reflection_loop_terminates_at_max_iters_with_persistent_failure(monkeypatch):
    """A reflector that always finds fault must not loop forever -- the
    configured hard cap must stop it (plan Task 4, 'use a stub reflector')."""
    n = config.Config.pda_max_iters
    diagnose_replies = [DIAGNOSE_REPLY] * n
    reflect_replies = ['{"flag": false, "diagnosis_error": "still not convincing"}'] * n
    fake = FakeLLM(_interleave(diagnose_replies, reflect_replies))
    monkeypatch.setattr(pda, "get_llm", lambda: fake)

    state = {"clinical_features": CLINICAL_FEATURES, "question": "q"}
    iterations = 0
    route = None
    while True:
        state.update(pda.pda_diagnose(state))
        state.update(pda.pda_reflect(state))
        iterations += 1
        route = pda.route_after_pda_reflect(state)
        assert iterations <= n, "loop did not respect the iteration cap"
        if route != "pda_diagnose":
            break

    assert iterations == n
    assert route == "dda_retrieve"
    assert len(fake.calls) == 2 * n


def test_pda_reflection_loop_stops_early_once_reflection_passes(monkeypatch):
    fake = FakeLLM([
        DIAGNOSE_REPLY, '{"flag": false, "diagnosis_error": "missing urgency detail"}',
        DIAGNOSE_REPLY, '{"flag": true}',
    ])
    monkeypatch.setattr(pda, "get_llm", lambda: fake)

    state = {"clinical_features": CLINICAL_FEATURES, "question": "q"}
    iterations = 0
    route = None
    while True:
        state.update(pda.pda_diagnose(state))
        state.update(pda.pda_reflect(state))
        iterations += 1
        route = pda.route_after_pda_reflect(state)
        if route != "pda_diagnose":
            break

    assert iterations == 2
    assert route == "dda_retrieve"
    assert state["pda_iters"] == 2
