"""Integration gate (plan Task 7): run the fully compiled HAR graph -- not
just individual node functions -- and confirm the paper's worked case comes
out the other end, that the in-graph reflection loop actually retries nodes
via the conditional edges wired in har/graph.py, and that the hierarchical
Coordinator route-back re-runs the right downstream agents.
"""
from har import config
from har.graph import build_har_graph
from har.nodes import coordinator as ca
from har.nodes import dda, ica, pda
from tests.conftest import FakeLLM

PATIENT_QUESTION = (
    "A 68-year-old woman reports urine leakage when coughing, sneezing, or "
    "feeling urgency. She used Mirabegron for one month with improvement, "
    "but symptoms recurred within two days of stopping. She underwent "
    "coronary intervention two months ago. Urinalysis white blood cell "
    "count was 27.7/HPF two months ago and 2.1/HPF most recently."
)

ICA_EXTRACT_REPLY = """Medical history:
1. The 68-year-old patient experiences urinary leakage when coughing, sneezing, or during urgency.
2. She used Mirabegron for one month, with symptom improvement, but symptoms recurred within two days after stopping.
3. She underwent coronary intervention two months ago.

Physical examination:
None

Auxiliary examination:
1. Urinalysis white blood cell count was 27.7/HPF two months ago and 2.1/HPF most recently."""

ICA_SUMMARIZE_REPLY = (
    "1. Involuntary leakage on coughing/sneezing/urgency, only partially responsive to Mirabegron. "
    "2. No physical exam findings recorded. "
    "3. Improving urinary WBC count (27.7 -> 2.1/HPF). "
    "4. Recent coronary intervention, relevant to medication choice."
)

PDA_DIAGNOSE_REPLY = """Initial diagnosis:
Stress incontinence, overactive bladder.

Diagnostic basis:
1. Leakage synchronous with coughing and sneezing without a preceding urge indicates a stress component.
2. Partial, non-sustained response to Mirabegron with relapse after stopping indicates an overactive bladder component."""

PDA_REFLECT_OK = '{"flag": true}'

DDA_DIFFERENTIATE_REPLY = (
    '{"diff_process": "1. Urge incontinence excluded: leakage is synchronous with exertion, '
    'not preceded by urge. 2. Overflow incontinence excluded: no elevated post-void residual. '
    '3. Lower urinary tract syndrome excluded: no voiding-phase symptoms."}'
)

DDA_REFLECT_OK = '{"flag": true, "Final_Diagnosis": "stress incontinence, overactive bladder"}'

CA_CHECK_OK = '{"flag": true}'


def _patch_all_llms(monkeypatch, fake):
    monkeypatch.setattr(ica, "get_llm", lambda: fake)
    monkeypatch.setattr(pda, "get_llm", lambda: fake)
    monkeypatch.setattr(dda, "get_llm", lambda: fake)
    monkeypatch.setattr(ca, "get_llm", lambda: fake)


def _assert_all_nine_fields_populated(result):
    for field in (
        "medical_history", "physical_examination", "auxiliary_examination",
        "clinical_features", "initial_diagnosis", "diagnostic_basis",
        "disease_list", "differential_process", "final_diagnosis", "clinical_note",
    ):
        assert result.get(field), f"{field} was not populated"


def test_end_to_end_reproduces_paper_worked_case(monkeypatch):
    fake = FakeLLM([
        ICA_EXTRACT_REPLY, ICA_SUMMARIZE_REPLY,
        PDA_DIAGNOSE_REPLY, PDA_REFLECT_OK,
        DDA_DIFFERENTIATE_REPLY, DDA_REFLECT_OK,
        CA_CHECK_OK, CA_CHECK_OK, CA_CHECK_OK,
    ])
    _patch_all_llms(monkeypatch, fake)

    result = build_har_graph().invoke({"question": PATIENT_QUESTION})

    assert "stress incontinence" in result["final_diagnosis"].lower()
    assert "overactive bladder" in result["final_diagnosis"].lower()
    _assert_all_nine_fields_populated(result)
    assert len(fake.calls) == 9


def test_end_to_end_pda_reflection_retry_is_honored_by_the_compiled_graph(monkeypatch):
    """Proves the conditional edge registered on pda_reflect in graph.py
    actually loops back to pda_diagnose -- not just that the router function
    returns the right string in isolation (already covered in test_pda.py)."""
    fake = FakeLLM([
        ICA_EXTRACT_REPLY, ICA_SUMMARIZE_REPLY,
        PDA_DIAGNOSE_REPLY, '{"flag": false, "diagnosis_error": "missing urge component detail"}',
        PDA_DIAGNOSE_REPLY, PDA_REFLECT_OK,
        DDA_DIFFERENTIATE_REPLY, DDA_REFLECT_OK,
        CA_CHECK_OK, CA_CHECK_OK, CA_CHECK_OK,
    ])
    _patch_all_llms(monkeypatch, fake)

    result = build_har_graph().invoke({"question": PATIENT_QUESTION})

    assert result["pda_iters"] == 2
    assert "stress incontinence" in result["final_diagnosis"].lower()
    _assert_all_nine_fields_populated(result)
    assert len(fake.calls) == 11


def test_end_to_end_pda_hard_cap_still_reaches_assemble(monkeypatch):
    """A PDA reflector that never approves must not hang the compiled graph
    -- it should still terminate at assemble once PDA_MAX_ITERS is hit."""
    n = config.Config.pda_max_iters
    diagnose_and_fail = []
    for _ in range(n):
        diagnose_and_fail += [PDA_DIAGNOSE_REPLY, '{"flag": false, "diagnosis_error": "still not convincing"}']

    fake = FakeLLM([
        ICA_EXTRACT_REPLY, ICA_SUMMARIZE_REPLY,
        *diagnose_and_fail,
        DDA_DIFFERENTIATE_REPLY, DDA_REFLECT_OK,
        CA_CHECK_OK, CA_CHECK_OK, CA_CHECK_OK,
    ])
    _patch_all_llms(monkeypatch, fake)

    result = build_har_graph().invoke({"question": PATIENT_QUESTION})

    assert result["pda_iters"] == n
    assert result["pda_error"] is not None  # cap was hit with the concern still open
    assert "stress incontinence" in result["final_diagnosis"].lower()
    _assert_all_nine_fields_populated(result)
    assert len(fake.calls) == 2 + 2 * n + 5


def test_end_to_end_coordinator_routes_back_to_ica_and_recovers(monkeypatch):
    """Proves the hierarchical part: when the Coordinator flags the ICA
    section, control returns to ica_summarize and the whole downstream
    chain (PDA, DDA) re-runs -- without resetting pda_iters/dda_iters,
    which only the Coordinator's own pda/dda targets do."""
    fake = FakeLLM([
        ICA_EXTRACT_REPLY,
        ICA_SUMMARIZE_REPLY,                                    # ica_summarize, pass 1
        PDA_DIAGNOSE_REPLY, PDA_REFLECT_OK,                      # pda, pass 1 (iter 1)
        DDA_DIFFERENTIATE_REPLY, DDA_REFLECT_OK,                 # dda, pass 1 (iter 1)
        '{"flag": false, "ICA_error": "clinical features omit the Mirabegron detail"}',  # ca: ICA check fails
        ICA_SUMMARIZE_REPLY,                                     # ica_summarize, pass 2 (route-back)
        PDA_DIAGNOSE_REPLY, PDA_REFLECT_OK,                      # pda, pass 2 (iter 2)
        DDA_DIFFERENTIATE_REPLY, DDA_REFLECT_OK,                 # dda, pass 2 (iter 2)
        CA_CHECK_OK, CA_CHECK_OK, CA_CHECK_OK,                   # ca, pass 2: all three checks pass
    ])
    _patch_all_llms(monkeypatch, fake)

    result = build_har_graph().invoke({"question": PATIENT_QUESTION})

    assert result["ca_iters"] == 2
    assert result["ca_target"] is None
    # route-back to "ica" must not have reset the unrelated PDA/DDA counters
    assert result["pda_iters"] == 2
    assert result["dda_iters"] == 2
    assert "stress incontinence" in result["final_diagnosis"].lower()
    _assert_all_nine_fields_populated(result)
    assert len(fake.calls) == 15
