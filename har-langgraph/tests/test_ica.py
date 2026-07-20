"""Tests for the Information Collection Agent (paper Sec. 3.2; plan Task 3).

ica_extract splits a patient's raw narrative into medical history, physical
examination, and auxiliary examination; ica_summarize itemizes those into
clinical features. Both call the configured LLM directly with no
LLM-independent contract to stub, so these run live against the real model
and are skipped if no endpoint is configured (see conftest.requires_llm).
"""
from har.nodes.ica import ica_extract, ica_summarize

from .conftest import PATIENT_QUESTION, requires_llm


@requires_llm
def test_ica_extract_splits_patient_narrative_into_sections():
    state = {"question": PATIENT_QUESTION}
    result = ica_extract(state)

    assert result.keys() == {"medical_history", "physical_examination", "auxiliary_examination"}
    assert result["medical_history"].strip()
    assert result["auxiliary_examination"].strip()
    # No physical exam findings were given in the narrative - the prompt
    # instructs the agent to mark that section "None" rather than assume.
    assert "none" in result["physical_examination"].strip().lower()
    assert "mirabegron" in result["medical_history"].lower()
    assert "27.7" in result["auxiliary_examination"]


@requires_llm
def test_ica_summarize_itemizes_clinical_features():
    state = {
        "question": PATIENT_QUESTION,
        "medical_history": (
            "1. Urinary leakage on coughing, sneezing, and urgency. "
            "2. One month of Mirabegron with improvement; recurrence within "
            "two days of stopping. 3. Coronary intervention two months ago."
        ),
        "physical_examination": "None",
        "auxiliary_examination": (
            "1. Urinalysis white blood cell count 27.7/HPF two months ago, "
            "2.1/HPF most recently."
        ),
    }
    result = ica_summarize(state)

    assert result.keys() == {"clinical_features"}
    features = result["clinical_features"]
    assert features.strip()
    assert "mirabegron" in features.lower()
    assert "27.7" in features
