from har.nodes import ica
from tests.conftest import FakeLLM

PATIENT_QUESTION = (
    "A 68-year-old woman reports urine leakage when coughing, sneezing, or "
    "feeling urgency. She used Mirabegron for one month with improvement, "
    "but symptoms recurred within two days of stopping. She underwent "
    "coronary intervention two months ago. Urinalysis white blood cell "
    "count was 27.7/HPF two months ago and 2.1/HPF most recently."
)

RAW_EXTRACTION_REPLY = """Medical history:
1. The patient experiences urinary leakage when coughing, sneezing, or during urgency.
2. She used Mirabegron for one month, with symptom improvement, but symptoms recurred within two days after stopping.
3. She underwent coronary intervention two months ago.

Physical examination:
None

Auxiliary examination:
1. Urinalysis white blood cell count was 27.7/HPF two months ago and 2.1/HPF most recently."""


def test_ica_extract_splits_into_three_sections(monkeypatch):
    fake = FakeLLM([RAW_EXTRACTION_REPLY])
    monkeypatch.setattr(ica, "get_llm", lambda: fake)

    result = ica.ica_extract({"question": PATIENT_QUESTION})

    assert "Mirabegron" in result["medical_history"]
    assert "coronary intervention" in result["medical_history"]
    assert "27.7" in result["auxiliary_examination"]
    assert result["physical_examination"] == "None"
    assert PATIENT_QUESTION in fake.calls[0]


def test_ica_extract_handles_sections_in_nonstandard_order(monkeypatch):
    # the prompt doesn't mandate section order; the splitter must not assume it
    reply = (
        "Auxiliary examination:\nWBC 3.1/HPF, unremarkable.\n\n"
        "Medical history:\nNo prior urinary complaints.\n\n"
        "Physical examination:\nNone"
    )
    fake = FakeLLM([reply])
    monkeypatch.setattr(ica, "get_llm", lambda: fake)

    result = ica.ica_extract({"question": "q"})

    assert "3.1" in result["auxiliary_examination"]
    assert "No prior urinary complaints" in result["medical_history"]
    assert result["physical_examination"] == "None"


def test_ica_extract_marks_missing_sections_as_none(monkeypatch):
    # only two of three headers present in the reply
    reply = "Medical history:\nNone reported.\n\nAuxiliary examination:\nNone"
    fake = FakeLLM([reply])
    monkeypatch.setattr(ica, "get_llm", lambda: fake)

    result = ica.ica_extract({"question": "No details given."})

    assert result["physical_examination"] == "None"
    assert result["medical_history"] != ""
    assert result["auxiliary_examination"] != ""


def test_ica_extract_returns_only_the_three_expected_keys(monkeypatch):
    fake = FakeLLM([RAW_EXTRACTION_REPLY])
    monkeypatch.setattr(ica, "get_llm", lambda: fake)

    result = ica.ica_extract({"question": PATIENT_QUESTION})

    assert set(result) == {"medical_history", "physical_examination", "auxiliary_examination"}


def test_ica_summarize_passes_through_extracted_fields_into_the_prompt(monkeypatch):
    fake = FakeLLM(["1. Stress urinary incontinence pattern. 2. Partial Mirabegron response."])
    monkeypatch.setattr(ica, "get_llm", lambda: fake)

    state = {
        "question": PATIENT_QUESTION,
        "medical_history": "Leakage on coughing; used Mirabegron for one month.",
        "physical_examination": "None",
        "auxiliary_examination": "WBC 27.7/HPF two months ago, 2.1/HPF most recently.",
    }
    result = ica.ica_summarize(state)

    assert "clinical_features" in result and result["clinical_features"]
    sent_prompt = fake.calls[0]
    assert "Mirabegron" in sent_prompt
    assert "27.7" in sent_prompt
    assert PATIENT_QUESTION in sent_prompt


def test_ica_summarize_includes_the_original_question_not_just_the_three_fields(monkeypatch):
    fake = FakeLLM(["features"])
    monkeypatch.setattr(ica, "get_llm", lambda: fake)

    ica.ica_summarize({
        "question": "unique-marker-xyz",
        "medical_history": "h",
        "physical_examination": "p",
        "auxiliary_examination": "a",
    })

    assert "unique-marker-xyz" in fake.calls[0]


def test_ica_summarize_result_has_only_clinical_features_key(monkeypatch):
    fake = FakeLLM(["features"])
    monkeypatch.setattr(ica, "get_llm", lambda: fake)

    result = ica.ica_summarize({
        "question": "q", "medical_history": "h",
        "physical_examination": "p", "auxiliary_examination": "a",
    })

    assert set(result) == {"clinical_features"}
