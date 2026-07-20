from har.knowledge import standardized_note
from har.nodes.dda import dda_differentiate, dda_reflect, dda_retrieve
from har.nodes.ica import ica_summarize, ica_extract
from har.nodes.pda import pda_diagnose

from .conftest import FakeLLM


TEMPLATE_FIELDS = {
    "medical_history",
    "physical_examination",
    "auxiliary_examination",
    "clinical_features",
    "initial_diagnosis",
    "diagnostic_basis",
    "diseases_list",
    "differential_diagnosis_process",
    "final_diagnosis",
}


def test_agent_outputs_merge_to_the_standardized_note_shape(monkeypatch):
    ica_llm = FakeLLM(
        [
            '{"medical_history":"history","physical_examination":"None",'
            '"auxiliary_examination":"tests"}',
            '{"clinical_features":"features"}',
        ]
    )
    monkeypatch.setattr(
        "har.nodes.ica.get_llm",
        lambda: ica_llm,
    )
    state = {"question": "patient question"}
    state.update(ica_extract(state))
    state.update(ica_summarize(state))

    monkeypatch.setattr(
        "har.nodes.pda.get_llm",
        lambda: FakeLLM(
            [
                '{"initial_diagnosis":"stress incontinence",'
                '"diagnostic_basis":"basis"}'
            ]
        ),
    )
    state.update(pda_diagnose(state))
    state.update(dda_retrieve(state))

    dda_llm = FakeLLM(
        [
            '{"differential_diagnosis_process":"process"}',
            '{"flag":true,"final_diagnosis":"stress incontinence"}',
        ]
    )
    monkeypatch.setattr("har.nodes.dda.get_llm", lambda: dda_llm)
    state.update(dda_differentiate(state))
    state.update(dda_reflect(state))

    generated_note = {field: state[field] for field in TEMPLATE_FIELDS}
    assert set(generated_note) == set(standardized_note("stress incontinence"))
