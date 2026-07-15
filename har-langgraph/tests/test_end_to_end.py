"""End-to-end integration test for the compiled HAR graph (plan Task 7).

This is the correctness gate: it runs the paper's own worked case (p.8)
through the full ICA -> PDA -> DDA -> Coordinator pipeline and checks the
result matches the paper's ground truth. Requires a live LLM endpoint.
"""
from har.graph import build_har_graph

from .conftest import PATIENT_QUESTION, requires_llm

NOTE_FIELDS = (
    "medical_history",
    "physical_examination",
    "auxiliary_examination",
    "clinical_features",
    "initial_diagnosis",
    "diagnostic_basis",
    "disease_list",
    "differential_process",
    "final_diagnosis",
)


def test_graph_compiles_with_the_expected_nodes():
    graph = build_har_graph()
    node_names = set(graph.get_graph().nodes)
    assert {
        "ica_extract",
        "ica_summarize",
        "pda_diagnose",
        "pda_reflect",
        "dda_retrieve",
        "dda_differentiate",
        "dda_reflect",
        "ca_match",
        "ca_reflect",
        "assemble",
    }.issubset(node_names)


@requires_llm
def test_full_pipeline_reproduces_the_papers_worked_case():
    graph = build_har_graph()
    result = graph.invoke({"question": PATIENT_QUESTION})

    for field in NOTE_FIELDS:
        assert result.get(field), f"missing or empty note field: {field}"

    final_diagnosis = result["final_diagnosis"].lower()
    assert "stress incontinence" in final_diagnosis
    assert "overactive bladder" in final_diagnosis
