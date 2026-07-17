"""Tests for the Coordinator Agent and its hierarchical route-back
(paper Sec. 3.2; plan Task 6)."""


from types import SimpleNamespace

from har.config import Config
from har.nodes.coordinator import assemble, ca_match, ca_reflect, route_after_ca

from .conftest import PAPER_CASE


def test_ca_match_fetches_the_standardized_note_for_the_final_diagnosis():
    state = {"final_diagnosis": "stress incontinence"}
    assert ca_match(state) == {"_std_note": PAPER_CASE}


def test_ca_reflect_passes_when_checks_agree(monkeypatch):
    monkeypatch.setattr(
        "har.nodes.coordinator._run_ca_checks",
        lambda raw, std: SimpleNamespace(flag=True, ica_error=None, pda_error=None, dda_error=None),
    )
    state = {"_std_note": PAPER_CASE, "ca_iters": 0}
    result = ca_reflect(state)
    assert result["ca_target"] is None
    assert result["ca_iters"] == 1


def test_ca_reflect_routes_back_to_ica_on_ica_error(monkeypatch):
    monkeypatch.setattr(
        "har.nodes.coordinator._run_ca_checks",
        lambda raw, std: SimpleNamespace(
            flag=False, ica_error="clinical features omit the coronary intervention", pda_error=None, dda_error=None
        ),
    )
    state = {"_std_note": PAPER_CASE, "ca_iters": 0}
    result = ca_reflect(state)
    assert result["ca_target"] == "ica"
    assert result["ca_feedback"]


def test_ca_reflect_routes_back_to_pda_and_resets_its_counter(monkeypatch):
    monkeypatch.setattr(
        "har.nodes.coordinator._run_ca_checks",
        lambda raw, std: SimpleNamespace(
            flag=False, ica_error=None, pda_error="diagnostic basis is unsupported", dda_error=None
        ),
    )
    state = {"_std_note": PAPER_CASE, "ca_iters": 0, "pda_iters": 5}
    result = ca_reflect(state)
    assert result["ca_target"] == "pda"
    assert result["pda_error"] == "diagnostic basis is unsupported"
    assert result["pda_iters"] == 0


def test_ca_reflect_routes_back_to_dda_and_resets_its_counter(monkeypatch):
    monkeypatch.setattr(
        "har.nodes.coordinator._run_ca_checks",
        lambda raw, std: SimpleNamespace(
            flag=False, ica_error=None, pda_error=None, dda_error="overflow incontinence not excluded"
        ),
    )
    state = {"_std_note": PAPER_CASE, "ca_iters": 0, "dda_iters": 5}
    result = ca_reflect(state)
    assert result["ca_target"] == "dda"
    assert result["dda_error"] == "overflow incontinence not excluded"
    assert result["dda_iters"] == 0


def test_route_after_ca_advances_to_assemble_once_checks_pass():
    state = {"ca_target": None, "ca_iters": 1}
    assert route_after_ca(state) == "assemble"


def test_route_after_ca_stops_at_iteration_cap_even_with_a_target():
    state = {"ca_target": "ica", "ca_iters": Config.CA_MAX_ITERS}
    assert route_after_ca(state) == "assemble"


def test_route_after_ca_routes_to_the_flagged_agent():
    assert route_after_ca({"ca_target": "ica", "ca_iters": 1}) == "ica_summarize"
    assert route_after_ca({"ca_target": "pda", "ca_iters": 1}) == "pda_diagnose"
    assert route_after_ca({"ca_target": "dda", "ca_iters": 1}) == "dda_differentiate"


def test_assemble_produces_a_note_containing_the_final_diagnosis():
    state = {
        "question": "patient narrative",
        **{k: v for k, v in PAPER_CASE.items() if k not in ("reviewed", "source")},
    }
    result = assemble(state)
    assert "clinical_note" in result
    note = result["clinical_note"]
    assert isinstance(note, str) and note.strip()
    assert PAPER_CASE["final_diagnosis"] in note
