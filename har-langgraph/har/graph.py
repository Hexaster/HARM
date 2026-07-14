"""Compile the HAR pipeline into a single LangGraph StateGraph: ICA -> PDA
(reflect loop) -> DDA (reflect loop) -> Coordinator (hierarchical route-back
to ICA/PDA/DDA) -> assemble (paper Algorithm 1).
"""
from langgraph.graph import END, START, StateGraph

from har.nodes import coordinator as ca
from har.nodes import dda, ica, pda
from har.state import ClinicalNoteState


def build_har_graph():
    g = StateGraph(ClinicalNoteState)
    for name, fn in [
        ("ica_extract", ica.ica_extract),
        ("ica_summarize", ica.ica_summarize),
        ("pda_diagnose", pda.pda_diagnose),
        ("pda_reflect", pda.pda_reflect),
        ("dda_retrieve", dda.dda_retrieve),
        ("dda_differentiate", dda.dda_differentiate),
        ("dda_reflect", dda.dda_reflect),
        ("ca_match", ca.ca_match),
        ("ca_reflect", ca.ca_reflect),
        ("assemble", ca.assemble),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "ica_extract")
    g.add_edge("ica_extract", "ica_summarize")
    g.add_edge("ica_summarize", "pda_diagnose")
    g.add_edge("pda_diagnose", "pda_reflect")
    g.add_conditional_edges(
        "pda_reflect", pda.route_after_pda_reflect,
        {"pda_diagnose": "pda_diagnose", "dda_retrieve": "dda_retrieve"},
    )
    g.add_edge("dda_retrieve", "dda_differentiate")
    g.add_edge("dda_differentiate", "dda_reflect")
    g.add_conditional_edges(
        "dda_reflect", dda.route_after_dda_reflect,
        {"dda_differentiate": "dda_differentiate", "ca_match": "ca_match"},
    )
    g.add_edge("ca_match", "ca_reflect")
    g.add_conditional_edges(
        "ca_reflect", ca.route_after_ca,
        {
            "ica_summarize": "ica_summarize",
            "pda_diagnose": "pda_diagnose",
            "dda_differentiate": "dda_differentiate",
            "assemble": "assemble",
        },
    )
    g.add_edge("assemble", END)
    return g.compile()
