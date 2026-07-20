from langgraph.graph import StateGraph, START, END
from har.state import ClinicalNoteState
from har.nodes import ica, pda, dda, coordinator as ca

def build_har_graph() -> StateGraph[ClinicalNoteState]:
    graph = StateGraph(ClinicalNoteState)

    for name, fn in [("ica_extract", ica.ica_extract), ("ica_summarize", ica.ica_summarize),
                     ("pda_diagnose", pda.pda_diagnose), ("pda_reflect", pda.pda_reflect),
                     ("dda_retrieve", dda.dda_retrieve), ("dda_differentiate", dda.dda_differentiate),
                     ("dda_reflect", dda.dda_reflect), ("ca_match", ca.ca_match),
                     ("ca_reflect", ca.ca_reflect), ("assemble", ca.assemble)]:
        graph.add_node(name, fn)

    graph.add_edge(START, "ica_extract")
    graph.add_edge("ica_extract", "ica_summarize")
    graph.add_edge("ica_summarize", "pda_diagnose")
    graph.add_edge("pda_diagnose", "pda_reflect")
    graph.add_conditional_edges("pda_reflect", pda.route_after_pda_reflect,
                                {"pda_diagnose": "pda_diagnose", "dda_retrieve": "dda_retrieve"})

    graph.add_edge("dda_retrieve", "dda_differentiate")
    graph.add_edge("dda_differentiate", "dda_reflect")
    graph.add_conditional_edges(
        "dda_reflect", dda.route_after_dda_reflect,
        {"dda_differentiate": "dda_differentiate", "ca_match": "ca_match"}
    )

    graph.add_edge("ca_match", "ca_reflect")
    graph.add_conditional_edges(
        "ca_reflect", ca.route_after_ca,
        {"ica_summarize": "ica_summarize", 
         "pda_diagnose": "pda_diagnose", 
         "dda_differentiate": "dda_differentiate",
         "assemble": "assemble"
         }
    )

    graph.add_edge("assemble", END)
    return graph.compile()