from typing import TypedDict, Optional

class ClinicalNoteState(TypedDict, total=False):
    question: str                # x_i
    medical_history: str         # y1
    physical_examination: str    # y2
    auxiliary_examination: str   # y3
    clinical_features: str       # y4
    initial_diagnosis: str       # y5
    diagnostic_basis: str        # y6
    diseases_list: list[str]      # y7
    differential_diagnosis_process: str  # y8
    final_diagnosis: str         # y9
    pda_iters: int
    dda_iters: int
    ca_iters: int
    pda_error: Optional[str]     # feedback for next PDA pass
    dda_error: Optional[str]
    ca_target: Optional[str]     # "ica" | "pda" | "dda" | None
    ca_feedback: Optional[str]
    # LangGraph drops any key a node returns that is not declared here, so
    # these must stay in the schema even though they are not note fields.
    _std_note: dict              # ca_match -> ca_reflect
    clinical_note: str           # assemble -> caller
