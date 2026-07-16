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
