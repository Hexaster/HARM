from dataclasses import dataclass

@dataclass
class Config:
    model_id: str = "DeepSeek-R1"  # DeepSeek-R1; or a Qwen instruct model, e.g. "qwen2.5-7b-instruct"
    max_tokens: int = 4096
    pda_max_iters: int = 5      # paper's optimum (Fig. 7)
    dda_max_iters: int = 5
    ca_max_iters: int = 3      # coordinator correction cap
    note_score_threshold: int = 30   # of 40, for optional note filtering