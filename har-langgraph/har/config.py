from dataclasses import dataclass

@dataclass
class Config:
    MODEL_ID: str = "DeepSeek-R1"  # DeepSeek-R1; or a Qwen instruct model, e.g. "qwen2.5-7b-instruct"
    MAX_TOKENS: int = 4096
    PDA_MAX_ITERS: int = 5      # paper's optimum (Fig. 7)
    DDA_MAX_ITERS: int = 5
    CA_MAX_ITERS: int = 3      # coordinator correction cap
    NOTE_SCORE_THRESHOLD: int = 30   # of 40, for optional note filtering