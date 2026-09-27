"""Shared constants, delimiters and default paths for distill_jev.

The delimiters are five of Gemma 4's reserved slots (`<unused0>`..`<unused4>`, token ids
6..10) registered as additional special tokens. No vocabulary resize and no new embedding
rows: the ids already exist, and only their 5 rows (in `embed_tokens` and in the 2.8B-row
`embed_tokens_per_layer` table) are trained via peft `trainable_token_indices`. This is Kev's
recipe (it reuses Qwen's `<|fim_prefix|>`-style reserved tokens the same way); adding new
tokens would transiently allocate ~6 GB more than the 4080 has.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
RUNS = Path("/mnt/f/distill_jev_runs")

MODEL = "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit"

DELIMITERS = ["<unused0>", "<unused1>", "<unused2>", "<unused3>", "<unused4>"]
STATE, QUESTION, OPTION, OPTION_END, DECIDE = DELIMITERS

# training context (Kev's numbers: state 384, branch 1024, packed 2048; the row form uses
# --max-seq for the whole row: state + one branch)
MAX_SEQ = 2048
MAX_STATE = 384
MAX_BRANCH = 1024

DEFAULTS = {
    "max_seq_length": MAX_SEQ,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "kl_weight": 0.3,
    "teacher_temperature": 1.0,
    "top_k": 32,
    "feature_weight": 0.0,
    "anchor_weight": 0.3,
    "garbage": True,
    "garbage_bias": -4.0,
    "garbage_frac": 0.1,
    "lr": 1e-4,
    "embedding_lr": 2e-4,
}
