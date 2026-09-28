"""Shared constants, delimiters and default paths for distill_jev.

The delimiters are five of Gemma 4's reserved slots (`<unused0>`..`<unused4>`, token ids
6..10) registered as additional special tokens. No vocabulary resize and no new embedding
rows: the ids already exist, and only their 5 rows (in `embed_tokens` and in the 2.8B-row
`embed_tokens_per_layer` table) are trained via peft `trainable_token_indices`. This is Kev's
recipe (it reuses Qwen's `<|fim_prefix|>`-style reserved tokens the same way); adding new
tokens would transiently allocate ~6 GB more than the 4080 has.

E2B keeps E4B's vocabulary layout exactly (262,144 rows, `<unused0>`..`<unused4>` still at
ids 6..10), so the encoder, the pointer head and the embedding deltas are reused verbatim
between the two runs -- only MODEL and the compute dtype differ.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
RUNS = Path(os.environ.get("KLEV_RUNS", "/mnt/f/distill_jev_runs"))

# Base for this branch: Qwen3.5 0.8B IT. Dense (no PLE table), 1024 hidden, 24 layers,
# vocab 248,320, tied embeddings, no final_logit_softcapping -- so the KL's softcap branch
# is simply skipped and `embed_tokens_per_layer` does not exist (delimiters.py already
# guards for that). 0.89 GB in 4-bit, so it trains on the 8 GB RX 6600M, which the Gemma 4
# arms cannot: Gemma 4's [262144, 8960] per-layer embedding is 4.70 GB of unquantizable
# fp16 (see docs/m10-gemma4-e2b.md).
#
# Qwen3.5 is in unsloth's FORCE_FLOAT32 list, so scripts/rocm_env.sh callers must set
# UNSLOTH_FORCE_FLOAT32=1 (plain fp16 NaNs the grad_norm in the backward).
MODEL = os.environ.get("KLEV_MODEL", "Qwen/Qwen3.5-0.8B")

# Qwen has no `<unusedN>` slots, so the delimiters are five of Qwen's own reserved tokens --
# the `<|fim_*|>` family Kev uses for the same purpose (`data/config.py` has said so all
# along: "it reuses Qwen's `<|fim_prefix|>`-style reserved tokens the same way"). They exist
# in the vocabulary already, so `add_special_tokens` registers them without a resize and
# only these 5 rows are trained. Ids 248060..248064; `<|file_sep|>` (248065) is left alone.
DELIMITERS = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|fim_pad|>", "<|repo_name|>"]
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
