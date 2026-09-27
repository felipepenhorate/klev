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

Unlike E2B, Qwen3.5 (branch qwen35-08b) has no `<unusedN>` slots and no per-layer embedding
table, so it needs its own delimiters and a no-BOS path in encode_row.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
RUNS = Path(os.environ.get("KLEV_RUNS", "/mnt/f/distill_jev_runs"))

# Base for this branch: Gemma 4 E2B IT, QAT-trained for q4_0 and shipped unquantized
# (Apache-2.0). The QAT weights are calibrated to survive 4-bit, so re-quantizing them with
# bitsandbytes NF4 at load costs much less accuracy than it would on the E4B base -- that is
# the point of the branch, and the reason this is not a pure size swap. It is still 5.1B
# total / 2B activated (Gemma-3n style PLE), 10.2 GB in bf16, so `load_in_4bit=True` is
# mandatory on any card under 12 GB.
#
# It does NOT fit the 8 GB RX 6600M, and not because of the parameter count: Gemma 4's
# per-layer (PLE) input embedding is [262144, 8960] = 4.70 GB of fp16, larger than every
# other tensor combined, and bitsandbytes replaces nn.Linear only, so it is never quantized.
# See docs/m10-gemma4-e2b.md.
MODEL = os.environ.get("KLEV_MODEL", "unsloth/gemma-4-E2B-it-qat-q4_0-unquantized")

# E2B keeps E4B's vocabulary layout exactly, so `<unused0>`..`<unused4>` are still at ids
# 6..10 and the encoder, the pointer head and the embedding deltas are reused verbatim.
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
