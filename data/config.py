"""Shared constants, base-model presets and default paths for klev.

The base model is a *preset*, not a constant. klev has been run on two families and the
delimiters differ between them, so a single hardcoded MODEL would have made one of the arms
unreachable without editing code:

  e4b           unsloth/gemma-4-e4b-it-unsloth-bnb-4bit   the original reference run
                (Gemma 4, hidden 2560). 16 GB class card; the 8 GB RX 6600M cannot hold it.
  qwen35-08b    Qwen/Qwen3.5-0.8B                          the arm that ran here: 0.89 GB in
                4-bit, 1.39 GB peak, full 2-epoch decision-v7 in 192 min on that card.
  e2b-qat       unsloth/gemma-4-E2B-it-qat-q4_0-unquantized
                BLOCKED on any card under ~12 GB -- see docs/m10-gemma4-e2b.md. Kept so the
                arm is runnable where it fits, not because it has been trained.

Select with `--preset <name>` on any entry point, or `KLEV_PRESET=<name>` in the environment.
`KLEV_MODEL` still overrides just the model path.

The delimiters are five of the base's own reserved tokens, registered as additional special
tokens. No vocabulary resize and no new embedding rows: the ids already exist, and only their
5 rows (in `embed_tokens`, and in the per-layer table where the model has one) are trained via
peft `trainable_token_indices`. This is Kev's recipe; adding new tokens would transiently
allocate several GB more than the card has. Which five differs per family:

  e4b / e2b-qat   Gemma 4's `<unused0>`..`<unused4>`, ids 6..10. E2B keeps E4B's vocabulary
                  layout exactly, so the encoder, the pointer head and the embedding deltas
                  are reused verbatim between those two.
  qwen35-08b      five of Qwen's own reserved tokens, ids 248060..248064
                  (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`, `<|fim_pad|>`,
                  `<|repo_name|>`); `<|file_sep|>` (248065) is left alone. Qwen has no
                  `<unusedN>` slots, and these are the `<|fim_prefix|>`-style tokens this
                  module has always credited Kev with reusing.

Everything else in the encoder is model-agnostic: `data/encoding.py` asks the tokenizer for
its own `bos_token_id` and adjusts the state-length arithmetic accordingly, because Qwen has
no BOS and returns None for `<bos>` where Gemma has a real one. Qwen3.5 also has no
`embed_tokens_per_layer` (no PLE table), which `model/delimiters.py` already guards for, and
no `final_logit_softcapping`, which the KL's softcap branch already skips.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
RUNS = Path(os.environ.get("KLEV_RUNS", "/mnt/f/distill_jev_runs"))

GEMMA_DELIMITERS = ["<unused0>", "<unused1>", "<unused2>", "<unused3>", "<unused4>"]
QWEN_DELIMITERS = [
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|fim_pad|>",
    "<|repo_name|>",
]

PRESETS = {
    "qwen35-08b": {
        "model": "Qwen/Qwen3.5-0.8B",
        "delimiters": QWEN_DELIMITERS,
        "note": "the alternative arm, trained and measured in docs/m11-qwen35-08b.md. Dense, "
        "1024 hidden, 24 layers, vocab 248,320, tied embeddings, no PLE table, so it fits the "
        "8 GB RX 6600M that the Gemma 4 bases cannot. Ties kev-0.8B in distribution (0.812) and "
        "beats it out of it (0.664 vs 0.613).",
    },
    "qwen35-08b-base": {
        "model": "Qwen/Qwen3.5-0.8B-Base",
        "delimiters": QWEN_DELIMITERS,
        "note": "same recipe as qwen35-08b but on the -Base checkpoint, which is what kev-0.8B "
                "uses. Exists so the stitch comparison is base-matched: the base determines how "
        "strong an external task LoRA can be (probe 0.715 on IT vs 0.660 on -Base), and the "
        "stitch can only import what the adapter has, so an IT-vs-Base comparison of the "
        "stitch is confounded. Same delimiters -- the -Base tokenizer resolves them to the "
        "same ids 248060..248064 -- but it needs its OWN teacher cache, since the KL anchor is "
        "computed from the frozen base's own logits.",
    },
    "e4b": {
        "model": "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit",
        "delimiters": GEMMA_DELIMITERS,
        "note": "the reference recipe, and the default. docs/m5-text-run.md. Gemma 4, hidden "
        "2560, 0.860 dev / 0.751 transfer. Needs a 16 GB class card.",
    },
    "e2b-qat": {
        "model": "unsloth/gemma-4-E2B-it-qat-q4_0-unquantized",
        "delimiters": GEMMA_DELIMITERS,
        "note": "QAT q4_0 weights, shipped unquantized. NOT trained: its [262144, 8960] "
        "per-layer embedding is 4.70 GB of fp16 that bitsandbytes never quantizes, "
        "so it OOMs at 7.73 GB on the RX 6600M. See docs/m10-gemma4-e2b.md.",
    },
}

# The default is e4b: it is the reference recipe -- the one the benchmark table in README.md
# and the comparisons in docs/m5 and m7 describe, and the first one trained. qwen35-08b is the
# alternative arm (docs/m11-qwen35-08b.md): also trained, on a card too small for the Gemma 4
# bases. Reach it with --preset qwen35-08b.
DEFAULT_PRESET = os.environ.get("KLEV_PRESET", "e4b")
if DEFAULT_PRESET not in PRESETS:
    raise SystemExit(
        f"unknown KLEV_PRESET {DEFAULT_PRESET!r}; expected one of {sorted(PRESETS)}"
    )


def apply_preset(name: str = "") -> str:
    """Point this module's MODEL/DELIMITERS at a preset. Mutates in place.

    Call before anything reads `config.MODEL` / `config.DELIMITERS`. An empty name is a no-op,
    which is what makes `--preset` default to "" and stay backward compatible.

    The entry points do `from data import config` rather than `from data.config import MODEL`,
    precisely so this takes effect: a from-import would capture the default at import time,
    before argparse has run.
    """
    global MODEL, DELIMITERS, STATE, QUESTION, OPTION, OPTION_END, DECIDE
    if not name:
        name = DEFAULT_PRESET
    if name not in PRESETS:
        raise SystemExit(f"unknown preset {name!r}; expected one of {sorted(PRESETS)}")
    MODEL = os.environ.get("KLEV_MODEL", PRESETS[name]["model"])
    DELIMITERS = list(PRESETS[name]["delimiters"])
    STATE, QUESTION, OPTION, OPTION_END, DECIDE = DELIMITERS
    return name


apply_preset(DEFAULT_PRESET)

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
