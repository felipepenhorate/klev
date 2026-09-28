# M12 — the base model is a preset, not a constant

Date: 2026-09-28

`main` used to hardcode a single `MODEL` in `data/config.py`, which meant the Gemma 4 E4B arm
and the Qwen3.5-0.8B arm could not both be run from the same checkout without editing code. It
is now a preset.

## Usage

```bash
python data/prep_dataset.py    --preset e4b         ...
python training/train_distill.py --preset qwen35-08b ...
KLEV_PRESET=e4b python eval/eval_decisions.py ...   # same thing, via the environment
python training/train_distill.py --preset e4b --model some/other-repo ...   # model override
```

| preset | model | delimiters | status |
|---|---|---|---|
| `qwen35-08b` (default) | `Qwen/Qwen3.5-0.8B` | `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`, `<|fim_pad|>`, `<|repo_name|>` (248060..248064) | trained; `docs/m11-qwen35-08b.md` |
| `e4b` | `unsloth/gemma-4-e4b-it-unsloth-bnb-4bit` | `<unused0>`..`<unused4>` (6..10) | the original reference run, `docs/m5-text-run.md` |
| `e2b-qat` | `unsloth/gemma-4-E2B-it-qat-q4_0-unquantized` | `<unused0>`..`<unused4>` (6..10) | **not trainable below ~12 GB VRAM**, `docs/m10-gemma4-e2b.md` |

`--preset` sets the model *and* the delimiters together, because the two must not drift: the
encoder builds each row by concatenating delimiter token strings, and the tokenizer has to
have the same family registered. `--model` still overrides just the path, for a base with the
same vocabulary layout as the preset it is layered onto.

## The import-time trap, and why this needed care

`data/encoding.py` used to do `from .config import DECIDE, OPTION, OPTION_END, QUESTION, STATE`
and build a module constant `_TOKENS` from it. A `from` import captures the value at import
time — before argparse runs — so `--preset e4b` built every row out of Qwen's delimiter
strings while the tokenizer had Gemma's `<unused0>`.. registered. The only symptom was a bare

```
AssertionError: row layout mismatch: {'state': 0, 'q': 0, 'opt': 0, 'opt_end': 0, 'decide': 0}
```

every count zero, which reads like a data problem rather than a preset problem. `encoding.py`
now resolves the delimiters through `_tokens()` at call time, and the entry points import
`from data import config` for the same reason.

Verified end to end on decision-v7 development, both presets:

```
[e4b]        3 rows
[qwen35-08b] 3 rows
```

with delimiter ids `[6, 7, 8, 9, 10]` / bos 2 for `e4b` and
`[248060, 248062, 248061, 248063, 248064]` / bos None for `qwen35-08b`.

## Qwen vs Gemma in the encoder

Only two things differ, and both are already handled: Qwen has no BOS
(`convert_tokens_to_ids("<bos>")` returns `None` rather than raising, which is why
`encode_row` asks the tokenizer for `bos_token_id` and adjusts the state-length arithmetic),
and Qwen has no `embed_tokens_per_layer` (no PLE table), which `model/delimiters.py` already
guards for.
