# M10 — Gemma 4 E2B (QAT q4_0) branch: what works, and why this card cannot train it

Date: 2026-09-27
Branch: `gemma4-e2b-qat` (off `main` @ 84c6a44)
Goal: rerun klev on `unsloth/gemma-4-E2B-it-qat-q4_0-unquantized` to compare against the
E4B main run.

## Verdict

**The code ports cleanly. The hardware does not fit the model.** Nothing about E2B is
*smaller* than E4B where it counts, because the memory is dominated by an embedding table
that is fixed by the architecture rather than by the model size.

## What was reused unchanged

E2B keeps E4B's vocabulary layout exactly, verified against the tokenizer:

| | E4B (main run) | E2B (this branch) |
|---|---|---|
| `vocab_size` | 262,144 | 262,144 |
| `vocab_size_per_layer_input` | 262,144 | 262,144 |
| `<unused0>`..`<unused4>` ids | 6..10 | 6..10 |
| `final_logit_softcapping` | 30.0 | 30.0 |

So `data/encoding.py`, `data/format.py`, `data/metrics.py`, `model/head.py` and
`model/delimiters.py` are reused verbatim. `hidden_size` is already read from config
(2560 → 1536). The only real changes are the base model and the compute dtype.

| | E2B |
|---|---|
| hidden_size / layers / head_dim | 1536 / 35 / 256 |
| total / activated params | 5.10B / ~2B (Gemma-3n style PLE) |
| checkpoint size | 10.21 GB bf16 |

## The blocker: `embed_tokens_per_layer`

From the safetensors header:

| tensor | shape | bf16 | 4-bit |
|---|---|---|---|
| `language_model.embed_tokens_per_layer.weight` | **[262144, 8960]** | **4.70 GB** | n/a |
| `language_model.layers.*` (35) | — | 3.73 GB | 1.86 GB |
| `language_model.embed_tokens.weight` | [262144, 1536] | 0.81 GB | n/a |
| `audio_tower` | — | 0.61 GB | 0.31 GB |
| `vision_tower` | — | 0.34 GB | 0.17 GB |
| **total** | | **10.21 GB** | ~5.10 GB |

Gemma 4's per-layer (PLE) input embedding is a 262,144 × 8,960 table — **bigger than every
other tensor in the model combined**, and it is `Gemma4TextScaledWordEmbedding`, i.e. an
`nn.Embedding`. bitsandbytes replaces `nn.Linear` only, so it is **never quantized**: it
stays fp16 at 4.70 GB, and `embed_tokens` stays fp16 at 0.81 GB. Measured after a real
`FastModel.from_pretrained(..., load_in_4bit=True)`:

```
embed_tokens             [262144, 1536] fp16  cuda:0 -> 0.81 GB
embed_tokens_per_layer   [262144, 8960] fp16  meta  -> 4.70 GB   (offloaded)
GPU total allocated: 3.60 GB of 7.98 GB
```

The 4.70 GB table is offloaded to CPU, so every forward has to stage it back onto the GPU
as one contiguous 4.70 GB allocation. On an 8 GB card that is the whole budget. Forcing
`device_map="cuda:0"` instead OOMs at **7.73 GB allocated** before a single activation — which
matches the table above to within 0.15 GB.

This is not an E2B-vs-E4B difference. The table's width (8,960) and the vocabulary (262,144)
are architecture constants; only the depth (35 layers here) scales down. klev's own code
already knew this — `data/config.py` warns that adding *new* tokens "would transiently
allocate ~6 GB more than the 4080 has", and `model/delimiters.py` exists precisely to avoid
peft's `trainable_token_indices`, which merges over the whole vocabulary every forward.

**Conclusion: the E2B swap buys no VRAM headroom. It needs the same 16 GB class of card the
E4B run used.**

## Second finding: Gemma 4 cannot train in plain fp16

`unsloth_zoo.model_lists.FORCE_FLOAT32` contains `gemma4`:

> fp16 NaNs the grad_norm in the backward (targeted float32 via gemma4_float32.py)

`unsloth/models/vision.py` refuses fp16 for gemma4 unless `UNSLOTH_FORCE_FLOAT32=1`, which
loads bf16 weights, computes matmuls in fp16 and keeps the residual stream in float32
(`gemma4_float32.py` patches ScaledWordEmbedding / RMSNorm / TextAttention). So
`scripts/rocm_env.sh` now sets it, and `model/load.py:compute_dtype()` returns bf16 on a bf16
card and fp16 here.

The `unsloth-fix.sh` claim that gfx1032 "lacks bf16 fdot2" is true only of **Triton**:
measured directly, native bf16 matmul + backward on this RX 6600M is finite and correct.
What aborts is `LLVM ERROR: Cannot select: %llvm.amdgcn.fdot2.bf16.bf16` from Triton's fused
kernels, hence `UNSLOTH_USE_TRITON=0`. The claim in that script's header ("PyTorch reports
bf16 as supported ... but the GPU lacks the bf16 fdot2 instruction") is misleading: it
conflates the two paths, and the `.venv-e2b` sitecustomize built from it disables bf16
entirely when native bf16 would have worked.

## Environment

`.venv` could never have run klev: it has **transformers 5.3.0, which has no `gemma4`
module at all** (`AutoConfig` raises "Transformers does not recognize this architecture",
which unsloth then misreports as a generic "not supported yet" ValueError). There is no
gemma4 implementation vendored in unsloth 2026.3.18 or unsloth_zoo 2026.3.7 either. The E4B
run was on a different machine.

The version constraint is genuinely tight:

- gemma4 landed in **transformers 5.5.0** (5.4.0 has no `models/gemma4`).
- unsloth 2026.9.11 and `main` both still cap at `transformers<=5.5.0`.
- but **5.5.0 is the release where a prequantized bnb-4bit checkpoint loses `quant_state`**
  (unslothai/unsloth#9867 etc.), and the **Gemma 4 LoRA fix only shipped in 5.5.2**
  (unslothai/unsloth#5355), one patch release out of reach of the cap
  (unslothai/unsloth-zoo#1227, closed but the ceiling is not lifted).

`.venv-e2b` therefore runs **transformers 5.6.2** (the checkpoint's own provenance, and
past the 5.5.2 LoRA fix) installed with `--no-deps`, plus **unsloth 2026.9.11 /
unsloth_zoo 2026.9.7**, which do carry gemma4 support (5 patch modules, 26 mapper entries).
`--no-deps` matters: unsloth's `pyproject.toml` would otherwise pull CUDA `xformers` and
replace the ROCm torch build.

`.venv-e2b` is a real copy of `.venv` (60,520 files, link count 1 — a hardlink clone was
tried first and abandoned, because a pip install writes through hardlinked inodes and would
have corrupted the only working env). The gfx1032 bitsandbytes build and the fp16
sitecustomize are both carried over. The host is btrfs, so the copy cost ~3 GB, not 14 GB.

## What was verified working

- `FastModel.from_pretrained` loads the E2B checkpoint in 4-bit: `Gemma4ForConditionalGeneration`,
  param dtype fp16, 3.60 GB VRAM, no errors.
- `FastModel.get_peft_model(r=16, alpha=32, 7 target modules)` attaches: 706 trainable
  tensors, 29.86 M params (the reference E4B run had 42.5 M, so the smaller `hidden_size`
  accounts for the difference).
- Dataset prep is model-agnostic and ran clean against decision-v7: 2,000 rows,
  p50 114 / p90 305 / p99 939 / p100 1010 tokens, 61 garbage rows (3.05%, vs 3.2% recorded
  for the full run).

Not reached: the training forward, the teacher cache, and the eval — all blocked on VRAM.

## To finish this on adequate hardware

Needs a 16 GB class CUDA card (what the E4B run used). On such a card the branch should work
as-is: the model change and the dtype resolver are the only code deltas. Two things to
re-check there, both cheap:

1. Whether the true 4-bit path is taken. With no finetuning method requested at load time,
   unsloth logged `QLoRA and full finetuning all not selected. Switching to 16bit LoRA` and
   the sampled `q_proj` came back fp16 with `quant_state=False`. klev's
   `load_in_4bit=True` + `get_peft_model` ordering reproduces that, and the reference run
   reported 4-bit — so confirm the adapter's `quantization_config` is really NF4 before
   trusting a size or a speed number.
2. `UNSLOTH_FORCE_FLOAT32` should be **unset** on a bf16 card; `compute_dtype()` returns
   bf16 there and the two must stay consistent.
