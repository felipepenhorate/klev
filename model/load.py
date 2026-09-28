"""Model loading helpers: delimiter resizing and module lookup.

Gemma 4 has two token embedding tables: `embed_tokens` (tied to `lm_head`) and
`embed_tokens_per_layer` (the per-layer input embeddings, `vocab_size_per_layer_input`).
`resize_token_embeddings` only knows about the first, and a token id past the old vocab
raises a CUDA gather assert in the per-layer lookup, so both are resized here.
"""
import os

import torch
import torch.nn as nn


def compute_dtype() -> torch.dtype:
    """bf16 where the card can do bf16 matmul, fp16 otherwise (see scripts/rocm_env.sh).

    Gemma 4 cannot be trained in plain fp16: unsloth lists `gemma4` in
    `unsloth_zoo.model_lists.FORCE_FLOAT32` because fp16 NaNs the grad_norm in the backward
    (activations overflow fp16's ~6.5e4 range) even though forward peaks stay near 350. The
    fix is `UNSLOTH_FORCE_FLOAT32=1`, which unsloth reads itself: it loads bf16 weights,
    runs matmuls in fp16 and keeps the residual stream in float32 via
    `unsloth_zoo/temporary_patches/gemma4_float32.py`. So on a bf16 card this returns bf16
    and that env var stays unset; on gfx103x it returns fp16 and the env var must be set.

    Native bf16 matmul *does* work on gfx103x (measured: fwd+bwd finite). What does not
    work there is bf16 inside Triton's fused kernels -- no bf16 `fdot2` instruction --
    hence UNSLOTH_USE_TRITON=0. `.venv-e2b`'s sitecustomize makes unsloth's
    `is_bf16_supported()` probe honest so this function routes correctly.
    """
    # Explicit override first. Do NOT trust torch.cuda.is_bf16_supported() here: importing
    # unsloth replaces it (unsloth/__init__.py:507, _gpu_init.py:415-431) and on this ROCm
    # box the replacement reports False for consumer Navi, so the probe silently forces fp16
    # in every run. Native bf16 matmul+backward is in fact fine on gfx1032 (measured finite);
    # it is only Triton's fused bf16 kernels that abort, which rocm_env.sh handles with
    # TORCH_ROCM_AOTRITON_ENABLE_BF16=0. So KLEV_DTYPE is the honest control.
    forced = os.environ.get("KLEV_DTYPE", "").strip().lower()
    if forced in ("bf16", "bfloat16"):
        return torch.bfloat16
    if forced in ("fp16", "float16"):
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def language_model(model):
    """The text backbone under whatever wrapper (PeftModel/Gemma4ForConditionalGeneration)."""
    inner = getattr(model, "model", model)
    return getattr(inner, "language_model", inner)


def resize_for_delimiters(model, tokenizer, init: str = "mean"):
    """Grow every vocabulary-sized table to len(tokenizer). New rows are initialised like
    transformers' mean resizing (mean of the existing rows) or small normal noise for the
    per-layer table. Returns (old_vocab, new_vocab)."""
    target = len(tokenizer)
    old = getattr(language_model(model), "embed_tokens", model.get_input_embeddings()).num_embeddings
    if target == old:
        return old, old
    model.resize_token_embeddings(target)
    text_config = getattr(model.config, "text_config", model.config)
    if getattr(text_config, "vocab_size_per_layer_input", None):
        per_layer = language_model(model).embed_tokens_per_layer
        if per_layer.num_embeddings != target:
            grown = type(per_layer)(target, per_layer.embedding_dim, per_layer.padding_idx,
                                    embed_scale=float(per_layer.embed_scale))
            grown = grown.to(device=per_layer.weight.device, dtype=per_layer.weight.dtype)
            with torch.no_grad():
                grown.weight[:per_layer.num_embeddings] = per_layer.weight
                if init == "mean":
                    grown.weight[per_layer.num_embeddings:] = per_layer.weight.mean(0, keepdim=True)
                else:
                    nn.init.normal_(grown.weight[per_layer.num_embeddings:], std=0.02)
            language_model(model).embed_tokens_per_layer = grown
        text_config.vocab_size_per_layer_input = target
    return old, target
