"""Model loading helpers: delimiter resizing and module lookup.

Gemma 4 has two token embedding tables: `embed_tokens` (tied to `lm_head`) and
`embed_tokens_per_layer` (the per-layer input embeddings, `vocab_size_per_layer_input`).
`resize_token_embeddings` only knows about the first, and a token id past the old vocab
raises a CUDA gather assert in the per-layer lookup, so both are resized here.
"""
import torch
import torch.nn as nn


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
