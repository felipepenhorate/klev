"""Trainable embedding rows for the delimiter tokens, without copying the tables.

peft's `trainable_token_indices` (TrainableTokensLayer) merges `base_weight + delta` over the
whole vocabulary on every forward: on Gemma 4 that is ~8.3 GB of transient allocation per pass
(embed_tokens 2.7 GB + embed_tokens_per_layer 5.6 GB), which does not fit a 16 GB card. These
wrappers add the delta only at the delimiter rows, which is free when the ids are absent.

The tables stay where they are (weight tying to `lm_head` is untouched: `base` is the original
module and `weight` points at its parameter), and the deltas are saved in `head.pt`.
"""
import torch
import torch.nn as nn


class EmbeddingDelta(nn.Module):
    """nn.Embedding that adds a trainable delta to a contiguous block of token ids."""

    def __init__(self, base: nn.Embedding, indices: list[int]):
        super().__init__()
        self.base = base
        self.first = int(min(indices))
        self.count = int(max(indices)) - self.first + 1
        # float32, not base.weight.dtype: the delta is a *trainable* parameter, and under
        # fp16 AMP the GradScaler refuses fp16 grads ("Attempting to unscale FP16
        # gradients."). It is only 5 rows, so fp32 costs nothing, and forward() casts back
        # to the embedding dtype anyway. (bf16 training never hit this: no scaler.)
        self.delta = nn.Parameter(torch.zeros(self.count, base.embedding_dim,
                                              dtype=torch.float32, device=base.weight.device))

    @property
    def weight(self):
        return self.base.weight

    @property
    def num_embeddings(self):
        return self.base.num_embeddings

    @property
    def embedding_dim(self):
        return self.base.embedding_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.base(input_ids)
        mask = (input_ids >= self.first) & (input_ids < self.first + self.count)
        if mask.any():
            out[mask] = out[mask] + self.delta[input_ids[mask] - self.first].to(out.dtype)
        return out


def apply_delimiter_deltas(model, language_model_fn, indices: list[int]) -> dict:
    """Replace `embed_tokens` and `embed_tokens_per_layer` with EmbeddingDelta wrappers.
    Returns {module name: delta parameter} for checkpointing."""
    lm = language_model_fn(model)
    lm.embed_tokens = EmbeddingDelta(lm.embed_tokens, indices)
    deltas = {"embed_tokens": lm.embed_tokens.delta}
    if getattr(lm, "embed_tokens_per_layer", None) is not None:
        lm.embed_tokens_per_layer = EmbeddingDelta(lm.embed_tokens_per_layer, indices)
        deltas["embed_tokens_per_layer"] = lm.embed_tokens_per_layer.delta
    return deltas
