#!/usr/bin/env python
"""Is peft really summing two active adapters? Compare logits under each activation mode."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unsloth  # noqa: F401

import torch
from datasets import load_from_disk

from eval.eval_decisions import load_checkpoint

model, tokenizer, head = load_checkpoint(Path("/mnt/f/distill_jev_runs/main"), "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit")
model.load_adapter("armand0e/Gemma-4-E4B-it-Fable-Distill-LoRA", adapter_name="fable")
device = next(model.parameters()).device
row = load_from_disk("/mnt/f/distill_jev_runs/prep/dv7-dev")[0]
ids = torch.tensor([row["input_ids"]], device=device)


def logits():
    with torch.no_grad():
        hidden = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[-1][0]
        return model.head(hidden[row["decide_pos"]].float(), hidden[torch.tensor(row["opt_pos"], device=device)].float())


def active():
    layer = model.base_model.model.model.language_model.layers[0].self_attn.q_proj
    return getattr(layer, "active_adapters", None) or getattr(layer, "_active_adapters", None)


model.set_adapter("default"); a = logits()
model.set_adapter("fable"); c = logits()
model.set_adapter("default"); model.base_model.set_adapter(["default", "fable"]); b = logits()
print("layer active_adapters after base_model.set_adapter(list):", active(), flush=True)
print("model.active_adapter:", model.active_adapter, flush=True)
print("max |ours - sum| :", float((a - b).abs().max()))
print("max |ours - fable|:", float((a - c).abs().max()))
print("logits ours :", [round(float(x), 4) for x in a])
print("logits sum  :", [round(float(x), 4) for x in b])
print("logits fable:", [round(float(x), 4) for x in c])
