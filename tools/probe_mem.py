#!/usr/bin/env python
"""Find what allocates the 8 GB on a long row: forward-only vs grad-forward vs backward, plus a
memory-history snapshot of the biggest blocks. Run:
    ./unsloth_uv/bin/python tools/probe_mem.py --gc unsloth
"""
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unsloth  # noqa: F401

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from data.config import MODEL
from training.train_distill import DistillCollator, build_model

DATASET = "/mnt/f/distill_jev_runs/prep/dv7-smoke"


def snapshot_top(n=12):
    snap = torch.cuda.memory._snapshot()
    blocks = []
    for segment in snap["segments"]:
        for block in segment["blocks"]:
            if block["state"] != "active_allocated":
                continue
            frames = block.get("frames") or []
            where = " <- ".join(f"{f['name']} ({f['filename'].split('/')[-1]}:{f['line']})" for f in frames[:2])
            blocks.append((block["size"], where))
    blocks.sort(reverse=True)
    for size, where in blocks[:n]:
        print(f"[mem-snap] {size / 1e9:.2f} GB  {where}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc", choices=["none", "unsloth"], default="unsloth")
    a = ap.parse_args()
    dataset = load_from_disk(DATASET)
    order = sorted(range(len(dataset)), key=lambda i: len(dataset[i]["input_ids"]))
    row = dataset[order[-3]]
    cfg = SimpleNamespace(model=MODEL, max_seq_length=2048, lora_r=16, lora_alpha=32, head_dim=256,
                          garbage=True, garbage_bias=-4.0, seed=0, no_gc=(a.gc == "none"), gc_mode=a.gc, kl_weight=0.3)
    model, tokenizer, head, deltas = build_model(cfg)
    model.train()
    batch = DistillCollator()([row])
    batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
    print(f"[mem] row {len(row['input_ids'])} tokens, {row['n_options']} options, gc={a.gc}", flush=True)

    def reset():
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated() / 1e9

    def forward(grad):
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        output_hidden_states=True, use_cache=False)
            return out.hidden_states[-1]

    base = reset()
    start = time.time()
    h = forward(False)
    print(f"[mem] no_grad forward: peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB ({time.time() - start:.1f}s)", flush=True)
    del h

    base = reset()
    start = time.time()
    h = forward(True)
    print(f"[mem] grad forward: peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB ({time.time() - start:.1f}s)", flush=True)

    print("[mem] top active allocations during grad forward:", flush=True)
    snapshot_top()
    logits = model.head(h[0, int(batch["decide_pos"][0])].float(), h[0, batch["opt_pos"][:int(batch["n_opts"][0])]].float())
    loss = -(batch["target"][:int(batch["n_opts"][0]) + 1] * F.log_softmax(logits, -1)).sum()
    start = time.time()
    loss.backward()
    print(f"[mem] after backward: peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB ({time.time() - start:.1f}s)", flush=True)
    print("[mem] top active allocations after backward:", flush=True)
    snapshot_top()

    del h, logits, loss
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
