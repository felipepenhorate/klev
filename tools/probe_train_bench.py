#!/usr/bin/env python
"""Benchmark the training step: where does the time and memory go?

Loads the model once per --gc mode, then times a forward+backward for a short and a long
prepared row (pointer only, then with the teacher forward + KL).

    /home/penhfel/unsloth_uv/bin/python tools/probe_train_bench.py --gc unsloth
"""
import argparse
import json
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
from training.train_distill import DistillCollator, DistillTrainer, build_model

DATASET = "/mnt/f/distill_jev_runs/prep/dv7-smoke"


class KlHelper:
    kl_chunk = 128
    teacher_temperature = 1.0
    content_kl = DistillTrainer.content_kl


HELPER = KlHelper()


def timing(fn):
    torch.cuda.synchronize()
    start = time.time()
    out = fn()
    torch.cuda.synchronize()
    return time.time() - start, out


def make_step(a, model, batch, teacher=True, kl=True):
    def step():
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[-1]
        terms = []
        for i in range(hidden.shape[0]):
            start, n = int(batch["opt_start"][i]), int(batch["n_opts"][i])
            logits = model.head(hidden[i, int(batch["decide_pos"][i])].float(), hidden[i, batch["opt_pos"][start:start + n]].float())
            terms.append(-(batch["target"][start:start + n + 1] * F.log_softmax(logits.float(), -1)).sum())
        loss = torch.stack(terms).mean()
        extra = {}
        if teacher:
            with torch.no_grad(), model.disable_adapter():
                t_out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                              output_hidden_states=True, use_cache=False)
            if kl:
                kl_loss = HELPER.content_kl(model, hidden, t_out.hidden_states[-1], batch["content_mask"])
                loss = loss + a.kl_weight * kl_loss
                extra["kl"] = float(kl_loss)
        loss.backward()
        return float(loss), extra
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc", choices=["none", "unsloth"], default="unsloth")
    ap.add_argument("--rows", default="short,long")
    ap.add_argument("--enable-gc", action="store_true", help="call gradient_checkpointing_enable() again explicitly")
    a = ap.parse_args()
    device = "cuda"
    dataset = load_from_disk(DATASET)
    lengths = [len(x["input_ids"]) for x in dataset]
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    indices = {"short": order[len(order) // 4], "long": order[-3]}
    collator = DistillCollator()

    cfg = SimpleNamespace(model=MODEL, max_seq_length=2048, lora_r=16, lora_alpha=32, head_dim=256,
                          garbage=True, garbage_bias=-4.0, seed=0, no_gc=(a.gc == "none"), gc_mode=a.gc, kl_weight=0.3)
    model, tokenizer, head, deltas = build_model(cfg)
    model.train()
    base = getattr(model, "base_model", model)
    inner = getattr(base, "model", base)          # Gemma4ForConditionalGeneration
    gemma = getattr(inner, "model", inner)         # Gemma4Model
    lm = getattr(gemma, "language_model", gemma)   # Gemma4TextModel
    layer0 = lm.layers[0]
    print(f"[gc] model.is_gradient_checkpointing={getattr(model, 'is_gradient_checkpointing', None)} "
          f"base={getattr(base, 'is_gradient_checkpointing', None)} layer0.gradient_checkpointing={getattr(layer0, 'gradient_checkpointing', None)} "
          f"use_cache={model.config.use_cache}", flush=True)
    if a.enable_gc:
        model.gradient_checkpointing_enable()
        print(f"[gc] after explicit enable: model={getattr(model, 'is_gradient_checkpointing', None)} "
              f"base={getattr(base, 'is_gradient_checkpointing', None)} layer0={getattr(layer0, 'gradient_checkpointing', None)}", flush=True)

    results = {}
    for label in [x for x in a.rows.split(",") if x]:
        index = indices[label]
        row = dataset[index]
        batch = collator([row])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        print(f"[bench:{a.gc}] {label} row: {len(row['input_ids'])} tokens, {row['n_options']} options, "
              f"content positions {sum(row['content_mask'])}", flush=True)
        for name, teacher, kl in (("pointer", False, False), ("full", True, True)):
            try:
                model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                seconds, out = timing(make_step(cfg, model, batch, teacher, kl))
                results[f"{label}_{name}"] = {"seconds": round(seconds, 3), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                                              "loss": round(out[0], 4)}
                print(f"[bench:{a.gc}] {label} {name}: {seconds:.2f}s, peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB, loss {out[0]:.4f}", flush=True)
            except Exception as e:
                results[f"{label}_{name}"] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
                print(f"[bench:{a.gc}] {label} {name} FAILED: {results[f'{label}_{name}']['error']}", flush=True)
                torch.cuda.empty_cache()
    out = ROOT / "runs" / f"probe_train_bench_{a.gc}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
