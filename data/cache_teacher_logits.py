#!/usr/bin/env python
"""Offline prefix teacher cache (SPEC 3.4, D2): the frozen base's top-k next-token
distributions at every content position of every prepared row.

Gemma 4 E-series KV sharing cannot run two forwards before one backward under gradient
checkpointing (Unsloth raises), so training keeps exactly one forward and reads the teacher
from this cache. The cache is augmentation-invariant by construction: content positions are
state + instruction tokens only, which option shuffles never touch.

Output: `--out.npz` with `topk_ids` [P, k] int32, `topk_logits` [P, k] fp16 and `offsets`
[N+1] int64 (row i's positions are offsets[i]:offsets[i+1], in content-mask order), plus
`--out.json` with the provenance.

    ./unsloth_uv/bin/python data/cache_teacher_logits.py \
        --dataset /mnt/f/distill_jev_runs/prep/dv7-smoke --out /mnt/f/distill_jev_runs/cache/dv7-smoke --top-k 32
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unsloth  # noqa: F401

import numpy as np
import torch
from datasets import load_from_disk

from data.config import DELIMITERS, MODEL
from data.suites import write_json
from model.load import compute_dtype


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=128, help="positions per logits chunk")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--no-resume", action="store_true", help="ignore a partial file and start over")
    return ap.parse_args()


def main():
    a = parse_args()
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name=a.model, max_seq_length=a.max_seq, dtype=compute_dtype(),
        load_in_4bit=True, use_gradient_checkpointing=False, trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.add_special_tokens({"additional_special_tokens": DELIMITERS})
    model.eval()
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    dataset = load_from_disk(a.dataset)
    if a.limit:
        dataset = dataset.select(range(min(a.limit, len(dataset))))
    weight = model.get_output_embeddings().weight
    softcap = getattr(getattr(model.config, "text_config", model.config), "final_logit_softcapping", None)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.npz")
    if partial.exists() and not a.no_resume:
        saved = np.load(partial)
        ids_out = [saved["topk_ids"]]
        logits_out = [saved["topk_logits"]]
        offsets = list(saved["offsets"].astype(int))
        print(f"[cache] resuming: {len(offsets) - 1} rows already cached", flush=True)
    else:
        ids_out, logits_out, offsets = [], [], [0]
    start_row = len(offsets) - 1
    start = time.time()

    def flush():
        np.savez(partial, topk_ids=np.concatenate(ids_out) if ids_out else np.zeros((0, a.top_k), dtype=np.int32),
                 topk_logits=np.concatenate(logits_out) if logits_out else np.zeros((0, a.top_k), dtype=np.float16),
                 offsets=np.asarray(offsets, dtype=np.int64))

    with torch.no_grad():
        for i, row in enumerate(dataset):
            if i < start_row:
                continue
            input_ids = torch.tensor([row["input_ids"][:a.max_seq]], device="cuda")
            # logits_to_keep=1: nothing here reads `out.logits` (the KL gathers its own
            # logits in chunks off hidden_states[-1]), but without it transformers projects
            # every position through lm_head -- 248,320 x seq_len of throwaway tensor.
            hidden = model(input_ids=input_ids, output_hidden_states=True, use_cache=False,
                           logits_to_keep=1).hidden_states[-1][0]
            positions = [t for t, m in enumerate(row["content_mask"][:a.max_seq]) if m]
            for c in range(0, len(positions), a.chunk):
                index = torch.tensor(positions[c:c + a.chunk], device="cuda")
                logits = (hidden[index].to(weight.dtype) @ weight.T).float()
                if softcap:
                    logits = torch.tanh(logits / softcap) * softcap
                values, topk = logits.topk(a.top_k, dim=-1)
                ids_out.append(topk.to(torch.int32).cpu().numpy())
                logits_out.append(values.to(torch.float16).cpu().numpy())
            offsets.append(offsets[-1] + len(positions))
            if (i + 1) % 500 == 0:
                flush()
            if (i + 1) % 200 == 0:
                print(f"[cache] {i + 1}/{len(dataset)} rows, {(time.time() - start) / 60:.1f} min", flush=True)
    topk_ids = np.concatenate(ids_out) if ids_out else np.zeros((0, a.top_k), dtype=np.int32)
    topk_logits = np.concatenate(logits_out) if logits_out else np.zeros((0, a.top_k), dtype=np.float16)
    np.savez(out.with_suffix(".npz"), topk_ids=topk_ids, topk_logits=topk_logits,
             offsets=np.asarray(offsets, dtype=np.int64))
    if partial.exists():
        partial.unlink()
    write_json(out.with_suffix(".json"), {"dataset": a.dataset, "rows": len(dataset), "positions": int(offsets[-1]),
                                          "top_k": a.top_k, "model": a.model, "teacher": "frozen base, no adapters",
                                          "dtype": str(compute_dtype()),
                                          "softcap": softcap, "chunk": a.chunk, "max_seq": a.max_seq,
                                          "seconds": round(time.time() - start, 1)})
    print(f"[cache] {len(dataset)} rows, {offsets[-1]} positions, k={a.top_k} -> {out.with_suffix('.npz')} "
          f"({(time.time() - start) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
