#!/usr/bin/env python
"""Knowledge retention, direct metric (SPEC F2): per-token drift between the trained model and
the frozen base on held-out general instructions.

For each prompt the trained model (adapter on) and the frozen base (adapters disabled) are run
on the same ids; the metric is the mean forward-KL(base ‖ trained) over positions, plus the
share of positions whose argmax token is unchanged. Low KL / high agreement means the
distillation anchor kept the base distribution; a plain SFT run drifts further (DuplexCascade
measured 4.76 vs 1.07 with the same statistic).

    ./unsloth_uv/bin/python eval/eval_drift.py --run /mnt/f/distill_jev_runs/main \
        --n 200 --out /mnt/f/distill_jev_runs/drift-trained
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unsloth  # noqa: F401

import torch
import torch.nn.functional as F
from datasets import load_dataset

from data.config import MODEL
from data.suites import write_json
from eval.eval_chat import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, tokenizer = load_model(a.model, a.run)
    device = next(model.parameters()).device
    weight = model.get_output_embeddings().weight
    softcap = getattr(getattr(model.config, "text_config", model.config), "final_logit_softcapping", None)

    dataset = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=0).select(range(a.n))
    texts = []
    for row in dataset:
        text = row["instruction"] + (("\n" + row["input"]) if row["input"] else "")
        chat = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=False)
        texts.append(tokenizer(chat, add_special_tokens=False)["input_ids"][:a.max_tokens])

    kls, agreements, positions = [], [], 0
    with torch.no_grad():
        for ids in texts:
            input_ids = torch.tensor([ids], device=device)
            trained = model(input_ids=input_ids, output_hidden_states=True, use_cache=False).hidden_states[-1][0]
            with model.disable_adapter():
                base = model(input_ids=input_ids, output_hidden_states=True, use_cache=False).hidden_states[-1][0]
            row_kl, row_agree = 0.0, 0
            for start in range(0, len(ids) - 1, a.chunk):
                index = torch.arange(start, min(start + a.chunk, len(ids) - 1), device=device)
                zt = (trained[index].to(weight.dtype) @ weight.T).float()
                zb = (base[index].to(weight.dtype) @ weight.T).float()
                if softcap:
                    zt = torch.tanh(zt / softcap) * softcap
                    zb = torch.tanh(zb / softcap) * softcap
                p_base = F.softmax(zb, -1)
                row_kl += float(-(p_base * F.log_softmax(zt, -1)).sum(-1).sum())
                row_agree += int((zt.argmax(-1) == zb.argmax(-1)).sum())
                del zt, zb, p_base
            n_pos = len(ids) - 1
            kls.append(row_kl / n_pos)
            agreements.append(row_agree / n_pos)
            positions += n_pos
    report = {"run": a.run, "n_prompts": len(texts), "positions": positions,
              "kl_mean": sum(kls) / len(kls), "kl_median": sorted(kls)[len(kls) // 2],
              "argmax_agreement": sum(agreements) / len(agreements),
              "prompt_max_tokens": a.max_tokens, "dataset": "yahma/alpaca-cleaned (seed 0)"}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "drift.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
