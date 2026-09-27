#!/usr/bin/env python
"""Generative CoSt-BR ("My Dataset") eval in the training-conversational-stance format.

Loads the base model, optionally M5's adapter (`--run`), optionally a third-party LoRA
(`--ext`) trained on the base model, and scores conditions: base, M5, ext, and M5+ext at
several weights. Generation uses the external LoRA's own alpaca prompt (it was trained in
plain completion format, not the chat template), greedy, max 3 new tokens; parsing is the
`get_parsed_predictions` rule of `Tests/My Dataset/Compare Results - My Dataset.ipynb`.

    ./unsloth_uv/bin/python eval/eval_costbr_gen.py \
        --run /mnt/f/distill_jev_runs/main --ext "./github/Trained Models/My Dataset/gemma-4-e4b_claim" \
        --weights 1.0,0.5 --out /mnt/f/distill_jev_runs/costbr-compose
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unsloth  # noqa: F401

import torch
from sklearn.metrics import classification_report

from data.config import MODEL
from data.suites import write_json

ALPACA = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

LABELS = ["Discussing", "Refuting", "Querying", "Supporting", "Irrelevant"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--run", default="", help="M5 checkpoint dir with adapter/ (omit for base)")
    ap.add_argument("--ext", default="", help="third-party LoRA dir trained on the base model")
    ap.add_argument("--weights", default="1.0", help="comma-separated ext weights (M5 weight is 1)")
    ap.add_argument("--conditions", default="", help="explicit conditions, e.g. base,default,ext,combo1.0")
    ap.add_argument("--data", default="./github/training-conversational-stance/Datasets/My Dataset/prompt_eval_my_dataset_claim.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def load_model(base, run, ext):
    from peft import PeftModel
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name=base, max_seq_length=4096, dtype=torch.bfloat16, load_in_4bit=True,
        use_gradient_checkpointing=False, trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "left"
    if run:
        model = PeftModel.from_pretrained(model, str(Path(run) / "adapter"))
    if ext:
        model.load_adapter(ext, adapter_name="ext", is_trainable=False)
    FastModel.for_inference(model)
    model.eval()
    model.config.use_cache = True
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True
    return model, tokenizer


def parse_pred(ans):
    ans = ans.lower()
    if "discussing" in ans:
        return "Discussing"
    if "deny" in ans or "denying" in ans or "denies" in ans or "refuting" in ans:
        return "Refuting"
    if "back up" in ans or "reinforce" in ans or "support" in ans or "supporting" in ans:
        return "Supporting"
    if "query" in ans or "querying" in ans or "queries" in ans:
        return "Querying"
    return "Irrelevant"


@torch.no_grad()
def generate_all(model, tokenizer, prompts, batch, max_new_tokens=3):
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        ids = tokenizer([ALPACA.format(p, "", "") for p in chunk], return_tensors="pt",
                        padding=True, add_special_tokens=True).to("cuda")
        gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        for row in range(len(chunk)):
            outs.append(tokenizer.decode(gen[row][ids["input_ids"].shape[1]:], skip_special_tokens=True))
        if (i // batch + 1) % 10 == 0:
            print(f"[gen] {i + len(chunk)}/{len(prompts)}", flush=True)
    return outs


def set_condition(model, condition, weights):
    from peft import PeftModel

    if condition == "base":
        if isinstance(model, PeftModel):
            model.disable_adapter_layers()
        return
    if condition == "ext":
        model.enable_adapter_layers()
        model.set_adapter("ext")
        return
    if condition == "default":
        model.enable_adapter_layers()
        model.set_adapter("default")
        return
    if condition.startswith("combo"):
        model.enable_adapter_layers()
        model.set_adapter(condition.replace(".", "_"))
        return
    raise ValueError(condition)


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = json.load(open(a.data))
    keys = list(data)
    if a.limit:
        keys = keys[:a.limit]
    prompts = [data[k]["prompt"] for k in keys]
    labels = [data[k]["label"] for k in keys]

    model, tokenizer = load_model(a.model, a.run, a.ext)

    conditions = [c for c in a.conditions.split(",") if c]
    if not conditions:
        conditions = ["base"]
        if a.run:
            conditions.append("default")
        if a.ext:
            conditions.append("ext")
            conditions += [f"combo{w}" for w in a.weights.split(",")]
    report = {"data": a.data, "n": len(keys), "conditions": {}}
    for condition in conditions:
        if condition.startswith("combo") and a.ext:
            adapter_name = condition.replace(".", "_")
            if not model.peft_config.get(adapter_name):
                weight = float(condition.replace("combo", ""))
                model.add_weighted_adapter(["default", "ext"], [1.0, weight], adapter_name,
                                           combination_type="linear")
        set_condition(model, condition, a.weights)
        start = time.time()
        raw = generate_all(model, tokenizer, prompts, a.batch)
        parsed = [parse_pred(x) for x in raw]
        rep = classification_report(labels, parsed, output_dict=True, zero_division=0)
        report["conditions"][condition] = {
            "acc": rep["accuracy"],
            "macro_f1": rep["macro avg"]["f1-score"],
            "macro_recall": rep["macro avg"]["recall"],
            "macro_precision": rep["macro avg"]["precision"],
            "per_class_f1": {c: rep[c]["f1-score"] for c in LABELS},
            "seconds": round(time.time() - start, 1),
        }
        preds = {}
        for i, k in enumerate(keys):
            preds[k] = {**data[k], "prediction": raw[i], "parsed": parsed[i]}
        write_json(out / f"predictions-{condition}.json", preds)
        print(f"[gen] {condition}: {json.dumps(report['conditions'][condition])}", flush=True)
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
