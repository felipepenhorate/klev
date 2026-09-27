#!/usr/bin/env python
"""Chat-mode knowledge retention: standard multiple-choice benchmarks scored by
loglikelihood under the model's own chat template.

lm-eval 0.4.12 cannot load Gemma 4 (no AutoModelForCausalLM mapping for `gemma4`), so this
implements the same scoring contract for the tasks that matter for retention:
  hellaswag      acc / acc_norm (character-normalized ending logprob)
  arc_challenge  acc / acc_norm
  mmlu           acc (letter continuation)
  truthfulqa_mc2 normalized probability mass on the true answers

Run the same command with and without `--run` (adapter) and compare:

    /home/penhfel/unsloth_uv/bin/python eval/eval_chat.py --tasks hellaswag,arc_challenge,mmlu \
        --limit 1000 --out /mnt/f/distill_jev_runs/chat-base
    /home/penhfel/unsloth_uv/bin/python eval/eval_chat.py --run /mnt/f/distill_jev_runs/main \
        --tasks hellaswag,arc_challenge,mmlu --limit 1000 --out /mnt/f/distill_jev_runs/chat-trained
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
import torch.nn.functional as F
from datasets import load_dataset

from data.config import MODEL
from data.suites import write_json

MMLU_LETTERS = ["A", "B", "C", "D"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="checkpoint dir with adapter/ (omit for the base model)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tasks", default="hellaswag,arc_challenge,mmlu")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def load_model(base, run=""):
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name=base, max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=True,
        use_gradient_checkpointing=False, trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    if run:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(Path(run) / "adapter"))
    model.eval()
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    return model, tokenizer


def chat_ids(tokenizer, prompt: str) -> list[int]:
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def choice_logprobs(model, tokenizer, prompt: str, continuations: list[str], device) -> list[float]:
    """Sum of continuation logprobs for every choice, under the chat template."""
    prompt_ids = chat_ids(tokenizer, prompt)
    scores = []
    for continuation in continuations:
        cont_ids = tokenizer(" " + continuation, add_special_tokens=False)["input_ids"]
        ids = prompt_ids + cont_ids
        keep = torch.arange(len(prompt_ids) - 1, len(ids) - 1, device=device)
        logits = model(input_ids=torch.tensor([ids], device=device), logits_to_keep=keep).logits[0].float()
        logprobs = F.log_softmax(logits, -1)
        targets = torch.tensor(cont_ids, device=device)
        scores.append(float(logprobs[torch.arange(len(cont_ids), device=device), targets].sum()))
    return scores


def build_task(name, limit, seed):
    if name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
        return [{"prompt": row["ctx"], "choices": row["endings"], "label": int(row["label"]), "norm": "char", "metric": "acc_norm"} for row in ds]
    if name == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
        out = []
        for row in ds:
            labels = list(row["choices"]["label"])
            texts = list(row["choices"]["text"])
            out.append({"prompt": f"Question: {row['question']}\nAnswer:", "choices": texts,
                        "label": labels.index(row["answerKey"]), "norm": "char", "metric": "acc_norm"})
        return out
    if name == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
        out = []
        for row in ds:
            choices = "\n".join(f"{letter}. {text}" for letter, text in zip(MMLU_LETTERS, row["choices"]))
            out.append({"prompt": f"Question: {row['question']}\n{choices}\nAnswer:", "choices": MMLU_LETTERS,
                        "label": int(row["answer"]), "norm": None, "metric": "acc"})
        return out
    if name == "truthfulqa_mc2":
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
        out = []
        for row in ds:
            targets = row["mc1_targets"]
            out.append({"prompt": f"Question: {row['question']}\nAnswer:", "choices": targets["choices"],
                        "label": targets["labels"].index(1), "labels": targets["labels"], "norm": None, "metric": "mc2"})
        return out
    raise ValueError(f"unknown task {name}")


def score(examples, model, tokenizer, device):
    correct, correct_norm, mc2_hits, mc2_probs = 0, 0, 0, []
    for i, example in enumerate(examples):
        scores = choice_logprobs(model, tokenizer, example["prompt"], example["choices"], device)
        best = max(range(len(scores)), key=lambda j: scores[j])
        correct += best == example["label"]
        if example["norm"] == "char":
            lengths = [max(len(c), 1) for c in example["choices"]]
            best_norm = max(range(len(scores)), key=lambda j: scores[j] / lengths[j])
            correct_norm += best_norm == example["label"]
        if example["metric"] == "mc2":
            probs = torch.softmax(torch.tensor(scores), -1).tolist()
            true_mass = sum(p for p, l in zip(probs, example["labels"]) if l == 1)
            mc2_hits += true_mass > 0.5
            mc2_probs.append(true_mass)
        if (i + 1) % 200 == 0:
            print(f"[chat] {i + 1}/{len(examples)}", flush=True)
    n = len(examples)
    result = {"n": n, "acc": correct / n}
    if examples[0]["norm"] == "char":
        result["acc_norm"] = correct_norm / n
    if examples[0]["metric"] == "mc2":
        result["mc2"] = sum(mc2_probs) / n
        result["mc2_at_0_5"] = mc2_hits / n
    return result


def main():
    a = parse_args()
    model, tokenizer = load_model(a.model, a.run)
    device = next(model.parameters()).device
    results = {"model": a.run or a.model, "limit": a.limit, "tasks": {}}
    for name in [t for t in a.tasks.split(",") if t]:
        examples = build_task(name, a.limit, a.seed)
        start = time.time()
        print(f"[chat] {name}: {len(examples)} examples", flush=True)
        results["tasks"][name] = {**score(examples, model, tokenizer, device), "seconds": round(time.time() - start, 1)}
        print(f"[chat] {name}: {json.dumps(results['tasks'][name])}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "chat_results.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
