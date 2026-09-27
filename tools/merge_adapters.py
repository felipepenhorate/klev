#!/usr/bin/env python
"""Can our decision adapter be merged with another Gemma 4 E4B LoRA (Fable distill) and still
work with the pointer head?

Loads our checkpoint (base + delimiter deltas + our adapter + head.pt), loads a second adapter
from the Hub, and evaluates several adapter combinations with the same head:

  ours     our adapter only (baseline)
  fable    the other adapter only, with our head
  sum      both adapters active (peft sums them; effective scales 2.0 : 1.0)
  svd_w_w  add_weighted_adapter(..., combination_type="svd") with the given weights

Metrics: decision-v7 development (system_one), MMLU/ARC pointer benchmarks, and a chat-mode
slice (HellaSwag/MMLU) so capability changes are visible.

    /home/penhfel/unsloth_uv/bin/python tools/merge_adapters.py \
        --run /mnt/f/distill_jev_runs/main --other armand0e/Gemma-4-E4B-it-Fable-Distill-LoRA \
        --out /mnt/f/distill_jev_runs/merge-fable
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
from datasets import load_from_disk

from data import metrics
from data.suites import write_json
from eval.eval_chat import build_task, choice_logprobs, score
from eval.eval_decisions import load_checkpoint

DEV = "/mnt/f/distill_jev_runs/prep/dv7-dev"
# the modules both adapters cover (Fable's regex also lists per-layer gates, which the merge drops)
BASE_TARGETS = "q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
BENCH = {"mmlu": "/mnt/f/distill_jev_runs/prep/bench-mmlu", "arc": "/mnt/f/distill_jev_runs/prep/bench-arc"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default="ours,fable,sum,svd_0.5_0.5,svd_0.7_0.3")
    ap.add_argument("--chat-tasks", default="", help="system_one only by default; set to e.g. hellaswag,mmlu to also score chat")
    ap.add_argument("--svd-rank", type=int, default=48)
    ap.add_argument("--decision-limit", type=int, default=0)
    ap.add_argument("--benches", default="mmlu", help="pointer benchmark datasets to run (mmlu, arc)")
    ap.add_argument("--force", action="store_true", help="re-run configs already present in merge_results.json")
    return ap.parse_args()


@torch.no_grad()
def eval_decision(model, head, dataset, device, limit=0):
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    rows = []
    start = time.time()
    for i, row in enumerate(dataset):
        if (i + 1) % 250 == 0:
            print(f"[merge]   {i + 1}/{len(dataset)} rows, {(time.time() - start) / 60:.1f} min", flush=True)
        ids = torch.tensor([row["input_ids"]], device=device)
        hidden = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[-1][0]
        option_hidden = hidden[torch.tensor(row["opt_pos"], device=device)]
        logits = model.head(hidden[row["decide_pos"]].float(), option_hidden.float())
        probs = torch.softmax(logits.float(), -1)
        n = row["n_options"]
        p = probs[:n] / probs[:n].sum().clamp_min(1e-9)
        rows.append({"id": row["id"], "question": row["qid"], "group": row["record_id"], "task": row["qtype"],
                     "source": row["source"], "type": row["qtype"], "keys": row["keys"], "variant": "clean",
                     "label": int(row["label"]), "p": p.tolist(), "logits": logits[:n].tolist(),
                     "inference_temperature": 1.0})
    result = metrics.metrics(rows)
    return {k: round(v, 4) for k, v in result.items() if isinstance(v, (int, float)) and k in
            ("n", "acc", "brier", "ece", "nll", "coverage_at_5pct_error", "aurc")}


def set_config(model, name, args):
    if name == "ours":
        model.set_adapter("default")
    elif name == "fable":
        model.set_adapter("fable")
    elif name == "sum":
        model.base_model.set_adapter(["default", "fable"])
    elif name.startswith("svd_"):
        weights = [float(w) for w in name.removeprefix("svd_").split("_")]
        merged = f"merged_{'_'.join(str(w) for w in weights)}"
        if merged not in getattr(model, "peft_config", {}):
            for adapter in ("default", "fable"):   # add_weighted_adapter needs the same target_modules type
                model.peft_config[adapter].target_modules = BASE_TARGETS
            model.add_weighted_adapter(["default", "fable"], weights, merged,
                                       combination_type="svd", svd_rank=args.svd_rank)
        model.set_adapter(merged)
    else:
        raise ValueError(name)
    return name


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer, head = load_checkpoint(Path(a.run), "unsloth/gemma-4-e4b-it-unsloth-bnb-4bit")
    device = next(model.parameters()).device
    print(f"[merge] loading second adapter {a.other}", flush=True)
    model.load_adapter(a.other, adapter_name="fable")
    print(f"[merge] adapters: {list(model.peft_config)}", flush=True)

    dev = load_from_disk(DEV)
    benches = {name: load_from_disk(BENCH[name]) for name in a.benches.split(",") if name}
    chat_examples = {name: build_task(name, a.chat_limit, 0) for name in a.chat_tasks.split(",") if name} if a.chat_tasks else {}
    results = {}
    previous = out / "merge_results.json"
    if previous.exists() and not a.force:
        results = json.loads(previous.read_text(encoding="utf-8"))
        print(f"[merge] keeping {list(results)} from {previous}", flush=True)
    for config in a.configs.split(","):
        if config in results and "error" not in results[config]:
            print(f"[merge] {config} already done, skipping", flush=True)
            continue
        try:
            set_config(model, config, a)
            model.eval()
            start = time.time()
            entry = {"adapters": list(model.active_adapters) if hasattr(model, "active_adapters") else config}
            entry["decision_dev"] = eval_decision(model, head, dev, device, a.decision_limit)
            for name, dataset in benches.items():
                entry[f"pointer_{name}"] = eval_decision(model, head, dataset, device)
            for name, examples in chat_examples.items():
                entry[f"chat_{name}"] = score(examples, model, tokenizer, device)
            entry["seconds"] = round(time.time() - start, 1)
            results[config] = entry
            write_json(out / "merge_results.json", results)
            print(f"[merge] {config}: {json.dumps(entry)}", flush=True)
        except Exception as e:
            results[config] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
            print(f"[merge] {config} FAILED: {results[config]['error']}", flush=True)
    write_json(out / "merge_results.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
