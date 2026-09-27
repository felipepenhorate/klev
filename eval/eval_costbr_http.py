#!/usr/bin/env python
"""Generative CoSt-BR eval against an OpenAI/llama.cpp HTTP server (Winnow-E4B + LoRA).

Sends the exact alpaca prompt the third-party LoRA was trained on (plain completion,
no chat template) and parses the repo's `get_parsed_predictions` rule.

    python eval/eval_costbr_http.py --base-url http://127.0.0.1:8091 --out /mnt/f/distill_jev_runs/winnow-costbr-gen
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from sklearn.metrics import classification_report

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
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--data", default="./github/training-conversational-stance/Datasets/My Dataset/prompt_eval_my_dataset_claim.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-predict", type=int, default=8)
    return ap.parse_args()


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


def complete(base_url, text, n_predict, timeout=180):
    body = {"prompt": text, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
            "cache_prompt": True, "stop": ["\n"]}
    r = requests.post(f"{base_url}/completion", json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return j.get("content", "")


def complete_openai(base_url, text, n_predict, timeout=180):
    body = {"model": "local", "prompt": text, "max_tokens": n_predict, "temperature": 0.0,
            "stop": ["\n"]}
    r = requests.post(f"{base_url}/v1/completions", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["text"]


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = json.load(open(a.data))
    keys = list(data)
    if a.limit:
        keys = keys[:a.limit]
    texts = [ALPACA.format(data[k]["prompt"], "", "") for k in keys]
    labels = [data[k]["label"] for k in keys]

    try:
        complete(a.base_url, "test", 1)
        fn = complete
        endpoint = "/completion"
    except Exception:
        fn = complete_openai
        endpoint = "/v1/completions"
    print(f"[http] using {endpoint}", flush=True)

    raw = [""] * len(texts)
    failures = 0
    start = time.time()

    def one(i):
        return i, fn(a.base_url, texts[i], a.n_predict)

    with ThreadPoolExecutor(a.concurrency) as pool:
        for done, (i, text) in enumerate(pool.map(one, range(len(texts))), start=1):
            raw[i] = text
            if done % 100 == 0:
                print(f"[http] {done}/{len(texts)} ({(time.time() - start):.0f}s)", flush=True)

    parsed = [parse_pred(x) for x in raw]
    rep = classification_report(labels, parsed, output_dict=True, zero_division=0)
    report = {"endpoint": endpoint, "n": len(texts), "failures": failures,
              "acc": rep["accuracy"], "macro_f1": rep["macro avg"]["f1-score"],
              "macro_recall": rep["macro avg"]["recall"], "macro_precision": rep["macro avg"]["precision"],
              "per_class_f1": {c: rep[c]["f1-score"] for c in LABELS},
              "seconds": round(time.time() - start, 1)}
    preds = {k: {**data[k], "prediction": raw[i], "parsed": parsed[i]} for i, k in enumerate(keys)}
    (out / "predictions.json").write_text(json.dumps(preds))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
