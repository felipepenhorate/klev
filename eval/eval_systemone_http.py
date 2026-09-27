#!/usr/bin/env python
"""Score a System One-compatible HTTP endpoint (e.g. the Winnow server) on our bench records.

    /home/penhfel/unsloth_uv/bin/python eval/eval_systemone_http.py \
        --base-url http://127.0.0.1:8091 --data /mnt/f/distill_jev_runs/bench/mmlu.jsonl \
        --out /mnt/f/distill_jev_runs/winnow-mmlu --concurrency 4

Each record is a labelled request (state + questions with criteria and label). The endpoint's
choice probabilities are re-normalised over the criteria keys and scored with data/metrics.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests

from data import metrics
from data.suites import read_jsonl, write_json


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--data", required=True, help="labelled requests JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", default="")
    return ap.parse_args()


def score_one(base_url, record, timeout):
    body = {"state": record["state"],
            "questions": {qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
                          for qid, q in record["questions"].items()}}
    response = requests.post(f"{base_url}/v1/systemone", json=body, timeout=timeout)
    response.raise_for_status()
    return response.json().get("answers", {})


def main():
    a = parse_args()
    records = read_jsonl(a.data)
    if a.limit:
        records = records[:a.limit]
    rows, failures, start = [None] * len(records), 0, time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures = {pool.submit(score_one, a.base_url, r, a.timeout): i for i, r in enumerate(records)}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            record = records[i]
            try:
                answers = future.result()
            except Exception:
                failures += 1
                continue
            for qid, question in record["questions"].items():
                answer = answers.get(qid) or {}
                if question["type"] == "noul":
                    keys = ["false", "true"]
                elif question["type"] == "score":
                    keys = [str(k) for k in range(len(question["criteria"]))]
                else:
                    keys = list(question["criteria"])
                probabilities = answer.get("probabilities") or {}
                if not probabilities:
                    failures += 1
                    continue
                p = [float(probabilities.get(k, 0.0)) for k in keys]
                total = sum(p) or 1.0
                p = [x / total for x in p]
                if question["type"] == "score":
                    label = int(question["label"])
                elif question["type"] == "noul":
                    label = int(bool(question["label"]))
                else:
                    label = keys.index(question["label"])
                rows[i] = {"id": record["_meta"]["id"], "question": qid, "group": record["_meta"]["id"],
                           "task": question["type"], "source": record["_meta"].get("source", "custom"),
                           "type": question["type"], "keys": keys, "variant": "clean", "label": label,
                           "p": p, "inference_temperature": 1.0}
            done += 1
            if done % 250 == 0:
                rate = done / (time.time() - start)
                print(f"[http] {done}/{len(records)} ({rate:.1f}/s, {failures} failures)", flush=True)
    scored = [r for r in rows if r]
    if not scored:
        raise SystemExit(f"no scored rows ({failures} failures)")
    report = {"endpoint": a.base_url, "data": a.data, "n": len(scored), "failures": failures,
              "metrics": metrics.metrics(scored), "by_source": metrics.grouped_metrics(scored, "source"),
              "by_type": metrics.grouped_metrics(scored, "type")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "rows.json", scored)
    write_json(out / "report.json", report)
    print(json.dumps({"n": len(scored), "failures": failures, **{k: round(v, 4) for k, v in report["metrics"].items() if isinstance(v, float)}}, indent=1))
    for source, m in report["by_source"].items():
        print(f"[http] {source:28s} n={m['n']:5d} acc={m['acc']:.3f} brier={m['brier']:.3f} ece={m['ece']:.3f}")


if __name__ == "__main__":
    main()
