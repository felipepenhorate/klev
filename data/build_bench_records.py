#!/usr/bin/env python
"""Standard knowledge benchmarks as TypeSafe decision records, for the fast system_one read.

MMLU / ARC-Challenge / HellaSwag are eval-only here (none is in decision-v7), so pointer-head
accuracy on them measures what the trained head can tap from the backbone after training.

    /home/penhfel/unsloth_uv/bin/python data/build_bench_records.py --task mmlu --limit 1000 \
        --out /mnt/f/distill_jev_runs/bench/mmlu.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from data.suites import write_jsonl

LETTERS = ["a", "b", "c", "d", "e"]


def mmlu(limit, seed):
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed).select(range(limit))
    out = []
    for row in ds:
        keys = LETTERS[: len(row["choices"])]
        out.append({"state": {"subject": row["subject"].replace("_", " "), "question": row["question"]},
                    "questions": {"answer": {"type": "choice", "instructions": "Which option correctly answers the question?",
                                             "criteria": dict(zip(keys, row["choices"])), "label": keys[int(row["answer"])],
                                             "src": "mmlu"}},
                    "_meta": {"source": "mmlu", "id": f"mmlu/{row['subject']}/{len(out)}"}})
    return out


def arc(limit, seed):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").shuffle(seed=seed).select(range(limit))
    out = []
    for row in ds:
        labels, texts = list(row["choices"]["label"]), list(row["choices"]["text"])
        keys = LETTERS[: len(texts)]
        out.append({"state": {"question": row["question"]},
                    "questions": {"answer": {"type": "choice", "instructions": "Which option answers the science question?",
                                             "criteria": dict(zip(keys, texts)), "label": keys[labels.index(row["answerKey"])],
                                             "src": "arc"}},
                    "_meta": {"source": "arc", "id": f"arc/{len(out)}"}})
    return out


def hellaswag(limit, seed):
    ds = load_dataset("Rowan/hellaswag", split="validation").shuffle(seed=seed).select(range(limit))
    out = []
    for row in ds:
        keys = LETTERS[: len(row["endings"])]
        out.append({"state": row["ctx"],
                    "questions": {"continuation": {"type": "choice", "instructions": "Which ending continues the passage most naturally?",
                                                   "criteria": dict(zip(keys, row["endings"])), "label": keys[int(row["label"])],
                                                   "src": "hellaswag"}},
                    "_meta": {"source": "hellaswag", "id": f"hellaswag/{len(out)}"}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["mmlu", "arc", "hellaswag"])
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    records = {"mmlu": mmlu, "arc": arc, "hellaswag": hellaswag}[a.task](a.limit, a.seed)
    write_jsonl(a.out, records)
    print(f"[bench] {len(records)} {a.task} records -> {a.out}")


if __name__ == "__main__":
    main()
