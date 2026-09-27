#!/usr/bin/env python
"""Combine synthetic decision records with a replay sample from decision-v7 for a delta stage.

    /home/penhfel/unsloth_uv/bin/python data/build_delta_data.py \
        --synth /mnt/f/distill_jev_runs/synth/qwen35_v1.jsonl --replay 4000 \
        --out /mnt/f/distill_jev_runs/synth/delta-v1.jsonl
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.suites import read_jsonl, write_jsonl

TRAIN = ROOT / "evals/v7/decision-v7/train.jsonl"


def source_of(record):
    meta = record.get("_meta") or {}
    return meta.get("source", "unknown")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True, help="comma-separated synthetic JSONL files")
    ap.add_argument("--replay", type=int, default=4000, help="decision-v7 train records to replay")
    ap.add_argument("--train-file", default=str(TRAIN))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    synth = []
    for path in a.synth.split(","):
        if path:
            synth += read_jsonl(path)
    replay = read_jsonl(a.train_file) if a.replay else []
    rng = random.Random(a.seed)
    if a.replay and len(replay) > a.replay:
        replay = rng.sample(replay, a.replay)
    records = synth + replay
    rng.shuffle(records)
    write_jsonl(a.out, records)
    print(f"[delta] {len(records)} records -> {a.out}", flush=True)
    print(f"[delta] synth {dict(Counter(source_of(r) for r in synth))}", flush=True)
    print(f"[delta] replay {dict(Counter(source_of(r) for r in replay))}", flush=True)


if __name__ == "__main__":
    main()
