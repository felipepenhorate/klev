#!/usr/bin/env python
"""Build the training rows for distill_jev from a frozen suite (or a labelled JSONL).

Each question becomes one row (encoder: data/encoding.py). Option order is shuffled per
record (deterministic in --seed), and a --garbage_frac share of the Choice questions has its
correct option removed and is relabelled onto the rejection candidate, so the garbage channel
is trained from the start.

Output: a `datasets` directory (`--out`) with columns
  id, record_id, qid, source, qtype, keys, n_options, label, garbage, target,
  input_ids, content_mask, opt_pos, decide_pos
plus `prep_stats.json` and `prep_config.json`.

    ./unsloth_uv/bin/python data/prep_dataset.py \
        --suite evals/v7/decision-v7 --split train --out /mnt/f/distill_jev_runs/prep/dv7-smoke \
        --limit 2000 --max-seq 2048 --garbage-frac 0.1
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from data import config
from data.config import MAX_BRANCH, MAX_SEQ, MAX_STATE
from data.encoding import ContextOverflow, build_target, encode_row
from data.format import materialize
from data.suites import load_split, write_json


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="evals/v7/decision-v7")
    ap.add_argument("--split", default="train")
    ap.add_argument("--data", default="", help="labelled JSONL instead of a suite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many rows (smoke)")
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--max-state", type=int, default=MAX_STATE)
    ap.add_argument("--max-branch", type=int, default=MAX_BRANCH)
    ap.add_argument("--garbage-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-shuffle", action="store_true", help="keep the criteria order as authored")
    ap.add_argument("--model", default=None,
                    help="override the preset's model path (unset = follow --preset)")
    ap.add_argument("--preset", default="",
                    help="base-model preset; sets the model and its delimiters together. One of: "
                         + ", ".join(sorted(config.PRESETS)) + ". Overrides KLEV_PRESET.")
    a = ap.parse_args()
    if not 0 <= a.garbage_frac <= 1:
        ap.error("--garbage-frac must be in [0, 1]")
    return a


def load_records(a):
    if a.data:
        records = []
        with Path(a.data).open(encoding="utf-8") as f:
            for n, line in enumerate(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                text = json.dumps(r["state"], sort_keys=True, ensure_ascii=False) if not isinstance(r["state"], str) else r["state"]
                r.setdefault("_meta", {"source": "custom", "id": f"custom/{n}", "variant": "clean", "group_id": f"custom/{n}",
                                       "row": n, "split": "custom",
                                       "text_sha256": __import__("hashlib").sha256(" ".join(text.casefold().split()).encode()).hexdigest()})
                records.append(r)
        return records
    return load_split(a.suite, a.split)


def augment_request(req, rng, garbage_frac, shuffle):
    """Shuffle the criteria of every Choice question and, for a slice of them, remove the
    labelled option and mark the question as garbage ("none of these fit")."""
    out = {"state": req["state"], "questions": {}}
    for qid, q in req["questions"].items():
        if q["type"] != "choice":
            out["questions"][qid] = q
            continue
        criteria = dict(q["criteria"])
        if shuffle:
            keys = list(criteria)
            rng.shuffle(keys)
            criteria = {k: criteria[k] for k in keys}
        question = {**q, "criteria": criteria}
        if garbage_frac > 0 and len(criteria) >= 3 and rng.random() < garbage_frac:
            question["criteria"] = {k: v for k, v in criteria.items() if k != q["label"]}
            question["garbage"] = 1.0
            question.pop("label", None)
        out["questions"][qid] = question
    return out


def main():
    a = parse_args()
    config.apply_preset(a.preset)
    a.model = a.model or config.MODEL
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    tokenizer.add_special_tokens({"additional_special_tokens": config.DELIMITERS})
    records = load_records(a)
    rows, stats = [], Counter()
    lengths = []
    for rec in records:
        rng = random.Random(f"{a.seed}:{rec['_meta']['id']}")
        internal = materialize(augment_request(rec, rng, a.garbage_frac, not a.no_shuffle))
        for question in internal["questions"]:
            try:
                enc = encode_row(tokenizer, internal, question, max_state=a.max_state, max_branch=a.max_branch)
            except ContextOverflow as e:
                stats["dropped_overflow"] += 1
                continue
            if len(enc["input_ids"]) > a.max_seq:
                stats["dropped_seq"] += 1
                continue
            rows.append({"id": f"{rec['_meta']['id']}/{question['qid']}", "record_id": rec["_meta"]["id"],
                         "qid": question["qid"], "source": question.get("src") or rec["_meta"]["source"],
                         "qtype": question["qtype"], "keys": question["keys"], "n_options": len(question["options"]),
                         "label": question["label"], "garbage": float(question.get("garbage", 0.0) or 0.0),
                         "target": build_target(question, len(question["options"])), **enc})
            lengths.append(len(enc["input_ids"]))
            stats["rows"] += 1
            stats["garbage_rows"] += float(question.get("garbage", 0.0) or 0.0) > 0
            if a.limit and stats["rows"] >= a.limit:
                break
        if a.limit and stats["rows"] >= a.limit:
            break
    if not rows:
        raise SystemExit("no rows built")
    from datasets import Dataset

    out = Path(a.out)
    Dataset.from_list(rows).save_to_disk(str(out))
    lengths.sort()
    percentiles = {f"p{p}": lengths[min(len(lengths) - 1, int(len(lengths) * p / 100))] for p in (50, 90, 99, 100)}
    write_json(out / "prep_config.json", {"args": vars(a), "tokenizer": a.model, "rows": len(rows), "sources": dict(Counter(r["source"] for r in rows)),
                                          "question_types": dict(Counter(r["qtype"] for r in rows)), "stats": dict(stats), "token_percentiles": percentiles})
    print(f"[prep] {len(rows)} rows -> {out}", flush=True)
    print(f"[prep] stats {dict(stats)}", flush=True)
    print(f"[prep] token percentiles {percentiles}", flush=True)


if __name__ == "__main__":
    main()
