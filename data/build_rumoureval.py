#!/usr/bin/env python
"""Build RumourEval 2019 as klev labelled records + the alpaca prompts the stitch probes.

Source: strombergnlp/rumoureval2019 (train/val/test CSVs; columns id, source_text, reply_text,
label). The task is the 2019 responding-relationship one: given a source tweet carrying a claim
and a reply, classify the reply as support / deny / comment / query.

Two artefacts per split, because the stitch needs the row in both of klev's worlds:
  <split>.jsonl  a klev labelled record (`prep_dataset.py --data` takes it directly)
  alpaca_<split>.json  conv_key -> {"prompt": ...}, the *entire* task in the alpaca Instruction
                  slot. `eval_lora_stitch.py` formats these with the ALPACA template and reads
                  the last-token hidden state, so everything the probe needs must be in here.

The four classes are heavily imbalanced (comment 3495 vs deny 367 in train), which a
few-shot nearest-class-mean probe cannot survive, so the stitch pool is balanced by sampling.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

# Option keys are the class names; the text is what the pointer readout actually reads, so it
# spells out the criterion rather than naming it.
CRITERIA = {
    "support": "the reply agrees with and supports the claim made in the source",
    "deny": "the reply disagrees with and denies the claim made in the source",
    "comment": "the reply comments on the claim without taking a side either way",
    "query": "the reply asks a question instead of making a claim of its own",
}
INSTRUCTION = ("How does the reply respond to the claim in the source? "
               "Answer with one of: support, deny, comment, query.")

ALPACA_PROMPT = (
    'Classify the responding relationship between a source claim and a reply.\n'
    'Answer with exactly one of: support, deny, comment, query.\n\n'
    'Source: "{source}"\nReply: "{reply}"\n\nRelationship:'
)
ALPACA_ANSWER = {
    "support": "support", "deny": "deny", "comment": "comment", "query": "query",
}

FILES = {"train": "rumoureval2019_train.csv", "validation": "rumoureval2019_val.csv",
         "test": "rumoureval2019_test.csv"}


def row_record(split, i, r):
    """One RumourEval row -> a klev labelled record, plus its alpaca prompt text."""
    state = {"source_tweet": str(r["source_text"]), "reply": str(r["reply_text"])}
    label = str(r["label"]).strip().lower()
    if label not in CRITERIA:
        return None, None
    rec = {
        "state": state,
        "questions": {"relationship": {
            "type": "choice", "instructions": INSTRUCTION,
            "criteria": dict(CRITERIA), "label": label}},
        "_meta": {"id": f"rumoureval2019/{split}/{i:06d}", "source": "rumoureval2019",
                  "group_id": f"rumoureval2019/{split}/{i:06d}", "row": i, "split": split,
                  "repo": "strombergnlp/rumoureval_2019", "variant": "clean"},
    }
    prompt = ALPACA_PROMPT.format(source=str(r["source_text"]), reply=str(r["reply_text"]))
    return rec, prompt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory holding the three RumourEval CSVs")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--per-class-train", type=int, default=0,
                    help="balanced rows per class for the stitch's few-shot pool (0 = all)")
    ap.add_argument("--per-class-test", type=int, default=0, help="balanced rows per class for eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="train,validation,test")
    return ap.parse_args()


def main():
    a = parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import random

    for split in a.splits.split(","):
        df = pd.read_csv(src / FILES[split])
        recs, prompts = [], {}
        for i, r in df.iterrows():
            rec, prompt = row_record(split, i, r)
            if rec is None:
                continue
            prompts[rec["_meta"]["id"].split("/")[-1]] = {"prompt": prompt,
                                                           "answer": ALPACA_ANSWER[rec["questions"]["relationship"]["label"]]}
            recs.append(rec)
        by_class = {}
        for rec in recs:
            by_class.setdefault(rec["questions"]["relationship"]["label"], []).append(rec)
        cap = a.per_class_train if split == "train" else a.per_class_test
        if cap:
            rng = random.Random(f"{a.seed}:{split}")
            picked = []
            for cls, items in sorted(by_class.items()):
                rng.shuffle(items)
                picked.extend(items[:cap])
            rng.shuffle(picked)
            recs = picked
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with (out / f"alpaca_{split}.json").open("w", encoding="utf-8") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        counts = {c: len(v) for c, v in sorted(by_class.items())}
        kept = {}
        for rec in recs:
            kept[rec["questions"]["relationship"]["label"]] = kept.get(
                rec["questions"]["relationship"]["label"], 0) + 1
        print(f"[rumour] {split}: {len(recs)} records -> {out}/{split}.jsonl")
        print(f"[rumour]   class counts available {counts}")
        print(f"[rumour]   class counts written  {dict(sorted(kept.items()))}")


if __name__ == "__main__":
    main()
