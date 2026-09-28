#!/usr/bin/env python
"""Build TweetEval hate-speech detection as klev labelled records + the stitch's alpaca prompts.

Source: cardiffnlp/tweet_eval, config `hate` -- ~7,000 short tweets, binary (not_hate / hate).

Chosen for the few-shot stitch because it maximises the gap the stitch has to close:
decision-v7 has no hate-speech training, so klev's and kev's pointer heads should sit near
chance (0.5) on it, while a LoRA over a 0.8B model reaches ~0.8. Everything is short -- measured
p50 58 / p90 87 / p100 119 alpaca tokens -- which matters because this card has a hard sequence
ceiling: past ~195 tokens Qwen3.5's deltanet attention autotunes into a bf16 candidate gfx1030
cannot compile. See scripts/rocm_env.sh and data/build_pubmedqa.py:abstracts_text.

Three disjoint parts, split deterministically so nothing is evaluated on what it trained:
  train.jsonl / alpaca_train.json     the 32-shot balanced pool (16/class) for the fusion
  validation.jsonl / alpaca_val...    the test set for the stitch
  train_extlora.jsonl / alpaca_...    everything left, for training the external adapter

Note on content: this is a standard public benchmark for hate-speech detection and does contain
slurs, as such benchmarks do. The labels are the dataset's own; nothing here generates new text.
"""
import argparse
import json
import random
from pathlib import Path

# Keyed and ordered so that the option index equals the dataset's raw label
# (0 = not_hate, 1 = hate). With --no-shuffle that makes `label` a stable class identity, which
# the stitch's NCM probe depends on -- see eval_lora_stitch.py:check_label_is_class.
CRITERIA = {
    "not_hate": "not_hate - the text does not attack or dehumanise a group or individual",
    "hate": "hate - the text attacks or dehumanises a group or individual",
}
INSTRUCTION = "Does the text contain hate speech?"
LABEL_NAME = {0: "not_hate", 1: "hate"}

ALPACA_PROMPT = (
    "Classify whether a tweet contains hate speech, meaning it attacks or dehumanises a "
    "group or an individual.\n"
    "Answer with exactly one of: not_hate, hate.\n\n"
    "Text: \"{text}\"\n\nLabel:"
)


def row_record(split, i, r):
    text = str(r["text"]).strip()
    if not text or int(r["label"]) not in LABEL_NAME:
        return None, None
    label = LABEL_NAME[int(r["label"])]
    rec = {
        "state": {"text": text},
        "questions": {"hate": {
            "type": "choice", "instructions": INSTRUCTION,
            "criteria": dict(CRITERIA), "label": label}},
        "_meta": {"id": f"tweet_eval/{split}/{i:06d}", "source": "tweet_eval_hate",
                  "group_id": f"tweet_eval/{split}/{i:06d}", "row": i, "split": split,
                  "repo": "cardiffnlp/tweet_eval:hate", "variant": "clean"},
    }
    return rec, ALPACA_PROMPT.format(text=text)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class-shot", type=int, default=16, help="balanced rows per class for the fusion")
    ap.add_argument("--test-cap", type=int, default=200)
    ap.add_argument("--max-text-chars", type=int, default=600,
                    help="tweets are short; this only guards against outliers")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    a = parse_args()
    from datasets import load_dataset

    raw = load_dataset("cardiffnlp/tweet_eval", "hate", split="train")
    recs, prompts = [], {}
    for i, r in enumerate(raw):
        rec, prompt = row_record("train", i, r)
        if rec is None:
            continue
        key = rec["_meta"]["id"].split("/")[-1]
        prompts[key] = {"prompt": prompt, "answer": rec["questions"]["hate"]["label"]}
        recs.append(rec)

    by_class = {}
    for rec in recs:
        by_class.setdefault(rec["questions"]["hate"]["label"], []).append(rec)
    rng = random.Random(a.seed)
    for items in by_class.values():
        rng.shuffle(items)
    shot, rest = [], []
    for cls, items in sorted(by_class.items()):
        shot.extend(items[: a.per_class_shot])
        rest.extend(items[a.per_class_shot:])
    # Order matters: shuffle, cap the test set, and only then carve the adapter's training set
    # out of what is left. Taking `test` as "everything not in shot" and capping afterwards
    # leaves nothing for the adapter, and capping before shuffling truncates on the first class
    # and silently drops the rest.
    rng.shuffle(shot); rng.shuffle(rest)
    test = rest[: a.test_cap] if a.test_cap else rest
    used = {id(r) for r in shot} | {id(r) for r in test}
    train = [r for r in rest if id(r) not in used]

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    def dump(name, rows, prompt_name):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (out / prompt_name).open("w", encoding="utf-8") as f:
            json.dump({r["_meta"]["id"].split("/")[-1]: prompts[r["_meta"]["id"].split("/")[-1]] for r in rows},
                      f, ensure_ascii=False, indent=2)
        dist = {}
        for r in rows:
            k = r["questions"]["hate"]["label"]; dist[k] = dist.get(k, 0) + 1
        print(f"[tweet] {name:12s} {len(rows):5d} rows  classes {dict(sorted(dist.items()))}")

    dump("train", shot, "alpaca_train.json")
    dump("validation", test, "alpaca_validation.json")
    dump("train_extlora", train, "alpaca_train_extlora.json")
    print(f"[tweet] class -> option index (prep with --no-shuffle): "
          f"{ {k: i for i, k in enumerate(CRITERIA)} }")


if __name__ == "__main__":
    main()
