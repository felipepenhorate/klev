#!/usr/bin/env python
"""Build PubMedQA (pqa_labeled) as klev labelled records + the alpaca prompts the stitch probes.

Source: qiaojin/PubMedQA, config `pqa_labeled` -- 1,000 expert-labelled biomedical yes/no/maybe
questions, each with 2-3 abstract snippets as context. Small, MIT, and far outside decision-v7's
domains (banking, news, reviews, NLI, intent), which is what the stitch wants: a decision model
that has learned none of it, and an adapter that can.

Same two artefacts per split as data/build_rumoureval.py, because the stitch needs the row in both
of klev's worlds: a klev `choice` record, and an alpaca prompt with the whole task in the
Instruction slot (that is the string whose last-token hidden state the probe reads).

pqa_labeled is skewed (yes 552 / no 338 / maybe 110), so the few-shot pool is sampled per class.
The test set keeps the natural distribution and is additionally reported macro-averaged, since on
a 55/34/11 split raw accuracy is mostly the majority rate.
"""
import argparse
import json
import random
from pathlib import Path

CRITERIA = {
    "yes": "yes - the abstracts support the answer to the question",
    "no": "no - the abstracts contradict the answer to the question",
    "maybe": "maybe - the abstracts do not settle the question either way",
}
INSTRUCTION = "Do the abstracts answer the question yes, no, or maybe?"

ALPACA_PROMPT = (
    "Read the abstracts and decide whether they answer the question with yes, no, or maybe.\n"
    "Answer with exactly one of: yes, no, maybe.\n\n"
    "Question: {question}\n"
    "Abstracts:\n{abstracts}\n\nAnswer:"
)


def abstracts_text(contexts, max_words=30, max_abstracts=1):
    """Truncated evidence, so both views of the row stay under this card's usable sequence length.

    The length is a measured limit of this card, not a guess. Qwen3.5's gated-deltanet
    attention autotunes over block sizes, and on gfx1030 (no bf16 fdot2) a fresh autotune at
    the larger block sizes aborts with
      LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16
    Measured on the alpaca prompt lengths actually produced:

      2 abstracts x 45 words   p50 221  p90 253  p100 304   aborts
      2 abstracts x 30 words   p50 190  p90 214  p100 238   aborts
      1 abstract  x 45 words   p50 159  p90 177  p100 212   aborts
      1 abstract  x 30 words   p50 144  p90 157  p100 177   trains
      1 abstract  x 20 words   p50 132  p90 145  p100 157   trains

    The RumourEval adapter, which trains on this card, tops out at p100 195. So the default is
    the most generous setting that fits under that. This is a genuine cost: 30 words of one
    abstract is thin evidence for a yes/no/maybe judgement, so expect the adapter to be weaker
    here than on full abstracts. It is the same cap for the klev row and the alpaca prompt --
    the stitch compares the decide space against the alpaca space, so the two must be
    truncated identically.
    """
    kept = [c.strip() for c in contexts[:max_abstracts]]
    out = []
    for c in kept:
        words = c.split()
        out.append("- " + (" ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")))
    return "\n".join(out)


def row_record(split, i, r, max_words=30, max_abstracts=1):
    contexts = r["context"]["contexts"]
    label = str(r["final_decision"]).strip().lower()
    if label not in CRITERIA or not contexts:
        return None, None
    state = {"question": str(r["question"]),
             "abstracts": abstracts_text(contexts, max_words, max_abstracts)}
    rec = {
        "state": state,
        "questions": {"answer": {
            "type": "choice", "instructions": INSTRUCTION,
            "criteria": dict(CRITERIA), "label": label}},
        "_meta": {"id": f"pubmedqa/{split}/{i:06d}", "source": "pubmedqa",
                  "group_id": f"pubmedqa/{split}/{i:06d}", "row": i, "split": split,
                  "repo": "qiaojin/PubMedQA:pqa_labeled", "pubid": str(r.get("pubid", "")),
                  "variant": "clean"},
    }
    prompt = ALPACA_PROMPT.format(question=str(r["question"]),
                                  abstracts=abstracts_text(contexts, max_words, max_abstracts))
    return rec, prompt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class-shot", type=int, default=8,
                    help="balanced rows per class for the stitch's few-shot pool")
    ap.add_argument("--test-cap", type=int, default=200, help="cap on the test set (0 = all)")
    ap.add_argument("--abstract-words", type=int, default=30,
                    help="words kept per abstract (see abstracts_text: this card aborts above ~195 total tokens)")
    ap.add_argument("--abstracts", type=int, default=1, help="how many abstracts to keep")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    a = parse_args()
    from datasets import load_dataset

    raw = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    recs, prompts = [], {}
    for i, r in enumerate(raw):
        rec, prompt = row_record("train", i, r, a.abstract_words, a.abstracts)
        if rec is None:
            continue
        prompts[rec["_meta"]["id"].split("/")[-1]] = {"prompt": prompt, "answer": rec["questions"]["answer"]["label"]}
        recs.append(rec)

    # Deterministic per-class partition: the few-shot pool is drawn first, then the test set is
    # whatever is left, so the pool is never evaluated on.
    rng = random.Random(a.seed)
    by_class = {}
    for rec in recs:
        by_class.setdefault(rec["questions"]["answer"]["label"], []).append(rec)
    for items in by_class.values():
        rng.shuffle(items)
    shot, test = [], []
    for cls, items in sorted(by_class.items()):
        shot.extend(items[: a.per_class_shot])
        test.extend(items[a.per_class_shot:])
    # Shuffle BEFORE capping. Appending per class in sorted order and then slicing drops the
    # tail classes outright -- capping the concatenation kept only `maybe` and `no` and lost
    # every `yes` row, which would have made the test set two-class by accident.
    rng.shuffle(shot)
    rng.shuffle(test)
    if a.test_cap:
        test = test[: a.test_cap]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", shot), ("validation", test)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # The alpaca prompts are keyed by conv_key over the whole file, so both splits can be
        # probed with the same dict; write the union once per split for clarity.
        with (out / f"alpaca_{name}.json").open("w", encoding="utf-8") as f:
            json.dump({r["_meta"]["id"].split("/")[-1]: prompts[r["_meta"]["id"].split("/")[-1]]
                       for r in rows}, f, ensure_ascii=False, indent=2)
        dist = {}
        for r in rows:
            dist[r["questions"]["answer"]["label"]] = dist.get(r["questions"]["answer"]["label"], 0) + 1
        print(f"[pubmed] {name}: {len(rows)} records, class counts {dict(sorted(dist.items()))}")

    # The guard this dataset is meant to exercise: class -> option index must be a bijection, so
    # record the expected mapping here rather than discovering it in the probe.
    print(f"[pubmed] class -> option index (authored order, prep with --no-shuffle): "
          f"{ {k: i for i, k in enumerate(CRITERIA)} }")


if __name__ == "__main__":
    main()
