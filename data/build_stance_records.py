#!/usr/bin/env python
"""Conversational stance datasets as system_one decision records.

CoSt-BR (Brazilian Portuguese Reddit conversations; cost-br/parsed_data/csv):
    state = claim + up to two parent messages + the current message
    question = choice over Concorda / Discorda / Discute / Irrelevante / Pede Informacoes

Stanceosaurus (multilingual tweets; hydrated prompts in
training-conversational-stance/Datasets/Stanceossaurus): the prompt files carry the claim,
the conversation and the label, so the instruction block is stripped and the claim +
conversation become the state, with our own stance question.

    ./unsloth_uv/bin/python data/build_stance_records.py --dataset stanceosaurus \
        --prompts .../prompt_eval_stanceossaurus_claim.json,...spanish.json,...russian.json,...arabic.json \
        --limit-per-language 1500 --out /mnt/f/distill_jev_runs/bench/stanceosaurus.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from data.suites import write_jsonl

COSTBR = Path("./github/cost-br/parsed_data/csv")
STANCEOSAURUS = Path("./github/training-conversational-stance/Datasets/Stanceossaurus")
COSTBR_LABELS = {
    "Concorda": ("concorda", "O autor concorda com a afirmação"),
    "Discorda": ("discorda", "O autor discorda da afirmação"),
    "Discute": ("discute", "O autor discute o tema sem tomar partido"),
    "Irrelevante": ("irrelevante", "A mensagem não é sobre a afirmação"),
    "Pede Informações": ("pede_informacoes", "O autor pede mais informações"),
}
STANCE_LABELS = {
    "Supporting": ("supporting", "The current message supports the claim"),
    "Refuting": ("refuting", "The current message refutes or questions the claim's validity"),
    "Querying": ("querying", "The current message asks for more information about the claim"),
    "Discussing": ("discussing", "The current message discusses the topic without taking a side"),
    "Irrelevant": ("irrelevant", "The current message is not about the claim"),
}
CLAIM_MARKER = 'The claim being discussed is the following "'


def costbr(split, max_parents):
    frame = pd.read_csv(COSTBR / f"{split}_data.csv").fillna("")
    by_id = {row["id"]: row for _, row in frame.iterrows()}
    records = []
    for _, row in frame.iterrows():
        if not str(row["text"]).strip() or row["stance"] not in COSTBR_LABELS:
            continue
        chain, parent = [], row["reply_to"]
        while parent and len(chain) < max_parents and parent in by_id:
            parent_row = by_id[parent]
            if str(parent_row["text"]).strip():
                chain.append({"message": str(parent_row["text"]).strip()})
            parent = parent_row["reply_to"]
        state = {"claim": str(row["claim"]).strip(), "conversation": list(reversed(chain)),
                 "current_message": str(row["text"]).strip()}
        key, _ = COSTBR_LABELS[row["stance"]]
        records.append({"state": state,
                        "questions": {"stance": {"type": "choice",
                                                 "instructions": "Qual é a posição da mensagem atual em relação à afirmação?",
                                                 "criteria": {k: d for k, d in COSTBR_LABELS.values()},
                                                 "label": key, "src": "costbr"}},
                        "_meta": {"source": "costbr", "id": f"costbr/{row['source']}/{row['id']}"}})
    return records


def parse_stanceosaurus_prompt(prompt):
    """Strip the instruction block; return (claim, conversation body) or None."""
    start = prompt.find(CLAIM_MARKER)
    if start < 0:
        return None
    rest = prompt[start + len(CLAIM_MARKER):]
    quote = rest.find('"\n')
    if quote < 0:
        return None
    claim = rest[:quote].strip()
    after = rest[quote + 2:]
    for marker in ("Conversation:", "Current Message:"):
        body_start = after.find(marker)
        if body_start >= 0:
            body = after[body_start + len(marker):]
            break
    else:
        return None
    for end_marker in ("\n\nGiven ", "\nGiven ", "\nAnswer:"):
        end = body.find(end_marker)
        if end >= 0:
            body = body[:end]
            break
    return claim, body.strip()


def stanceosaurus(prompts, limit, seed):
    import random
    rng = random.Random(seed)
    records = []
    for path in [p for p in prompts.split(",") if p]:
        path = Path(path)
        language = ("spanish" if "spanish" in path.name else "russian" if "russian" in path.name
                    else "arabic" if "arabic" in path.name else "english")
        entries = json.loads(path.read_text(encoding="utf-8"))
        items = list(entries.items())
        if limit and len(items) > limit:
            items = rng.sample(items, limit)
        built = 0
        for key, entry in items:
            label = entry.get("label")
            if label not in STANCE_LABELS or not entry.get("prompt"):
                continue
            parsed = parse_stanceosaurus_prompt(entry["prompt"])
            if not parsed:
                continue
            claim, conversation = parsed
            key_name, _ = STANCE_LABELS[label]
            records.append({"state": {"claim": claim, "conversation": conversation},
                            "questions": {"stance": {"type": "choice",
                                                     "instructions": "What is the stance of the current message toward the claim?",
                                                     "criteria": {k: d for k, d in STANCE_LABELS.values()},
                                                     "label": key_name, "src": f"stanceosaurus_{language}"}},
                            "_meta": {"source": f"stanceosaurus_{language}", "id": f"stanceosaurus_{language}/{key}"}})
            built += 1
        print(f"[stance] {language}: {built} records from {path.name}", flush=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["costbr", "stanceosaurus"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-parents", type=int, default=2)
    ap.add_argument("--prompts", default=",".join(str(STANCEOSAURUS / name) for name in (
        "prompt_eval_stanceossaurus_claim.json", "prompt_eval_stanceossaurus_claim_spanish.json",
        "prompt_eval_stanceossaurus_claim_russian.json", "prompt_eval_stanceossaurus_claim_arabic.json")))
    ap.add_argument("--limit-per-language", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.dataset == "costbr":
        records = costbr(a.split, a.max_parents)
    else:
        records = stanceosaurus(a.prompts, a.limit_per_language, a.seed)
    write_jsonl(a.out, records)
    print(f"[stance] {len(records)} {a.dataset} records -> {a.out}")
    print(f"[stance] labels {dict(Counter(r['questions']['stance']['label'] for r in records))}")
    print(f"[stance] sources {dict(Counter(r['_meta']['source'] for r in records))}")


if __name__ == "__main__":
    main()
