#!/usr/bin/env python
"""Out-of-distribution multilingual eval suite (eval only; never trained on).

Tasks/languages: XNLI (14 non-English languages, 3-way NLI), Belebele (20 languages, 4-way
reading comprehension), XStoryCloze (10 languages, 2-way story completion), PAWS-X (6
languages, paraphrase `noul`), MasakhaNEWS (8 African languages + fr, topic choice).
Together with Stanceosaurus and CoSt-BR (built separately) this is the multilingual OOD
scope every model is scored on.

    ./unsloth_uv/bin/python data/build_ood_multilingual.py \
        --out /mnt/f/distill_jev_runs/ood-multilingual.jsonl
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from data.suites import write_jsonl

XNLI_LANGS = ["ar", "bg", "de", "el", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
BELEBELE_LANGS = ["por_Latn", "spa_Latn", "arb_Arab", "rus_Cyrl", "deu_Latn", "fra_Latn", "ita_Latn",
                  "zho_Hans", "jpn_Jpan", "hin_Deva", "ind_Latn", "tur_Latn", "vie_Latn", "tha_Thai",
                  "swh_Latn", "ell_Grek", "heb_Hebr", "kor_Hang", "pol_Latn", "nld_Latn"]
XSTORY_LANGS = ["ru", "zh", "es", "ar", "hi", "id", "te", "sw", "eu", "my"]
PAWSX_LANGS = ["de", "es", "fr", "ja", "ko", "zh"]
MASAKHA_LANGS = ["swa", "amh", "hau", "ibo", "yor", "fra", "som", "lug"]
NLI_LABELS = ["entailment", "neutral", "contradiction"]


def choice(state, instructions, criteria, label, source, item_id):
    return {"state": state,
            "questions": {"answer": {"type": "choice", "instructions": instructions,
                                     "criteria": criteria, "label": label, "src": source}},
            "_meta": {"source": source, "id": item_id}}


def noul(state, instructions, label, criteria, source, item_id):
    return {"state": state,
            "questions": {"answer": {"type": "noul", "instructions": instructions,
                                     "criteria": criteria, "label": bool(label), "src": source}},
            "_meta": {"source": source, "id": item_id}}


def xnli(per_lang, rng):
    out = []
    for lang in XNLI_LANGS:
        try:
            ds = load_dataset("facebook/xnli", lang, split="test")
        except Exception:
            continue
        rows = rng.sample(list(ds), min(per_lang, len(ds)))
        for n, row in enumerate(rows):
            label = NLI_LABELS[int(row["label"])]
            out.append(choice({"premise": row["premise"], "hypothesis": row["hypothesis"]},
                              "How does the hypothesis relate to the premise?",
                              {k: None for k in NLI_LABELS}, label, f"xnli_{lang}", f"xnli/{lang}/{n}"))
    return out


def belebele(per_lang, rng):
    out = []
    for lang in BELEBELE_LANGS:
        try:
            ds = load_dataset("facebook/belebele", lang, split="test")
        except Exception:
            continue
        rows = rng.sample(list(ds), min(per_lang, len(ds)))
        for n, row in enumerate(rows):
            answers = [row[f"mc_answer{i}"] for i in range(1, 5)]
            label = "abcd"[int(row["correct_answer_num"]) - 1]
            out.append(choice({"passage": row["flores_passage"], "question": row["question"]},
                              "Which option correctly answers the question?",
                              dict(zip("abcd", answers)), label, f"belebele_{lang}", f"belebele/{lang}/{n}"))
    return out


def xstory(per_lang, rng):
    out = []
    for lang in XSTORY_LANGS:
        try:
            ds = load_dataset("juletxara/xstory_cloze", lang, split="eval")
        except Exception:
            continue
        rows = rng.sample(list(ds), min(per_lang, len(ds)))
        for n, row in enumerate(rows):
            story = " ".join(str(row[f"input_sentence_{i}"]) for i in range(1, 5))
            label = "ab"[int(row["answer_right_ending"]) - 1]
            out.append(choice(story, "Which ending completes the story?",
                              {"a": row["sentence_quiz1"], "b": row["sentence_quiz2"]}, label,
                              f"xstory_{lang}", f"xstory/{lang}/{n}"))
    return out


def pawsx(per_lang, rng):
    out = []
    for lang in PAWSX_LANGS:
        try:
            ds = load_dataset("google-research-datasets/paws-x", lang, split="test")
        except Exception:
            continue
        rows = rng.sample(list(ds), min(per_lang, len(ds)))
        for n, row in enumerate(rows):
            label = int(row["label"]) == 1
            out.append(noul({"sentence1": row["sentence1"], "sentence2": row["sentence2"]},
                            "Do the two sentences mean the same thing?", label,
                            {"true": "Same meaning", "false": "Different meaning"},
                            f"pawsx_{lang}", f"pawsx/{lang}/{n}"))
    return out


def masakhanews(per_lang, rng):
    out = []
    for lang in MASAKHA_LANGS:
        try:
            ds = load_dataset("masakhane/masakhanews", lang, split="test")
        except Exception:
            continue
        names = ds.features["label"].names
        rows = rng.sample(list(ds), min(per_lang, len(ds)))
        for n, row in enumerate(rows):
            label = names[int(row["label"])]
            negatives = rng.sample([x for x in names if x != label], 3)
            keys = [label] + negatives
            rng.shuffle(keys)
            out.append(choice(str(row.get("text", row.get("headline", "")))[:800],
                              "Which topic is this news item about?",
                              {k: None for k in keys}, label, f"masakhanews_{lang}", f"masakha/{lang}/{n}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-lang", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = []
    for name, fn in (("xnli", xnli), ("belebele", belebele), ("xstory", xstory), ("pawsx", pawsx), ("masakhanews", masakhanews)):
        try:
            rows = fn(a.per_lang, rng)
            out += rows
            print(f"[ood] {name}: {len(rows)}", flush=True)
        except Exception as exc:
            print(f"[ood] {name} SKIPPED: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
    write_jsonl(a.out, out)
    sources = Counter(r["_meta"]["source"].split("_")[0] for r in out)
    print(f"[ood] {len(out)} records -> {a.out}")
    print(f"[ood] by task {dict(sources)}")
    print(f"[ood] languages {sorted({r['_meta']['source'] for r in out})}")


if __name__ == "__main__":
    main()
