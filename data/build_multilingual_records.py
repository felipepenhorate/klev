#!/usr/bin/env python
"""Multilingual decision training data from permissively licensed benchmark datasets.

Licences kept: Apache-2.0, MIT, CC-BY-4.0 (attribution recorded in the manifest). Datasets
used (all excluded from our eval suites): MASSIVE intents, XCOPA, XWinograd, TyDi QA, Russian
SuperGLUE, PORTULAN ExtraGLUE, multilingual sentiments, B2W reviews, OLID-BR.

Each dataset contributes a capped train sample and a held-out eval sample (per dataset and
language), so every model can be scored on the same multilingual scope later. The train side
is sized for the <=8h delta budget (target ~12k rows); the eval side is never trained on.

    ./unsloth_uv/bin/python data/build_multilingual_records.py \
        --out-dir /mnt/f/distill_jev_runs/multilingual
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from data.suites import read_jsonl, write_json, write_jsonl

# (dataset, config, split, licence)
LICENCES = {
    "massive": "Apache-2.0 (mteb/amazon_massive_intent)",
    "xcopa": "CC-BY-4.0 (cambridgeltl/xcopa)",
    "xwinograd": "CC-BY-4.0 (muennighoff/xwinograd)",
    "tydiqa": "Apache-2.0 (google-research-datasets/tydiqa)",
    "rusglue": "MIT (RussianNLP/russian_super_glue)",
    "extraglue": "MIT (PORTULAN/extraglue)",
    "sentiments": "Apache-2.0 (tyqiangz/multilingual-sentiments)",
    "b2w": "CC-BY-4.0 (ruanchaves/b2w-reviews01)",
    "olidbr": "CC-BY-4.0 (dougtrajano/olid-br)",
}
NORMALISE = lambda text: " ".join(str(text).casefold().split())
# non-English first; pt-BR gets the largest MASSIVE slice
MASSIVE_LANGS = ["pt", "es", "ru", "ar", "de", "fr", "it", "zh-CN", "ja", "hi", "id", "tr", "vi", "th"]
XCOPA_LANGS = ["it", "id", "tr", "vi", "th", "sw", "zh", "ta"]
XWINO_LANGS = ["pt", "fr", "ru", "ja", "zh"]
TYDIQA_LANGS = ["arabic", "russian", "indonesian", "thai", "japanese", "korean", "finnish", "swahili", "telugu", "bengali"]


def load_any(repo, config=None, split=None, parquet=False):
    kwargs = {"revision": "refs/convert/parquet"} if parquet else {}
    return load_dataset(repo, config, split=split, **kwargs)


def choice(state, qid, instructions, criteria, label):
    return {"state": state, "questions": {qid: {"type": "choice", "instructions": instructions,
                                                "criteria": criteria, "label": label}}}


def noul(state, qid, instructions, label, criteria=None):
    question = {"type": "noul", "instructions": instructions, "label": bool(label)}
    if criteria:
        question["criteria"] = criteria
    return {"state": state, "questions": {qid: question}}


def take(items, cap, rng):
    if cap and len(items) > cap:
        return rng.sample(items, cap)
    return items


# --- adapters: each returns (train, eval) lists of records ---------------------------------

def massive(caps, rng):
    ds = load_any("mteb/amazon_massive_intent", "default", "train")
    ds_eval = load_any("mteb/amazon_massive_intent", "default", "test")
    train, eval_ = [], []
    for split, out, cap in (("train", train, caps["massive_train"]), ("test", eval_, caps["massive_eval"])):
        source = ds if split == "train" else ds_eval
        by_lang, intents = defaultdict(list), defaultdict(set)
        for row in source:
            if row["lang"] in MASSIVE_LANGS:
                by_lang[row["lang"]].append(row)
                intents[row["lang"]].add(row["label_text"])
        per_lang = max(1, cap // max(len(by_lang), 1))
        for lang, rows in sorted(by_lang.items()):
            sample = take(rows, per_lang, rng)
            for row in sample:
                negs = [i for i in sorted(intents[lang]) if i != row["label_text"]]
                if len(negs) < 3:
                    continue
                keys = [row["label_text"]] + rng.sample(negs, 3)
                rng.shuffle(keys)
                instruction = "Qual é a intenção do pedido?" if lang == "pt" else "Which intent best describes this request?"
                out.append({**choice(row["text"], "intent", instruction,
                                     {k: k.replace("_", " ") for k in keys}, row["label_text"]),
                            "_meta": {"source": f"massive_{lang}", "id": f"massive/{lang}/{row['id']}"}})
    return train, eval_


def xcopa(caps, rng):
    train, eval_ = [], []
    for lang in XCOPA_LANGS:
        for split, out, cap in (("validation", train, caps["xcopa_train"]), ("test", eval_, caps["xcopa_eval"])):
            try:
                ds = load_any("cambridgeltl/xcopa", lang, split)
            except Exception:
                continue
            for row in take(list(ds), max(1, cap // len(XCOPA_LANGS)), rng):
                choices = [row["choice1"], row["choice2"]]
                try:
                    label = int(row["label"])
                except Exception:
                    continue
                if label not in (0, 1):
                    continue
                question = ("What is the cause?" if row["question"] == "cause" else "What is the effect?")
                out.append({**choice(row["premise"], "relation", question,
                                     {"a": choices[0], "b": choices[1]}, "ab"[label]),
                            "_meta": {"source": f"xcopa_{lang}", "id": f"xcopa/{lang}/{row['idx']}"}})
    return train, eval_


def xwinograd(caps, rng):
    train, eval_ = [], []
    for lang in XWINO_LANGS:
        try:
            ds = load_any("muennighoff/xwinograd", lang, "test")
        except Exception:
            continue
        rows = list(ds)
        rng.shuffle(rows)
        cut = int(len(rows) * 0.8)
        for split_rows, out, cap in ((rows[:cut], train, caps["xwino_train"]), (rows[cut:], eval_, caps["xwino_eval"])):
            for row in take(split_rows, max(1, cap // len(XWINO_LANGS)), rng):
                if row["answer"] not in ("1", "2"):
                    continue
                out.append({**choice(row["sentence"], "reference", "What does the marked expression refer to?",
                                     {"a": row["option1"], "b": row["option2"]}, "ab"[int(row["answer"]) - 1]),
                            "_meta": {"source": f"xwinograd_{lang}", "id": f"xwinograd/{lang}/{hashlib.sha1(str(row).encode()).hexdigest()[:12]}"}})
    return train, eval_


def tydiqa(caps, rng):
    train, eval_ = [], []
    try:
        ds = load_any("google-research-datasets/tydiqa", "secondary_task", "train")
        ds_eval = load_any("google-research-datasets/tydiqa", "secondary_task", "validation")
    except Exception:
        return train, eval_
    answers_by_lang = defaultdict(list)
    for row in ds:
        if row.get("answers", {}).get("text"):
            answers_by_lang[row["language"]].append(row["answers"]["text"][0])
    for split, source, out, cap in (("train", ds, train, caps["tydiqa_train"]), ("eval", ds_eval, eval_, caps["tydiqa_eval"])):
        rows = [r for r in source if r.get("language") in TYDIQA_LANGS and r.get("answers", {}).get("text")]
        per_lang = max(1, cap // max(len(TYDIQA_LANGS), 1))
        by_lang = defaultdict(list)
        for row in rows:
            by_lang[row["language"]].append(row)
        for lang, lang_rows in sorted(by_lang.items()):
            for row in take(lang_rows, per_lang, rng):
                gold = row["answers"]["text"][0]
                answer_start = row["answers"]["answer_start"][0]
                context = row["context"]
                start = max(0, answer_start - 500)
                end = min(len(context), answer_start + len(gold) + 500)
                passage = context[start:end]
                pool = [a for a in answers_by_lang[lang] if a.strip() and a != gold]
                if len(pool) < 3:
                    continue
                keys = [gold] + rng.sample(pool, 3)
                rng.shuffle(keys)
                out.append({**choice(passage, "answer", row["question"],
                                     {k: None for k in keys}, gold),
                            "_meta": {"source": f"tydiqa_{lang}", "id": f"tydiqa/{lang}/{row['id']}"}})
    return train, eval_


def rusglue(caps, rng):
    train, eval_ = [], []
    try:
        tera = load_any("RussianNLP/russian_super_glue", "tera", "train", parquet=True)
        tera_eval = load_any("RussianNLP/russian_super_glue", "tera", "validation", parquet=True)
    except Exception:
        return train, eval_
    for source, out, cap in ((tera, train, caps["rusglue_train"]), (tera_eval, eval_, caps["rusglue_eval"])):
        for row in take(list(source), cap, rng):
            label = 0 if row["label"] in (0, "entailment", "Entailment") else 1
            out.append({**noul(row["premise"], "follows", f'Does the statement follow from this text: "{row["hypothesis"]}"', label,
                               {"true": "The statement follows", "false": "The statement does not follow or contradicts"}),
                        "_meta": {"source": "rusglue_tera", "id": f"rusglue/tera/{len(out)}"}})
    return train, eval_


def extraglue(caps, rng):
    train, eval_ = [], []
    plan = [("boolq_pt-BR", "noul"), ("copa_pt-BR", "choice2"), ("rte_pt-BR", "choice2"), ("mnli_matched_pt-BR", "choice3")]
    per_config = max(1, caps["extraglue_train"] // len(plan))
    for config, kind in plan:
        try:
            ds = load_any("PORTULAN/extraglue", config, "test")
        except Exception:
            continue
        rows = list(ds)
        rng.shuffle(rows)
        cut = int(len(rows) * 0.8)
        for split_rows, out, cap in ((rows[:cut], train, per_config), (rows[cut:], eval_, max(1, caps["extraglue_eval"] // len(plan)))):
            for row in take(split_rows, cap, rng):
                try:
                    if kind == "noul":
                        label = bool(int(row["label"]))
                        out.append({**noul({"question": row.get("question", ""), "passage": row.get("passage", row.get("sentence1", ""))},
                                           "answer", "Does the passage answer the question?", label),
                                    "_meta": {"source": f"extraglue_{config}", "id": f"extraglue/{config}/{len(out)}"}})
                    elif kind == "choice2":
                        criteria = {"a": row.get("sentence2", row.get("choice1", "")), "b": row.get("sentence1", row.get("choice2", ""))}
                        out.append({**choice(row.get("sentence1", row.get("premise", "")), "relation",
                                             "Which option best fits?", criteria, "ab"[int(row["label"])]),
                                    "_meta": {"source": f"extraglue_{config}", "id": f"extraglue/{config}/{len(out)}"}})
                    else:
                        labels = ["entailment", "neutral", "contradiction"]
                        out.append({**choice(row.get("premise", ""), "relation",
                                             f'How does the hypothesis relate: "{row.get("hypothesis", "")}"',
                                             {k: None for k in labels}, labels[int(row["label"])]),
                                    "_meta": {"source": f"extraglue_{config}", "id": f"extraglue/{config}/{len(out)}"}})
                except Exception:
                    continue
    return train, eval_


def sentiments(caps, rng):
    train, eval_ = [], []
    for lang in ("pt", "es", "ar", "fr", "de", "it"):
        try:
            ds = load_any("tyqiangz/multilingual-sentiments", lang, "train", parquet=True)
            ds_eval = load_any("tyqiangz/multilingual-sentiments", lang, "test", parquet=True)
        except Exception:
            continue
        labels = ["negative", "neutral", "positive"]
        for source, out, cap in ((ds, train, caps["sentiments_train"] // 6), (ds_eval, eval_, caps["sentiments_eval"] // 6)):
            for row in take(list(source), max(1, cap), rng):
                label = int(row.get("label", 0))
                out.append({**choice(row.get("text", ""), "sentiment", "What is the sentiment of this message?",
                                     {k: None for k in labels}, labels[min(label, 2)]),
                            "_meta": {"source": f"sentiments_{lang}", "id": f"sentiments/{lang}/{len(out)}"}})
    return train, eval_


def b2w(caps, rng):
    train, eval_ = [], []
    try:
        ds = load_any("ruanchaves/b2w-reviews01", None, "train", parquet=True)
    except Exception:
        return train, eval_
    rows = list(ds)
    rng.shuffle(rows)
    cut = int(len(rows) * 0.8)
    text_key = next((k for k in ("review_text", "review_text_processed", "text") if k in rows[0]), None)
    score_key = next((k for k in ("overall_rating", "rating", "review_score") if k in rows[0]), None)
    if not text_key or not score_key:
        return train, eval_
    levels = ["1 star: terrible", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"]
    for split_rows, out, cap in ((rows[:cut], train, caps["b2w_train"]), (rows[cut:], eval_, caps["b2w_eval"])):
        for row in take(split_rows, cap, rng):
            try:
                stars = int(round(float(row[score_key])))
            except Exception:
                continue
            if not 1 <= stars <= 5:
                continue
            text = str(row[text_key])[:600]
            out.append({**{"state": text, "questions": {
                "rating": {"type": "score", "instructions": "Quantas estrelas este avaliador deu?",
                           "criteria": levels, "label": stars - 1},
                "recommend": {"type": "noul", "instructions": "Este avaliador recomendaria o produto?",
                              "criteria": {"true": "Avaliação claramente positiva", "false": "Negativa ou mista"},
                              "label": stars >= 4}}},
                "_meta": {"source": "b2w", "id": f"b2w/{len(out)}"}})
    return train, eval_


def olidbr(caps, rng):
    train, eval_ = [], []
    try:
        ds = load_any("dougtrajano/olid-br", None, "train")
        ds_eval = load_any("dougtrajano/olid-br", None, "test")
    except Exception:
        return train, eval_
    for source, out, cap in ((ds, train, caps["olidbr_train"]), (ds_eval, eval_, caps["olidbr_eval"])):
        for row in take(list(source), cap, rng):
            if not str(row.get("text", "")).strip():
                continue
            label = str(row.get("is_offensive", "")).upper().startswith("OFF")
            out.append({**noul(row["text"], "offensive", "Esta mensagem é ofensiva?", label,
                               {"true": "Contém insultos, ameaças ou ódio", "false": "Não é ofensiva"}),
                        "_meta": {"source": "olidbr", "id": f"olidbr/{row.get('id', len(out))}"}})
    return train, eval_


ADAPTERS = {"massive": massive, "xcopa": xcopa, "xwinograd": xwinograd, "tydiqa": tydiqa,
            "rusglue": rusglue, "extraglue": extraglue, "sentiments": sentiments, "b2w": b2w, "olidbr": olidbr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-train", type=int, default=12000, help="cap on multilingual training records (excl. replay)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contamination", default="", help="extra JSONL files to screen against")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    caps = {"massive_train": 5000, "massive_eval": 1200, "xcopa_train": 800, "xcopa_eval": 800,
            "xwino_train": 1800, "xwino_eval": 400, "tydiqa_train": 1500, "tydiqa_eval": 300,
            "rusglue_train": 800, "rusglue_eval": 200, "extraglue_train": 1500, "extraglue_eval": 300,
            "sentiments_train": 900, "sentiments_eval": 300, "b2w_train": 800, "b2w_eval": 200,
            "olidbr_train": 800, "olidbr_eval": 300}

    train, eval_ = [], []
    for name, adapter in ADAPTERS.items():
        try:
            t, e = adapter(caps, rng)
            train += t
            eval_ += e
            print(f"[multi] {name}: train {len(t)} eval {len(e)}", flush=True)
        except Exception as exc:
            print(f"[multi] {name} SKIPPED: {type(exc).__name__}: {str(exc)[:120]}", flush=True)

    # cap the training side proportionally across sources, keep all eval
    by_source = defaultdict(list)
    for record in train:
        by_source[record["_meta"]["source"]].append(record)
    capped = []
    for source, rows in sorted(by_source.items()):
        share = max(1, int(a.max_train * len(rows) / len(train)))
        capped += take(rows, share, rng)
    train = capped

    # contamination screen (normalised state hashes)
    hashes = set()
    runs = Path("/mnt/f/distill_jev_runs")
    paths = [ROOT / "evals/v7/decision-v7/development.jsonl", ROOT / "evals/v4/transfer-v4/development.jsonl",
             *[p for p in a.contamination.split(",") if p]] + sorted((runs / "bench").glob("*.jsonl"))
    for path in paths:
        for record in read_jsonl(path):
            state = record.get("state")
            text = json.dumps(state, sort_keys=True, ensure_ascii=False) if not isinstance(state, str) else state
            hashes.add(hashlib.sha256(NORMALISE(text).encode()).hexdigest())
    clean = []
    dropped = 0
    for record in train + eval_:
        state = record["state"]
        text = json.dumps(state, sort_keys=True, ensure_ascii=False) if not isinstance(state, str) else state
        if hashlib.sha256(NORMALISE(text).encode()).hexdigest() in hashes:
            dropped += 1
            continue
        clean.append(record)
    clean_ids = {id(r) for r in clean}
    train = [r for r in train if id(r) in clean_ids]
    eval_ = [r for r in eval_ if id(r) in clean_ids]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "eval.jsonl", eval_)
    manifest = {"seed": a.seed, "max_train": a.max_train, "contamination_dropped": dropped,
                "licences": LICENCES, "train_records": len(train), "eval_records": len(eval_),
                "train_sources": dict(Counter(r["_meta"]["source"] for r in train)),
                "eval_sources": dict(Counter(r["_meta"]["source"] for r in eval_)),
                "attribution": "CC-BY-4.0 datasets require attribution; MIT/Apache-2.0 require licence retention."}
    write_json(out / "manifest.json", manifest)
    print(f"[multi] train {len(train)} / eval {len(eval_)} -> {out}", flush=True)
    print(f"[multi] train sources {dict(Counter(r['_meta']['source'] for r in train))}", flush=True)
    print(f"[multi] eval sources {dict(Counter(r['_meta']['source'] for r in eval_))}", flush=True)


if __name__ == "__main__":
    main()
