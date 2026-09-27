#!/usr/bin/env python
"""Synthetic system_one training data from the local Qwen3.5-9B (llama.cpp, SPEC 3.5).

Two-step generation per record: first the document as plain text, then the questions as
structured JSON conditioned on that document. (One-step JSON made the model stuff status
words into `state` and the document into the criteria descriptions; conditioning fixes it.)

Families (all supervised for the decision format, not chat):
  policy        policy/case documents + routing (choice), yes/no (noul), severity (score)
  rule          AND/OR/NOT/unless rules over stated facts, with an exception case
  abstention    a question whose answer is deliberately absent -> garbage target
  ambiguous     genuinely ambiguous case -> soft `target` distribution
  long_document 600-900 word manual with the deciding rule buried

Every record is validated (shapes + label/criteria consistency), deduplicated on the
normalized state, screened against the local eval partitions (exact normalized hash), and
optionally verified by a third pass that answers the questions from the state alone.

    ./unsloth_uv/bin/python data/synth/generate_records.py \
        --n 2000 --workers 4 --out /mnt/f/distill_jev_runs/synth/qwen35_v1.jsonl
"""
import argparse
import hashlib
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import requests

from data.suites import read_jsonl, write_json

SCHEMA = json.loads((ROOT / "data" / "synth" / "schema.json").read_text(encoding="utf-8"))
QUESTIONS_SCHEMA = {"name": "questions", "strict": True, "schema": {
    "type": "object", "properties": {"questions": SCHEMA["schema"]["properties"]["questions"]},
    "required": ["questions"], "additionalProperties": False}}
VERIFY_SCHEMA = {"name": "answers", "strict": True, "schema": {
    "type": "object", "properties": {"answers": {"type": "array", "minItems": 1, "items": {
        "type": "object", "properties": {"q": {"type": "string"}, "answer": {"type": "string"}},
        "required": ["q", "answer"], "additionalProperties": False}}},
    "required": ["answers"], "additionalProperties": False}}

DOMAINS = ["healthcare triage", "insurance claims", "legal intake", "travel disruption", "HR requests",
           "SaaS customer support", "e-commerce returns", "banking disputes", "university student services", "logistics"]

DOC = {
    "policy": """Write a realistic {domain} case document of 150-400 words: a policy, a ticket thread, or a case note. It must state the facts needed to answer routing and priority questions about it. Return only the document text, no JSON, no headings like "Document:".""",
    "rule": """Write a short business rule that uses several conditions (AND, OR, NOT, unless, and at least one exception) over named facts, followed by a case that states every fact explicitly. 120-300 words. Return only the text, no JSON.""",
    "abstention": """Write a policy or case note of 120-300 words about {domain}. It must be complete and realistic, but deliberately omit the single fact that a reader would need to decide one specific question (for example a deadline, a threshold, or whether an approval exists). Return only the document text, no JSON.""",
    "ambiguous": """Write a case of 120-300 words about {domain} that is genuinely ambiguous between two options: both are defensible from the text. Return only the case text, no JSON.""",
    "long_document": """Write a policy manual of 600-900 words about {domain} with 4-6 sections (definitions, eligibility, exceptions, escalation, service levels). Bury one deciding rule in the middle. Return only the manual text, no JSON.""",
}

QUESTIONS = {
    "policy": """Read the document and write 2-3 questions a routing system would ask about this case.
- one choice question with 3-5 concrete options (short option keys, clear descriptions);
- one noul question; its criteria keys must be exactly "false" and "true";
- optionally one score question whose criteria keys are "0","1","2",... ordered low to high.
Every label must be the single option the document supports. Return JSON only.""",
    "rule": """Read the rule and case. Write 1-2 questions whose answers follow strictly from the rule and the stated facts. Include one question where ignoring the exception gives the wrong answer. Use choice and/or noul questions (noul criteria keys exactly "false"/"true"). Return JSON only.""",
    "abstention": """Read the document. Write exactly ONE question with 3-5 plausible options whose answer is deliberately NOT stated in the document. Set "garbage" to true and set "label" to the first criteria key. Do not include the missing fact. Return JSON only.""",
    "ambiguous": """Read the case. Write exactly ONE question with 3-5 options where two options are genuinely defensible. Set "label" to the more likely key and provide "target" probabilities over the criteria keys that sum to 1. Return JSON only.""",
    "long_document": """Read the manual. Write 1-2 questions whose answers require reading the whole document, using choice and/or noul questions. Every label must follow from the manual. Return JSON only.""",
}

MIN_WORDS = {"policy": 100, "rule": 80, "abstention": 80, "ambiguous": 80, "long_document": 400}
DOC_TOKENS = {"policy": 700, "rule": 600, "abstention": 600, "ambiguous": 600, "long_document": 1500}
FAMILIES = {"policy": 0.40, "rule": 0.20, "abstention": 0.15, "ambiguous": 0.15, "long_document": 0.10}
NORMALISE = lambda text: " ".join(text.casefold().split())


def resolve_key(label, descriptions):
    """Model labels sometimes repeat a description, change case, or use the option letter;
    map them back to a key."""
    if label in descriptions:
        return label
    folded = label.casefold()
    for key in descriptions:
        if key.casefold() == folded:
            return key
    for key, description in descriptions.items():
        if folded and (folded in description.casefold() or description.casefold() in folded):
            return key
    letter = re.search(r"\b([A-Ea-e])\b", label)
    if letter:
        for key in descriptions:
            if key.casefold() == letter.group(1).casefold():
                return key
    return label

_stats_lock = threading.Lock()
STATS = {"generated": 0, "kept": 0, "invalid": 0, "unverified": 0, "duplicate": 0, "contaminated": 0, "errors": 0}


def normalised_hashes(paths):
    hashes = set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        for record in read_jsonl(path):
            state = record.get("state")
            text = json.dumps(state, sort_keys=True, ensure_ascii=False) if not isinstance(state, str) else state
            hashes.add(hashlib.sha256(NORMALISE(text).encode()).hexdigest())
    return hashes


def chat(api_base, messages, schema, max_tokens, seed):
    payload = {"model": "qwen", "messages": messages,
               "chat_template_kwargs": {"enable_thinking": False},
               "temperature": 1.0, "top_p": 0.95, "max_tokens": max_tokens, "seed": seed}
    if schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": schema}
    for attempt in range(3):
        try:
            response = requests.post(f"{api_base}/chat/completions", json=payload, timeout=900)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def to_request(state, raw, family, index, prompt_hash):
    """(document, generated questions) -> TypeSafe-shaped request record, or ValueError."""
    if not isinstance(state, str) or len(state.split()) < MIN_WORDS[family]:
        raise ValueError(f"document too short ({len(str(state).split())} words)")
    questions = {}
    for i, q in enumerate(raw.get("questions") or []):
        qtype = q.get("type")
        criteria = q.get("criteria") or []
        keys = [str(c["key"]) for c in criteria]
        if len(keys) != len(set(keys)) or not keys:
            raise ValueError("duplicate or empty criteria keys")
        descriptions = {str(c["key"]): str(c["description"]) for c in criteria}
        label = str(q.get("label", "")).strip()
        label = resolve_key(label, descriptions)
        question = {"type": qtype, "instructions": str(q.get("instructions", "")).strip(), "src": f"synth_{family}"}
        if qtype == "noul":
            if set(keys) != {"false", "true"}:
                if len(keys) != 2:
                    raise ValueError("noul needs exactly two criteria")
                negative = next((k for k in keys if any(w in k.casefold() for w in ("no", "false", "not", "deny"))), keys[0])
                order = [negative] + [k for k in keys if k != negative]
                descriptions = {("false" if i == 0 else "true"): descriptions[k] for i, k in enumerate(order)}
            if label.lower() not in ("true", "false"):
                raise ValueError("noul label must be true/false")
            question["criteria"] = descriptions
            question["label"] = label.lower() == "true"
        elif qtype == "choice":
            if label not in descriptions:
                raise ValueError("choice label not a criteria key")
            question["criteria"] = descriptions
            question["label"] = label
        elif qtype == "score":
            if label.isdigit():
                level = int(label)
            elif label in descriptions:
                level = keys.index(label)
            else:
                raise ValueError("score label is neither an index nor a key")
            if not 0 <= level < len(keys):
                raise ValueError("score label out of range")
            question["criteria"] = [descriptions[k] for k in keys]
            question["label"] = level
        else:
            raise ValueError(f"unknown question type {qtype}")
        if q.get("garbage") is True:
            question["garbage"] = 1.0
        if q.get("target"):
            target = {str(t["key"]): float(t["p"]) for t in q["target"] if str(t["key"]) in descriptions}
            if not target or not 0.9 <= sum(target.values()) <= 1.1:
                raise ValueError("target probabilities do not sum to 1")
            total = sum(target.values())
            question["target"] = {k: v / total for k, v in target.items()}
        questions[f"q{i + 1}"] = question
    if not questions:
        raise ValueError("no questions")
    return {"state": state.strip(), "questions": questions,
            "_meta": {"source": f"synth_{family}", "id": f"synth_{family}/{index}", "family": family,
                      "generator": "qwen3.5-9b-ud-q4_k_xl", "prompt_sha256": prompt_hash}}


def answers_match(question, got, expected):
    criteria = question["criteria"]
    keys = list(criteria) if isinstance(criteria, dict) else [str(i) for i in range(len(criteria))]
    folded = got.strip().casefold()
    if question.get("garbage"):
        return folded in ("unanswerable", "none", "none of the above", "cannot be determined")
    if question.get("target"):   # ambiguous by construction: any plausible option (or unanswerable) is fine
        if folded in ("unanswerable", "none", "cannot be determined"):
            return True
        plausible = {k.casefold() for k, p in question["target"].items() if p >= 0.15}
        return folded in plausible or folded == str(expected).casefold()
    return folded == str(expected).casefold() or folded in {k.casefold() for k in keys if str(expected).casefold() == k.casefold()}


def verify(api_base, record, seed):
    """Third pass: answer the questions from the state alone; keep only agreeing records."""
    questions = []
    for qid, q in record["questions"].items():
        criteria = q["criteria"]
        keys = list(criteria) if isinstance(criteria, dict) else [str(i) for i in range(len(criteria))]
        questions.append({"q": qid, "instructions": q["instructions"], "keys": keys, "type": q["type"]})
    prompt = ("Read the document and answer each question using ONLY the document. "
              "If the document does not support any option, answer \"unanswerable\".\n\n"
              f"Document:\n{record['state']}\n\nQuestions:\n{json.dumps(questions, ensure_ascii=False)}\n\n"
              "Return JSON with one answer per question id, using a criteria key or \"unanswerable\".")
    raw = json.loads(chat(api_base, [{"role": "user", "content": prompt}], VERIFY_SCHEMA, 400, seed))
    answers = {a["q"]: str(a["answer"]).strip() for a in raw.get("answers", [])}
    for qid, q in record["questions"].items():
        expected = "unanswerable" if q.get("garbage") else str(q["label"]).lower() if q["type"] == "noul" else str(q["label"])
        if not answers_match(q, answers.get(qid, ""), expected):
            return f"{qid}: got {answers.get(qid, '')!r}, expected {expected!r}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://127.0.0.1:8083/v1")
    ap.add_argument("--n", type=int, default=2000, help="target records to keep")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--contamination", default="", help="extra JSONL files to screen against")
    ap.add_argument("--debug-invalid", type=int, default=0, help="save the first N invalid generations")
    ap.add_argument("--debug-unverified", type=int, default=0, help="save the first N verification mismatches")
    ap.add_argument("--append", action="store_true", help="keep existing records and generate up to --n total")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    contamination = normalised_hashes([
        ROOT / "evals/v7/decision-v7/development.jsonl",
        ROOT / "evals/v4/transfer-v4/development.jsonl",
        *[p for p in a.contamination.split(",") if p],
    ])
    seen = set()
    mode = "w"
    if a.append and out.exists():
        existing = read_jsonl(out)
        mode = "a"
        for record in existing:
            seen.add(hashlib.sha256(NORMALISE(record["state"]).encode()).hexdigest())
        print(f"[synth] appending: {len(existing)} records already in {out}", flush=True)
    families = {name: FAMILIES[name] for name in a.families.split(",") if name in FAMILIES}
    names = list(families)
    weights = [families[name] for name in names]

    def fail(reason, family, raw=None):
        with _stats_lock:
            STATS["invalid"] += 1
            reasons = STATS.setdefault("invalid_reasons", {})
            reasons[reason] = reasons.get(reason, 0) + 1
            if a.debug_invalid and STATS["invalid"] <= a.debug_invalid:
                with out.with_suffix(".invalid.jsonl").open("a", encoding="utf-8") as dbg:
                    dbg.write(json.dumps({"family": family, "error": reason, "raw": raw}, ensure_ascii=False) + "\n")

    def generate_one(index):
        family = rng.choices(names, weights=weights, k=1)[0]
        domain = rng.choice(DOMAINS)
        seed = rng.randrange(1 << 30)
        prompt_hash = hashlib.sha256(f"{family}:{seed}".encode()).hexdigest()[:16]
        try:
            document = chat(a.api_base, [{"role": "user", "content": DOC[family].format(domain=domain)}], None, DOC_TOKENS[family], seed).strip()
            if len(document.split()) < MIN_WORDS[family]:
                fail(f"document too short ({len(document.split())} words)", family)
                return None
            raw = json.loads(chat(a.api_base, [{"role": "user", "content": QUESTIONS[family] + f"\n\nDocument:\n{document}"}],
                                  QUESTIONS_SCHEMA, 600, seed + 1))
            record = to_request(document, raw, family, index, prompt_hash)
        except Exception as e:
            fail(f"{type(e).__name__}: {str(e)[:80]}", family, raw if "raw" in dir() else None)
            return None
        digest = hashlib.sha256(NORMALISE(record["state"]).encode()).hexdigest()
        with _stats_lock:
            STATS["generated"] += 1
            if digest in seen or digest in contamination:
                STATS["contaminated" if digest in contamination else "duplicate"] += 1
                return None
            seen.add(digest)
        if not a.no_verify:
            try:
                reason = verify(a.api_base, record, seed + 2)
                if reason:
                    with _stats_lock:
                        STATS["unverified"] += 1
                        if a.debug_unverified and STATS["unverified"] <= a.debug_unverified:
                            with out.with_suffix(".unverified.jsonl").open("a", encoding="utf-8") as dbg:
                                dbg.write(json.dumps({"family": family, "reason": reason, "state": record["state"][:600],
                                                      "questions": {k: {"label": v.get("label"), "garbage": bool(v.get("garbage")),
                                                                         "keys": list(v["criteria"]) if isinstance(v["criteria"], dict) else v["criteria"]}
                                                                    for k, v in record["questions"].items()}}, ensure_ascii=False) + "\n")
                    return None
            except Exception:
                with _stats_lock:
                    STATS["errors"] += 1
                return None
        with _stats_lock:
            STATS["kept"] += 1
        return record

    start = time.time()
    kept = len(seen) if mode == "a" else 0
    index = 0
    max_attempts = a.n * 8
    with out.open(mode, encoding="utf-8", newline="\n") as f, ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = set()
        while kept < a.n and index < max_attempts:
            while len(futures) < a.workers * 2 and index < max_attempts and kept + len(futures) < a.n * 2:
                futures.add(pool.submit(generate_one, index)); index += 1
            if not futures:
                break
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for finished in done:
                try:
                    record = finished.result()
                except Exception:
                    record = None
                if record is not None:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n"); f.flush(); kept += 1
                    if kept % 25 == 0:
                        rate = kept / max(time.time() - start, 1)
                        print(f"[synth] kept {kept}/{a.n} in {time.time() - start:.0f}s ({rate:.2f}/s) {STATS}", flush=True)
    write_json(out.with_suffix(".stats.json"), {**STATS, "target": a.n, "seconds": round(time.time() - start, 1),
                                                "out": str(out), "families": list(families)})
    print(f"[synth] done: {kept} records -> {out} ({time.time() - start:.0f}s) {STATS}", flush=True)


if __name__ == "__main__":
    main()
