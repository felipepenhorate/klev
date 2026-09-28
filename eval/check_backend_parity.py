#!/usr/bin/env python
"""Backend parity: the same checkpoint, predictions, `unsloth` vs `plain`.

Unsloth is what makes the 4-bit *fine-tune* fit one consumer GPU. A prediction is one forward
pass, so the question is whether it is needed at inference at all. This scores both loaders on
the same records and writes the per-row probabilities, so the comparison is not just an
accuracy number but a distribution comparison.

Run it in the two environments (one with unsloth, one without) and diff the two outputs:

    # with unsloth
    python eval/check_backend_parity.py --backend unsloth --out /tmp/backends/unsloth.json
    # without unsloth installed at all
    python eval/check_backend_parity.py --backend plain  --out /tmp/backends/plain.json
    python eval/check_backend_parity.py --compare /tmp/backends/unsloth.json /tmp/backends/plain.json

`--compare` needs no model, so it is safe to run anywhere.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/f/distill_jev_runs/main",
                    help="local checkpoint dir (adapter/ + head.pt) or Hub repo id")
    ap.add_argument("--backend", default="unsloth", choices=["unsloth", "plain"])
    ap.add_argument("--base", default=None)
    ap.add_argument("--preset", default="")
    ap.add_argument("--data", default="/mnt/f/distill_jev_runs/bench/costbr-test.jsonl")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-state", type=int, default=1024, help="answer() state-token cap")
    ap.add_argument("--out", default="", help="where to write the per-row output (not needed with --compare)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="two outputs of this script; prints the parity table and exits")
    return ap.parse_args()


def score(a):
    from model.infer import answer, load_klev

    records = [json.loads(l) for l in open(a.data)]
    if a.limit:
        records = records[:a.limit]
    start = time.time()
    model, tokenizer, head = load_klev(a.ckpt, base=a.base, preset=a.preset, backend=a.backend)
    load_s = time.time() - start

    rows, start = [], time.time()
    for i, rec in enumerate(records):
        qid = next(iter(rec["questions"]))
        gold = rec["questions"][qid].get("label")
        try:
            result = answer(model, tokenizer, head, rec, temperature=a.temperature,
                            max_state=a.max_state)
        except ValueError as e:            # a row over the cap is skipped, identically in both backends
            rows.append({"id": rec.get("_meta", {}).get("id", str(i)), "gold": gold,
                         "skipped": str(e)[:80]})
            continue
        got = result["answers"][qid]
        rows.append({"id": rec.get("_meta", {}).get("id", str(i)), "gold": gold,
                     "choice": got.get("choice") or got.get("noul"),
                     "confidence": got.get("confidence"), "none": got.get("none"),
                     "probabilities": got.get("probabilities")})
        if (i + 1) % 50 == 0:
            print(f"[{a.backend}] {i + 1}/{len(records)}", flush=True)
    scored = [r for r in rows if "skipped" not in r]
    correct = sum(r["choice"] == r["gold"] for r in scored)
    out = {"backend": a.backend, "ckpt": a.ckpt, "data": a.data, "n": len(scored),
           "skipped": len(rows) - len(scored),
           "accuracy": correct / len(rows), "load_seconds": round(load_s, 1),
           "answer_seconds": round(time.time() - start, 1), "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    return out


def compare(path_a, path_b):
    a, b = json.load(open(path_a)), json.load(open(path_b))
    rows_a = {r["id"]: r for r in a["rows"]}
    rows_b = {r["id"]: r for r in b["rows"]}
    shared = [i for i in rows_a if i in rows_b and "skipped" not in rows_a[i] and "skipped" not in rows_b[i]]
    agree = sum(rows_a[i]["choice"] == rows_b[i]["choice"] for i in shared)
    deltas = []
    for i in shared:
        pa = rows_a[i].get("probabilities") or {}
        pb = rows_b[i].get("probabilities") or {}
        for k in set(pa) | set(pb):
            deltas.append(abs(pa.get(k, 0.0) - pb.get(k, 0.0)))
    flips = [(i, rows_a[i]["gold"], rows_a[i]["choice"], rows_b[i]["choice"]) for i in shared
             if rows_a[i]["choice"] != rows_b[i]["choice"]]
    print(json.dumps({
        "n_shared": len(shared),
        "accuracy": {a["backend"]: a["accuracy"], b["backend"]: b["accuracy"]},
        "n_skipped": {a["backend"]: a.get("skipped"), b["backend"]: b.get("skipped")},
        "choice_agreement": agree / len(shared) if shared else None,
        "max_abs_prob_delta": max(deltas) if deltas else None,
        "load_seconds": {a["backend"]: a["load_seconds"], b["backend"]: b["load_seconds"]},
        "answer_seconds": {a["backend"]: a["answer_seconds"], b["backend"]: b["answer_seconds"]},
        "flips": [{"id": i, "gold": g, a["backend"]: ca, b["backend"]: cb} for i, g, ca, cb in flips[:20]],
    }, indent=2))


def main():
    a = parse_args()
    if a.compare:
        compare(*a.compare)
        return
    if a.backend == "unsloth":
        import unsloth  # noqa: F401  (patches transformers/peft; must come first)
    score(a)


if __name__ == "__main__":
    main()
