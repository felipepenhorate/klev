#!/usr/bin/env python
"""Answer a SystemOne request inline -- no server, no HTTP, no daemon.

klev is a library first: load it once, call it as many times as you like in the same process.
This is the whole thing.

    # with the built-in demo request
    python examples/system_one.py --ckpt lumierenoir/klev-0.8b

    # or feed it a request body
    python examples/system_one.py --ckpt lumierenoir/klev-0.8b --request my_request.json
    echo '{"state": "...", "questions": {...}}' | python examples/system_one.py --ckpt ...

Or from Python, which is the point:

    from model.infer import load_klev, answer
    model, tokenizer, head = load_klev("lumierenoir/klev-0.8b")
    result = answer(model, tokenizer, head, {
        "state": "My card was declined at a shop yesterday.",
        "questions": {
            "issue": {"type": "noul", "instructions": "Is this a card problem?",
                      "criteria": {"false": "not about the card", "true": "about the card"}},
            "intent": {"type": "choice", "instructions": "What should the agent do first?",
                       "criteria": {"card_block": "ask about a card block",
                                    "atm": "point them at the ATM network"}},
        }})
    print(result["answers"]["issue"]["noul"], result["answers"]["intent"]["choice"])

Each question is encoded into the same row layout the model was trained on and read out through
the pointer head, so this returns the same probabilities eval/eval_decisions.py scores -- the
default temperature is 1.0, the raw readout. Pass --temperature to serve a calibrated one.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO = {
    "state": ("Customer writes: my debit card was declined twice at a supermarket yesterday, "
              "and the ATM also refused it this morning. I only used it for the weekly shop."),
    "questions": {
        "channel_issue": {
            "type": "noul",
            "instructions": "Is the card being declined?",
            "criteria": {"false": "no, something else is the subject",
                         "true": "yes, the card is being declined"},
        },
        "next_step": {
            "type": "choice",
            "instructions": "What should the agent do first?",
            "criteria": {
                "card_usage": "ask how the card was used and whether it is a contactless card",
                "atm_fault": "raise an ATM fault with the network",
                "fraud_check": "run a fraud check on recent transactions",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["no action needed", "next working day", "same day", "immediate"],
        },
    },
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lumierenoir/klev-0.8b",
                    help="Hub repo id or a local dir with adapter/ and head.pt")
    ap.add_argument("--base", default=None, help="override the base model (defaults to the preset)")
    ap.add_argument("--preset", default="qwen35-08b", choices=["qwen35-08b", "qwen35-08b-base", "e4b", "e2b-qat"])
    ap.add_argument("--request", default="", help="JSON file with {state, questions}; - for stdin")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="1.0 is the raw readout the benchmarks score. The model card's served "
                         "numbers use 2.2974 (IT) / 2.2449 (-Base) from scripts/calibrate.py.")
    ap.add_argument("--max-seq", type=int, default=None)
    return ap.parse_args()


def main():
    a = parse_args()
    from model.infer import answer, load_klev

    if a.request == "-":
        request = json.load(sys.stdin)
    elif a.request:
        request = json.load(open(a.request, encoding="utf-8"))
    else:
        request = DEMO

    print(f"[klev] loading {a.ckpt} (preset {a.preset}) ...", flush=True)
    model, tokenizer, head = load_klev(a.ckpt, base=a.base, preset=a.preset)
    kwargs = {"temperature": a.temperature}
    if a.max_seq:
        kwargs["max_seq"] = a.max_seq
    result = answer(model, tokenizer, head, request, **kwargs)

    print(json.dumps(result, indent=2))
    print("\n[readback]")
    for qid, out in result["answers"].items():
        if out["type"] == "noul":
            verdict = "yes" if out["noul"] >= 0.5 else "no"
            print(f"  {qid:14s} noul    p(true)={out['noul']:.3f}  -> {verdict}   reject={out['none']:.3f}")
        elif out["type"] == "choice":
            print(f"  {qid:14s} choice  -> {out['choice']:22s} conf={out['confidence']:.3f}  "
                  f"reject={out['none']:.3f}")
        else:
            print(f"  {qid:14s} score   -> {out['score']:.2f}   reject={out['none']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
