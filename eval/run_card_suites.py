#!/usr/bin/env python
"""Every suite behind the model cards, scored in one model load per backend.

The card tables are the claim; this is the run that backs it, and `--backend unsloth|plain`
scores the identical rows through the identical readout (eval/eval_decisions.py's
`score_dataset`) so the two columns can be compared rather than assumed. Loading the base is the
expensive part, so all suites share it.

    python eval/run_card_suites.py --backend unsloth --out /mnt/f/distill_jev_runs/card-unsloth
    python eval/run_card_suites.py --backend plain   --out /mnt/f/distill_jev_runs/card-plain
    python eval/run_card_suites.py --compare /mnt/f/distill_jev_runs/card-unsloth /mnt/f/distill_jev_runs/card-plain

`--compare` needs no GPU: it prints the markdown table used in the cards and the row-level
agreement between the two runs.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402

from data import config, metrics  # noqa: E402
from data.suites import write_json  # noqa: E402
from eval.eval_decisions import load_checkpoint, load_plain, score_dataset  # noqa: E402

PREP = Path("/mnt/f/distill_jev_runs/prep")

# name -> (prepared dir, what the card calls it)
SUITES = {
    "dv7-dev": (PREP / "dv7-dev", "decision-v7 dev"),
    "transfer-v4": (PREP / "transfer-v4-dev", "transfer-v4 dev"),
    "mmlu": (PREP / "bench-mmlu", "MMLU"),
    "arc": (PREP / "bench-arc", "ARC-Challenge"),
    "hellaswag": (PREP / "bench-hellaswag", "HellaSwag"),
    "stanceosaurus": (PREP / "bench-stanceosaurus", "Stanceosaurus ar/en/ru/es"),
    "costbr": (PREP / "bench-costbr-test", "CoSt-BR"),
    "ood-multilingual": (PREP / "bench-ood", "OOD multilingual"),
    "in-dist-multilingual": (PREP / "bench-multilingual", "in-distribution multilingual"),
}

SMOKE = {"dv7-dev", "transfer-v4", "mmlu", "arc", "hellaswag", "costbr"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/mnt/f/distill_jev_runs/main")
    ap.add_argument("--backend", default="unsloth", choices=["unsloth", "plain"])
    ap.add_argument("--preset", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suites", default="", help="comma-separated subset of " + ",".join(SUITES))
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    return ap.parse_args()


def run(a):
    if a.backend == "unsloth":
        import unsloth  # noqa: F401  (patches transformers/peft; must come first)
    config.apply_preset(a.preset)
    out = Path(a.out or f"/mnt/f/distill_jev_runs/card-{a.backend}")
    out.mkdir(parents=True, exist_ok=True)
    wanted = [s for s in (a.suites.split(",") if a.suites else SUITES) if s]
    loader = load_checkpoint if a.backend == "unsloth" else load_plain
    t0 = time.time()
    model, tokenizer, head = loader(Path(a.run), config.MODEL)
    print(f"[card] {a.backend}: base loaded in {time.time() - t0:.0f}s", flush=True)
    device = next(model.parameters()).device
    summary = {}
    for name in wanted:
        path, label = SUITES[name]
        dataset = load_from_disk(str(path))
        if a.limit:
            dataset = dataset.select(range(min(a.limit, len(dataset))))
        rows, seconds = score_dataset(model, head, dataset, device)
        report = {"n": len(rows), "backend": a.backend, "label": label, "dataset": str(path),
                  "metrics": metrics.metrics(rows), "by_source": metrics.grouped_metrics(rows, "source"),
                  "mean_none": sum(r["none"] for r in rows) / len(rows), "seconds": round(seconds, 1)}
        write_json(out / f"{name}.json", report)
        write_json(out / f"{name}-rows.json", rows)
        summary[name] = {"n": len(rows), "acc": report["metrics"]["acc"], "brier": report["metrics"]["brier"],
                         "seconds": report["seconds"]}
        print(f"[card] {name:22s} n={len(rows):5d} acc={report['metrics']['acc']:.4f} "
              f"({seconds / 60:.1f} min)", flush=True)
    write_json(out / "summary.json", {"backend": a.backend, "run": a.run, "suites": summary,
                                      "total_minutes": round((time.time() - t0) / 60, 1)})
    print(json.dumps(summary, indent=2))


def compare(dir_a, dir_b):
    a, b = Path(dir_a), Path(dir_b)
    sa = json.loads((a / "summary.json").read_text())
    sb = json.loads((b / "summary.json").read_text())
    print(f"backend A = {sa['backend']} ({sa['run']}), backend B = {sb['backend']} ({sb['run']})\n")
    print("| suite | n | A acc | B acc | delta | A Brier | B Brier | A min | B min |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in SUITES:
        if name not in sa["suites"] or name not in sb["suites"]:
            continue
        x, y = sa["suites"][name], sb["suites"][name]
        print(f"| {SUITES[name][1]} | {x['n']} | {x['acc']:.3f} | {y['acc']:.3f} | "
              f"{y['acc'] - x['acc']:+.3f} | {x['brier']:.3f} | {y['brier']:.3f} | "
              f"{x['seconds'] / 60:.1f} | {y['seconds'] / 60:.1f} |")
    print("\nrow-level agreement (per suite):")
    for name in SUITES:
        ra = a / f"{name}-rows.json"
        rb = b / f"{name}-rows.json"
        if not (ra.exists() and rb.exists()):
            continue
        rows_a = {r["id"]: r for r in json.loads(ra.read_text())}
        rows_b = {r["id"]: r for r in json.loads(rb.read_text())}
        shared = [i for i in rows_a if i in rows_b]
        if not shared:
            continue
        agree = sum(max(range(len(rows_a[i]["p"])), key=lambda k: rows_a[i]["p"][k]) ==
                    max(range(len(rows_b[i]["p"])), key=lambda k: rows_b[i]["p"][k]) for i in shared)
        delta = max(abs(x - y) for i in shared for x, y in zip(rows_a[i]["p"], rows_b[i]["p"]))
        print(f"  {name:22s} n={len(shared):5d} choice agreement {agree / len(shared):.4f}  "
              f"max |dp| {delta:.4f}")


def main():
    a = parse_args()
    if a.compare:
        compare(*a.compare)
        return
    run(a)


if __name__ == "__main__":
    main()
