#!/usr/bin/env python
"""Score a trained checkpoint on prepared rows (the M1 dev read; grows into the M6 eval).

Loads the base + saved adapter + head.pt (+ delimiter deltas), runs the pointer readout on
every prepared row and writes `rows.json` + `report.json` (accuracy, Brier, ECE, NLL,
coverage@5% error, AURC) with `data/metrics.py`.

    ./unsloth_uv/bin/python eval/eval_decisions.py \
        --run /mnt/f/distill_jev_runs/smoke --dataset /mnt/f/distill_jev_runs/prep/dv7-dev \
        --out /mnt/f/distill_jev_runs/smoke-eval
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unsloth  # noqa: F401

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from data import metrics
from data.config import DELIMITERS, MODEL
from data.suites import write_json
from model.delimiters import apply_delimiter_deltas
from model.head import PointerHead
from model.load import language_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="checkpoint dir with adapter/ and head.pt")
    ap.add_argument("--dataset", required=True, help="prepared rows (data/prep_dataset.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4, help="rows per forward")
    ap.add_argument("--extra-adapter", default="", help="third-party LoRA dir to combine with the run adapter")
    ap.add_argument("--extra-weight", type=float, default=1.0, help="weight for --extra-adapter (run weight is 1)")
    return ap.parse_args()


def load_checkpoint(run: Path, base: str, device="cuda", extra: str = "", extra_weight: float = 1.0):
    from peft import PeftModel
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name=base, max_seq_length=2048, dtype=torch.bfloat16, load_in_4bit=True,
        use_gradient_checkpointing=False, trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.add_special_tokens({"additional_special_tokens": DELIMITERS})
    delimiter_ids = [tokenizer.convert_tokens_to_ids(t) for t in DELIMITERS]
    deltas = apply_delimiter_deltas(model, language_model, delimiter_ids)
    model = PeftModel.from_pretrained(model, str(run / "adapter"))
    if extra:
        model.load_adapter(extra, adapter_name="ext", is_trainable=False)
        model.add_weighted_adapter(["default", "ext"], [1.0, extra_weight], "combo", combination_type="linear")
        model.set_adapter("combo")
        print(f"[eval] adapter composed: default + {extra_weight} * ext", flush=True)
    saved = torch.load(run / "head.pt", map_location="cpu", weights_only=False)
    for name, tensor in saved["deltas"].items():
        deltas[name].data.copy_(tensor.to(deltas[name].device))
    hidden_size = getattr(getattr(model.config, "text_config", model.config), "hidden_size")
    head = PointerHead(hidden_size, dp=saved["head"]["q.weight"].shape[0], garbage=bool(saved.get("garbage", False)))
    head.load_state_dict(saved["head"])
    model.head = head.to(device).eval()
    model.eval()
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    return model, tokenizer, head


def main():
    a = parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(a.dataset)
    if a.limit:
        dataset = dataset.select(range(min(a.limit, len(dataset))))
    model, tokenizer, head = load_checkpoint(run, a.model, extra=a.extra_adapter, extra_weight=a.extra_weight)
    device = next(model.parameters()).device
    weight = model.get_output_embeddings().weight
    softcap = getattr(getattr(model.config, "text_config", model.config), "final_logit_softcapping", None)

    rows, start = [], time.time()
    with torch.no_grad():
        for i, row in enumerate(dataset):
            ids = torch.tensor([row["input_ids"]], device=device)
            hidden = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[-1][0]
            option_hidden = hidden[torch.tensor(row["opt_pos"], device=device)]
            decide = hidden[row["decide_pos"]]
            logits = model.head(decide.float(), option_hidden.float())
            probs = torch.softmax(logits.float(), -1)
            n = row["n_options"]
            p_options = probs[:n] / probs[:n].sum().clamp_min(1e-9)   # renormalized over the options
            rows.append({"id": row["id"], "question": row["qid"], "group": row["record_id"], "task": row["qtype"],
                         "source": row["source"], "type": row["qtype"], "keys": row["keys"], "variant": "clean",
                         "label": int(row["label"]), "p": p_options.tolist(), "logits": logits[:n].tolist(),
                         "none": float(probs[n]) if head.garbage else 0.0, "inference_temperature": 1.0})
            if (i + 1) % 250 == 0:
                print(f"[eval] {i + 1}/{len(dataset)} rows, {(time.time() - start) / 60:.1f} min", flush=True)
    report = {"n": len(rows), "run": str(run), "dataset": str(a.dataset), "metrics": metrics.metrics(rows),
              "by_type": metrics.grouped_metrics(rows, "type"), "by_source": metrics.grouped_metrics(rows, "source"),
              "mean_none": sum(r["none"] for r in rows) / len(rows)}
    write_json(out / "rows.json", rows)
    write_json(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("by_type", "by_source")}, indent=2))
    print(f"[eval] {len(rows)} rows, {(time.time() - start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
