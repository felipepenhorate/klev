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

try:                       # unsloth patches transformers/peft at import, so it has to come first
    import unsloth  # noqa: F401
except ModuleNotFoundError:  # ...but predictions do not need it (--backend plain, docs/m15)
    pass

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from data import metrics
from data import config
from data.suites import write_json
from model.delimiters import apply_delimiter_deltas
from model.head import PointerHead
from model.load import compute_dtype, language_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="checkpoint dir with adapter/ and head.pt")
    ap.add_argument("--dataset", required=True, help="prepared rows (data/prep_dataset.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None,
                    help="override the preset's model path (unset = follow --preset)")
    ap.add_argument("--preset", default="",
                    help="base-model preset; sets the model and its delimiters together. One of: "
                         + ", ".join(sorted(config.PRESETS)) + ". Overrides KLEV_PRESET.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4, help="rows per forward")
    ap.add_argument("--backend", default="unsloth", choices=["unsloth", "plain"],
                    help="base loader: unsloth FastModel (default) or plain transformers+bitsandbytes "
                         "(same checkpoint, same readout; see docs/m15-plain-inference.md)")
    ap.add_argument("--extra-adapter", default="", help="third-party LoRA dir to combine with the run adapter")
    ap.add_argument("--extra-weight", type=float, default=1.0, help="weight for --extra-adapter (run weight is 1)")
    return ap.parse_args()


def load_checkpoint(run: Path, base: str, device="cuda", extra: str = "", extra_weight: float = 1.0):
    """The base dtype must match the one the checkpoint was trained in: the readout is a
    dot product on raw hidden states, so a bf16/fp16 split between train and eval shows up
    directly as a shift in the option logits."""
    from peft import PeftModel
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name=base, max_seq_length=2048, dtype=compute_dtype(), load_in_4bit=True,
        use_gradient_checkpointing=False, trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.add_special_tokens({"additional_special_tokens": config.DELIMITERS})
    delimiter_ids = [tokenizer.convert_tokens_to_ids(t) for t in config.DELIMITERS]
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


def score_dataset(model, head, dataset, device, progress_every=250):
    """Pointer readout over prepared rows -> (rows, seconds). One implementation, shared with
    eval/run_card_suites.py, so a card table can never be scored by a different rule than a
    single-suite run."""
    rows, start = [], time.time()
    with torch.no_grad():
        for i, row in enumerate(dataset):
            ids = torch.tensor([row["input_ids"]], device=device)
            # logits_to_keep=1: the readout works entirely off hidden_states[-1], but without
            # it transformers projects every position through lm_head -- a throwaway
            # vocab-projection over the whole sequence per row.
            hidden = model(input_ids=ids, output_hidden_states=True, use_cache=False,
                           logits_to_keep=1).hidden_states[-1][0]
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
            if progress_every and (i + 1) % progress_every == 0:
                print(f"[eval] {i + 1}/{len(dataset)} rows, {(time.time() - start) / 60:.1f} min", flush=True)
    return rows, time.time() - start


def load_plain(run: Path, base: str, device="cuda", extra: str = "", extra_weight: float = 1.0):
    """The published checkpoint through plain transformers + bitsandbytes, no unsloth
    (model/infer.py's load path). The extra-adapter composition stays unsloth-only: it is a
    research path, not part of serving."""
    if extra:
        raise SystemExit("--extra-adapter needs the unsloth backend (adapter composition is a research path)")
    from model.infer import load_klev

    model, tokenizer, head = load_klev(str(run), base=base or None, backend="plain")
    return model, tokenizer, head


def main():
    a = parse_args()
    config.apply_preset(a.preset)
    a.model = a.model or config.MODEL
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = load_from_disk(a.dataset)
    if a.limit:
        dataset = dataset.select(range(min(a.limit, len(dataset))))
    loader = load_checkpoint if a.backend == "unsloth" else load_plain
    model, tokenizer, head = loader(run, a.model, extra=a.extra_adapter, extra_weight=a.extra_weight)
    device = next(model.parameters()).device
    rows, seconds = score_dataset(model, head, dataset, device)
    report = {"n": len(rows), "run": str(run), "dataset": str(a.dataset), "backend": a.backend,
              "metrics": metrics.metrics(rows),
              "by_type": metrics.grouped_metrics(rows, "type"), "by_source": metrics.grouped_metrics(rows, "source"),
              "mean_none": sum(r["none"] for r in rows) / len(rows), "seconds": round(seconds, 1)}
    write_json(out / "rows.json", rows)
    write_json(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("by_type", "by_source")}, indent=2))
    print(f"[eval] {len(rows)} rows, {seconds / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
