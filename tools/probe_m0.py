#!/usr/bin/env python
"""M0 probe: validate the plumbing distill_jev depends on, on the local RTX 4080.

Checks (each one wrapped, the probe reports all of them even if one fails):
  1. load Gemma 4 E4B 4-bit with Unsloth (FastModel) and report VRAM
  2. add the five decision delimiters and resize the embeddings
  3. attach QLoRA and print the trainable parameter count
  4. text forward: hidden states at known delimiter positions + logits_to_keep
  5. pointer-head readout + backward; LoRA parameters receive gradients
  6. disable_adapter() gives the frozen base distribution (teacher path)
  7. image forward through the processor: soft-token span, hidden states, mm_token_type_ids
  8. peak VRAM

Run:  /home/penhfel/unsloth_uv/bin/python tools/probe_m0.py
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unsloth  # noqa: F401  (must be imported before transformers/peft/trl: it patches them)

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from data.config import DECIDE, DELIMITERS, MODEL, OPTION, OPTION_END, QUESTION, STATE

RESULTS = {}
PEAKS = {}


def record(name, ok, detail=None):
    RESULTS[name] = {"ok": bool(ok), "detail": detail}
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail if detail is not None else ''}", flush=True)


def memory(tag):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[mem] {tag}: allocated {allocated:.2f} GB, reserved {reserved:.2f} GB, peak {peak:.2f} GB, free {free / 1e9:.2f} GB", flush=True)
    return {"allocated_gb": round(allocated, 3), "reserved_gb": round(reserved, 3), "peak_gb": round(peak, 3), "free_gb": round(free / 1e9, 3)}


def peak_since(tag):
    """Peak allocated since the last reset (find which section allocates the spike)."""
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[peak] {tag}: {peak:.2f} GB", flush=True)
    return round(peak, 3)


def delimiter_positions(ids, tokenizer):
    """The token index of each delimiter occurrence, in order, by id."""
    wanted = {name: tokenizer.convert_tokens_to_ids(tok) for name, tok in
              zip(("state", "q", "opt", "opt_end", "decide"), DELIMITERS)}
    found = {name: [] for name in wanted}
    for i, token_id in enumerate(ids):
        for name, wanted_id in wanted.items():
            if token_id == wanted_id:
                found[name].append(i)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--no-resize", action="store_true", help="skip adding the delimiter tokens")
    ap.add_argument("--no-lora", action="store_true", help="skip QLoRA")
    ap.add_argument("--out", default=str(ROOT / "runs" / "probe_m0.json"))
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"[env] torch {torch.__version__} cuda {torch.version.cuda} device {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[env] model {args.model}", flush=True)
    record("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))

    t0 = time.time()
    from unsloth import FastModel
    model, processor = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        use_gradient_checkpointing=False,
        trust_remote_code=False,
    )
    # multimodal loads return the processor; the tokenizer lives inside it
    tokenizer = getattr(processor, "tokenizer", processor)
    torch.cuda.reset_peak_memory_stats()
    print(f"[load] {time.time() - t0:.1f} s; model {type(model).__name__}; processor {type(processor).__name__}", flush=True)
    record("load", True, f"{type(model).__name__} in {time.time() - t0:.1f}s")
    record("mem_after_load", True, memory("after load"))
    print(f"[model] config {type(model.config).__name__} model_type {getattr(model.config, 'model_type', None)}", flush=True)
    print(f"[model] top-level children: {[(n, type(m).__name__) for n, m in model.named_children()]}", flush=True)
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    record("language_model_path", language_model is not None,
           type(language_model).__name__ if language_model is not None else "model.model.language_model missing; see children log")

    record("processor", type(processor).__name__ == "Gemma4Processor", type(processor).__name__)

    before_vocab = len(tokenizer)
    added = tokenizer.add_special_tokens({"additional_special_tokens": DELIMITERS})
    delimiter_ids = [tokenizer.convert_tokens_to_ids(t) for t in DELIMITERS]
    PEAKS["delimiters"] = peak_since("after delimiters")
    record("delimiters", added == len(DELIMITERS) and len(tokenizer) == before_vocab and len(set(delimiter_ids)) == len(DELIMITERS),
           f"registered {added}, vocab unchanged at {len(tokenizer)}, ids {delimiter_ids}")
    memory("after delimiters")

    from model.delimiters import apply_delimiter_deltas
    from model.load import language_model

    delimiter_deltas = apply_delimiter_deltas(model, language_model, delimiter_ids)
    record("delimiter_deltas", all(d.requires_grad for d in delimiter_deltas.values()),
           {k: tuple(v.shape) for k, v in delimiter_deltas.items()})

    if not args.no_lora:
        model = FastModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing=False,
            random_state=0,
        )
        for delta in delimiter_deltas.values():   # prepare_model_for_kbit_training freezes everything it does not know
            delta.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        record("qlora", True, f"{trainable / 1e6:.1f}M trainable")
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"[lora] first trainable names: {names[:6]}", flush=True)
        print(f"[lora] delimiter rows: {[n for n in names if '.delta' in n][:6]}", flush=True)
        print(f"[lora] input embeddings: {type(model.get_input_embeddings()).__name__}", flush=True)
    else:
        record("qlora", True, "skipped")

    head_dim = getattr(getattr(model.config, "text_config", model.config), "hidden_size", None)
    record("hidden_size", head_dim is not None, head_dim)
    device = "cuda"

    # --- 4/5. text forward + pointer head backward ---------------------------------------------------------------
    keep = None
    try:
        prompt = (f"{STATE}The sky is blue and the grass is green. "
                  f"{QUESTION}What colour is the sky?{OPTION}blue{OPTION_END}{OPTION}green{OPTION_END}{DECIDE}")
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        positions = delimiter_positions(ids, tokenizer)
        record("encode_delimiters", bool(positions["decide"]) and len(positions["opt_end"]) == 2,
               {k: v for k, v in positions.items()})
        input_ids = torch.tensor([ids], device=device)
        out = model(input_ids=input_ids, output_hidden_states=True)
        hidden = out.hidden_states[-1][0]
        record("text_hidden", tuple(hidden.shape) == (len(ids), head_dim), f"hidden {tuple(hidden.shape)}")
        # logits_to_keep with an index tensor: should return logits only at the requested positions
        keep = torch.tensor(positions["opt_end"] + positions["decide"], device=device)
        out2 = model(input_ids=input_ids, logits_to_keep=keep)
        record("logits_to_keep", out2.logits.shape[1] == len(keep), f"logits {tuple(out2.logits.shape)}")
        PEAKS["text_forward"] = peak_since("after text forward")
        del out, out2
        torch.cuda.empty_cache()

        from model.head import PointerHead

        torch.manual_seed(0)
        head = PointerHead(head_dim, 256).to(device)
        model.train()
        out = model(input_ids=input_ids, output_hidden_states=True)
        h = out.hidden_states[-1][0].float()
        decide = h[positions["decide"][0]]
        opts = torch.stack([h[i] for i in positions["opt_end"]])
        z = head(decide, opts)
        loss = F.cross_entropy(z.unsqueeze(0), torch.tensor([0], device=device))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] + list(head.parameters()), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        lora_grads = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
        head_grads = [n for n, p in head.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        head_all = dict(head.named_parameters())
        missing = [n for n, p in head_all.items() if p.grad is None or p.grad.abs().sum() == 0]
        record("head_step", len(head_grads) > 0 and len(lora_grads) > 0,
               f"loss {loss.item():.4f}; head grads {len(head_grads)}/{len(head_all)}"
               f"{f' (zero: {missing})' if missing else ''}; lora grad tensors {len(lora_grads)}")
        PEAKS["backward"] = peak_since("after backward")
        optimizer.step()
        del out, h, decide, opts, z, loss
        torch.cuda.empty_cache()
    except Exception:
        record("head_step", False, traceback.format_exc().splitlines()[-1])

    # --- 6. disable_adapter gives the frozen base -----------------------------------------------------------------
    try:
        if keep is None:
            raise RuntimeError("text forward did not reach logits_to_keep")
        with torch.no_grad():
            student = model(input_ids=input_ids, logits_to_keep=keep).logits.float()
            with model.disable_adapter():
                teacher = model(input_ids=input_ids, logits_to_keep=keep).logits.float()
        diff = (student - teacher).abs().max().item()
        record("disable_adapter", diff > 0, f"max |student - teacher| = {diff:.6f} (must be > 0 after a step)")
    except Exception:
        record("disable_adapter", False, traceback.format_exc().splitlines()[-1])

    # --- 7. image forward ------------------------------------------------------------------------------------------
    try:
        image = Image.new("RGB", (224, 224), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([20, 20, 100, 200], fill="steelblue")
        draw.rectangle([120, 80, 200, 200], fill="indianred")
        prompt = (f"{STATE}<|image|> The image shows two coloured bars. "
                  f"{QUESTION}Which bar is taller?{OPTION}left{OPTION_END}{OPTION}right{OPTION_END}{DECIDE}")
        proc = processor(text=[prompt], images=[image], return_tensors="pt", return_mm_token_type_ids=True)
        record("image_processor_keys", True, {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in proc.items()})
        input_ids_i = proc["input_ids"].to(device)
        boi = tokenizer.boi_token_id if hasattr(tokenizer, "boi_token_id") else tokenizer.convert_tokens_to_ids("<|image>")
        eoi = tokenizer.eoi_token_id if hasattr(tokenizer, "eoi_token_id") else tokenizer.convert_tokens_to_ids("<image|>")
        image_token = tokenizer.image_token_id
        row = input_ids_i[0].tolist()
        span = (row.index(boi), len(row) - 1 - row[::-1].index(eoi))
        n_soft = sum(1 for token_id in row[span[0] + 1:span[1]] if token_id == image_token)
        record("image_span", n_soft > 0, f"boi {boi} eoi {eoi} soft tokens {n_soft} span {span} of {len(row)}")
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in proc.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden_i = out.hidden_states[-1][0]
        record("image_forward", hidden_i.shape[0] == len(row), f"hidden {tuple(hidden_i.shape)} logits {tuple(out.logits.shape)}")
        # pointer positions after image expansion: find the delimiters in the expanded ids
        positions_i = delimiter_positions(row, tokenizer)
        record("image_delimiters", bool(positions_i["decide"]) and len(positions_i["opt_end"]) == 2,
               {k: v for k, v in positions_i.items()})
        PEAKS["image_forward"] = peak_since("after image forward")
        del out, inputs, hidden_i
        torch.cuda.empty_cache()
    except Exception:
        record("image_forward", False, traceback.format_exc().splitlines()[-1])

    # --- 8. garbage candidate on real hidden states: can the head route "none of these"? ----------------
    try:
        from model.head import PointerHead as GarbageHead

        def features(row_text):
            row_ids = tokenizer(row_text, add_special_tokens=False)["input_ids"]
            row_ids = [tokenizer.bos_token_id] + row_ids
            ids_t = torch.tensor([row_ids], device=device)
            with torch.no_grad():
                h_row = model(input_ids=ids_t, output_hidden_states=True).hidden_states[-1][0].float()
            pos = delimiter_positions(row_ids, tokenizer)
            return h_row, pos, row_ids

        state = "The sky is blue and the grass is green."
        knowable = (f"{STATE}{state} {QUESTION}What colour is the sky?"
                    f"{OPTION}blue{OPTION_END}{OPTION}green{OPTION_END}{DECIDE}")
        nonsense = (f"{STATE}{state} {QUESTION}What colour is the sky?"
                    f"{OPTION}purple{OPTION_END}{OPTION}orange{OPTION_END}{DECIDE}")
        hk, pk, _ = features(knowable)
        hn, pn, _ = features(nonsense)
        head_g = GarbageHead(head_dim, 256, garbage=True).to(device)
        with torch.no_grad():
            p_before = torch.softmax(head_g(hk[pk["decide"][0]], torch.stack([hk[i] for i in pk["opt_end"]])), -1)
        optimizer_g = torch.optim.AdamW(head_g.parameters(), lr=1e-2)

        def garbage_logits(h_row, pos):
            return head_g(h_row[pos["decide"][0]], torch.stack([h_row[i] for i in pos["opt_end"]]))

        grads = None
        for step in range(300):
            z_k, z_n = garbage_logits(hk, pk), garbage_logits(hn, pn)
            target_k = torch.tensor([1.0, 0.0, 0.0], device=device)   # option 0 ("blue"), no garbage
            target_n = torch.tensor([0.0, 0.0, 1.0], device=device)   # garbage
            loss_g = (-(target_k * F.log_softmax(z_k, -1)).sum() - (target_n * F.log_softmax(z_n, -1)).sum()) / 2
            optimizer_g.zero_grad(); loss_g.backward()
            if step == 0:
                grads = {n: float(p.grad.abs().sum()) for n, p in head_g.named_parameters() if p.grad is not None}
            optimizer_g.step()
        with torch.no_grad():
            p_k = torch.softmax(garbage_logits(hk, pk), -1)
            p_n = torch.softmax(garbage_logits(hn, pn), -1)
        record("garbage_head", float(p_k[0]) > 0.5 and float(p_k[2]) < 0.1 and float(p_n[2]) > 0.5
               and grads.get("g", 0) > 0 and grads.get("g_bias", 0) > 0,
               {"p_before_garbage": round(float(p_before[2]), 4),
                "knowable_after": [round(float(x), 4) for x in p_k], "nonsense_after": [round(float(x), 4) for x in p_n],
                "grads": {k: round(v, 4) for k, v in grads.items()}})
    except Exception:
        record("garbage_head", False, traceback.format_exc().splitlines()[-1])

    peak = torch.cuda.max_memory_allocated() / 1e9
    record("peak_vram", peak < 15.5, f"{peak:.2f} GB allocated peak")
    memory("after probe")

    Path(args.out).write_text(json.dumps({"checks": RESULTS, "peaks": PEAKS}, indent=2), encoding="utf-8")
    failed = [name for name, result in RESULTS.items() if not result["ok"]]
    print(f"\n[probe] results -> {args.out}", flush=True)
    print(f"[probe] {'ALL CHECKS PASSED' if not failed else 'FAILED: ' + ', '.join(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
