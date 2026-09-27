#!/usr/bin/env python
"""M1 trainer: pointer head (with the rejection candidate) + content forward-KL to the frozen base.

    L = CE(pointer over K options + garbage)  +  λ · mean_{t ∈ content} KL(teacher_t ‖ student_t)

The teacher is the same weights with the adapters disabled (`disabled_adapter()`), read on the
same rows, so the KL is the SDFT anchor: it keeps the student's next-token distribution on
state/instruction text at the frozen base's. Logits for the KL are computed from the gathered
hidden states in chunks (the 262k vocabulary never materialises as one tensor), and the
softcap/tied `lm_head` path is reused.

Rows come from data/prep_dataset.py (`datasets` directory); one row = one question.

    /home/penhfel/unsloth_uv/bin/python training/train_distill.py \
        --dataset /mnt/f/distill_jev_runs/prep/dv7-smoke --max-steps 500 \
        --out-dir /mnt/f/distill_jev_runs/smoke --kl-weight 0.3
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unsloth  # noqa: F401  (must precede transformers/peft/trl)

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import Trainer, TrainerCallback, TrainingArguments

from data.config import DEFAULTS, DELIMITERS, MODEL
from model.delimiters import apply_delimiter_deltas
from model.head import PointerHead
from model.load import language_model

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="a prepared dataset directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl-weight", type=float, default=DEFAULTS["kl_weight"])
    ap.add_argument("--teacher-temperature", type=float, default=DEFAULTS["teacher_temperature"])
    ap.add_argument("--max-seq-length", type=int, default=DEFAULTS["max_seq_length"])
    ap.add_argument("--lora-r", type=int, default=DEFAULTS["lora_r"])
    ap.add_argument("--lora-alpha", type=int, default=DEFAULTS["lora_alpha"])
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--garbage", type=int, choices=[0, 1], default=int(DEFAULTS["garbage"]))
    ap.add_argument("--garbage-bias", type=float, default=DEFAULTS["garbage_bias"])
    ap.add_argument("--kl-chunk", type=int, default=128, help="positions per logits chunk for the KL")
    ap.add_argument("--teacher", choices=["cache", "live"], default="cache",
                    help="cache = offline top-k targets (one forward per step, GC-compatible); live = frozen-base forward (needs no GC)")
    ap.add_argument("--teacher-cache", default="", help="path to the teacher cache (with or without .npz)")
    ap.add_argument("--top-k", type=int, default=32, help="top-k support (live teacher)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N rows")
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--save-steps", type=int, default=500, help="write adapter + head.pt every N steps (0 = only at the end)")
    ap.add_argument("--resume", default="", help="checkpoint directory to resume from")
    ap.add_argument("--init-from", default="", help="checkpoint dir (adapter/ + head.pt) to warm start a delta run from")
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gc-mode", choices=["unsloth", "none"], default="unsloth",
                    help="unsloth = its offloading checkpointer (enabled at load); none = no checkpointing")
    a = ap.parse_args()
    a.no_gc = a.gc_mode == "none"
    a.out_dir = Path(a.out_dir)
    return a


class HeadCheckpointCallback(TrainerCallback):
    """Trainer saves the adapter at each checkpoint; this writes head.pt (head + delimiter
    deltas) beside it, so a long run is reconstructible from any checkpoint."""

    def __init__(self, head, deltas, meta):
        self.head, self.deltas, self.meta = head, deltas, meta

    def on_save(self, args, state, control, **kwargs):
        step_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"head": self.head.state_dict(), "temperature": 1.0, "garbage": self.head.garbage,
                    "deltas": {k: v.detach().cpu() for k, v in self.deltas.items()}, "meta": self.meta},
                   step_dir / "head.pt")


class DistillCollator:
    """Pads input ids/masks, flattens the per-row readout positions and targets."""

    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, features):
        B = len(features)
        L = max(len(f["input_ids"]) for f in features)
        input_ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attention = torch.zeros(B, L, dtype=torch.long)
        content = torch.zeros(B, L, dtype=torch.long)
        for i, f in enumerate(features):
            n = len(f["input_ids"])
            input_ids[i, :n] = torch.tensor(f["input_ids"], dtype=torch.long)
            attention[i, :n] = 1
            content[i, :n] = torch.tensor(f["content_mask"], dtype=torch.long)
        opt_pos = [p for f in features for p in f["opt_pos"]]
        targets = [t for f in features for t in f["target"]]
        n_opts = torch.tensor([f["n_options"] for f in features], dtype=torch.long)
        opt_start = torch.tensor([sum(f["n_options"] for f in features[:i]) for i in range(B)], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention, "content_mask": content,
                "row_index": torch.tensor([int(f.get("row_index", 0)) for f in features], dtype=torch.long),
                "decide_pos": torch.tensor([f["decide_pos"] for f in features], dtype=torch.long),
                "opt_pos": torch.tensor(opt_pos, dtype=torch.long), "target": torch.tensor(targets, dtype=torch.float32),
                "n_opts": n_opts, "opt_start": opt_start}


class DistillTrainer(Trainer):
    kl_weight: float = DEFAULTS["kl_weight"]
    teacher_temperature: float = 1.0
    kl_chunk: int = 128
    teacher_ids = None        # [P, k] long, cached teacher top-k token ids
    teacher_vals = None       # [P, k] float, cached teacher top-k logits
    teacher_offsets = None    # [N+1] int64, row i owns offsets[i]:offsets[i+1]

    def content_kl(self, model, hidden, teacher_hidden, content_mask):
        """mean forward-KL(teacher ‖ student) over the masked content positions, chunked."""
        positions = content_mask.bool().nonzero(as_tuple=False)   # [P, 2] (row, token)
        if len(positions) == 0:
            return hidden.new_zeros(())
        weight = model.get_output_embeddings().weight
        softcap = getattr(getattr(model.config, "text_config", model.config), "final_logit_softcapping", None)
        total, count = 0.0, 0
        for start in range(0, len(positions), self.kl_chunk):
            index = positions[start:start + self.kl_chunk]
            s = hidden[index[:, 0], index[:, 1]].to(weight.dtype)
            t = teacher_hidden[index[:, 0], index[:, 1]].to(weight.dtype)
            zs = (s @ weight.T).float()
            with torch.no_grad():
                zt = (t @ weight.T).float()
                if softcap:
                    zt = torch.tanh(zt / softcap) * softcap
                p_t = F.softmax(zt / self.teacher_temperature, -1)
            if softcap:
                zs = torch.tanh(zs / softcap) * softcap
            logp_s = F.log_softmax(zs, -1)
            total = total + (-(p_t * logp_s).sum(-1)).sum()
            count += len(index)
            del zs, zt, p_t, logp_s, s, t
        return total / count

    def content_kl_cached(self, model, hidden, inputs):
        """Forward-KL against the offline teacher top-k, one forward per step (3.4/D2)."""
        weight = model.get_output_embeddings().weight
        softcap = getattr(getattr(model.config, "text_config", model.config), "final_logit_softcapping", None)
        device = hidden.device
        total, count = 0.0, 0
        for i in range(hidden.shape[0]):
            r = int(inputs["row_index"][i])
            positions = inputs["content_mask"][i].bool().nonzero(as_tuple=False).flatten()
            start, end = int(self.teacher_offsets[r]), int(self.teacher_offsets[r + 1])
            if end - start != len(positions):
                raise ValueError(f"teacher cache mismatch for row {r}: {end - start} cached vs {len(positions)} content positions")
            for c in range(0, len(positions), self.kl_chunk):
                pos = positions[c:c + self.kl_chunk]
                t_ids = self.teacher_ids[start + c:start + c + len(pos)].to(device)
                t_vals = self.teacher_vals[start + c:start + c + len(pos)].to(device)
                s = (hidden[i, pos].to(weight.dtype) @ weight.T).float()
                if softcap:
                    s = torch.tanh(s / softcap) * softcap
                s_lse = torch.logsumexp(s, -1)
                s_val = s.gather(-1, t_ids)
                p = torch.softmax(t_vals / self.teacher_temperature, -1)
                total = total + (-(p * (s_val - s_lse[:, None])).sum(-1)).sum()
                count += len(pos)
                del s, s_lse, s_val, p
        return total / max(count, 1)

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        input_ids, attention = inputs["input_ids"], inputs["attention_mask"]
        out = model(input_ids=input_ids, attention_mask=attention, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[-1]                     # [B, L, d] bf16, with grad

        pointer_terms, garbage_mass, answerable_mass = [], [], []
        for i in range(hidden.shape[0]):
            start, n = int(inputs["opt_start"][i]), int(inputs["n_opts"][i])
            # the row's options can include a garbage entry (n_options already excludes nothing;
            # the target has n+1 entries and the head returns n+1 logits when garbage is on)
            option_positions = inputs["opt_pos"][start:start + n]
            option_hidden = hidden[i, option_positions]
            decide = hidden[i, int(inputs["decide_pos"][i])]
            logits = model.head(decide.float(), option_hidden.float())
            target = inputs["target"][start:start + n + 1]
            pointer_terms.append(-(target * F.log_softmax(logits.float(), -1)).sum())
            with torch.no_grad():
                if model.head.garbage:
                    (garbage_mass if float(target[-1]) > 0.5 else answerable_mass).append(float(torch.softmax(logits.float(), -1)[-1]))
        pointer = torch.stack(pointer_terms).mean()

        if self.teacher_ids is not None:
            kl = self.content_kl_cached(model, hidden, inputs)
        else:
            with torch.no_grad(), model.disable_adapter():
                teacher = model(input_ids=input_ids, attention_mask=attention, output_hidden_states=True, use_cache=False)
            kl = self.content_kl(model, hidden, teacher.hidden_states[-1], inputs["content_mask"])
        loss = pointer + self.kl_weight * kl
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        self._last = {"pointer": float(pointer.detach()), "kl": float(kl.detach()),
                      "p_garbage_on_garbage": mean(garbage_mass), "p_garbage_on_answerable": mean(answerable_mass),
                      "garbage_rows": len(garbage_mass), "answerable_rows": len(answerable_mass)}
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss")
        if return_outputs:
            return loss, {"pointer": pointer.detach(), "kl": kl.detach()}
        return loss

    def log(self, logs, start_time=None):
        last = getattr(self, "_last", None)
        if last:
            logs.update({k: round(v, 5) if isinstance(v, float) else v for k, v in last.items()})
        try:
            super().log(logs, start_time)
        except TypeError:
            super().log(logs)


def build_model(a):
    from unsloth import FastModel

    gc = False if getattr(a, "no_gc", False) else getattr(a, "gc_mode", "unsloth")
    model, processor = FastModel.from_pretrained(
        model_name=a.model, max_seq_length=a.max_seq_length, dtype=torch.bfloat16,
        load_in_4bit=True, use_gradient_checkpointing=gc, trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.add_special_tokens({"additional_special_tokens": DELIMITERS})
    delimiter_ids = [tokenizer.convert_tokens_to_ids(t) for t in DELIMITERS]
    deltas = apply_delimiter_deltas(model, language_model, delimiter_ids)
    model = FastModel.get_peft_model(
        model, r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, target_modules=TARGET_MODULES,
        use_gradient_checkpointing=gc, random_state=a.seed)
    for delta in deltas.values():   # prepare_model_for_kbit_training freezes what it does not know
        delta.requires_grad_(True)
    if gc:
        # reentrant checkpointing only recomputes (and frees activations) when the segment input
        # requires grad; the frozen embedding tables would otherwise make GC a silent no-op
        model.enable_input_require_grads()
    hidden_size = getattr(getattr(model.config, "text_config", model.config), "hidden_size")
    if getattr(a, "init_from", ""):
        # delta mode: load the released checkpoint's adapter as a second adapter and activate it,
        # then its pointer head and delimiter deltas (the same rule as kev.train --init_from)
        init_dir = Path(a.init_from)
        saved = torch.load(init_dir / "head.pt", map_location="cpu", weights_only=False)
        # drop the random adapter get_peft_model created and load the checkpoint as the only
        # adapter, keeping the standard name so the final save is a single clean adapter dir
        model.delete_adapter("default")
        model.load_adapter(str(init_dir / "adapter"), adapter_name="default", is_trainable=True)
        model.set_adapter("default")                       # activates it and applies requires_grad
        model.set_requires_grad("default", requires_grad=True)   # load_adapter alone loads it frozen
        for name, tensor in saved["deltas"].items():
            deltas[name].data.copy_(tensor.to(deltas[name].device))
        head = PointerHead(hidden_size, dp=saved["head"]["q.weight"].shape[0],
                           garbage=bool(saved.get("garbage", a.garbage)), garbage_bias=a.garbage_bias)
        head.load_state_dict(saved["head"])
        print(f"[train] warm start from {init_dir}: adapter + head + deltas loaded", flush=True)
    else:
        head = PointerHead(hidden_size, dp=a.head_dim, garbage=bool(a.garbage), garbage_bias=a.garbage_bias)
    model.head = head.to(next(model.parameters()).device)
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable {trainable / 1e6:.1f}M (head {sum(p.numel() for p in head.parameters()) / 1e3:.1f}k, "
          f"deltas {sum(d.numel() for d in deltas.values()) / 1e3:.1f}k, garbage={head.garbage})", flush=True)
    return model, tokenizer, head, deltas


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    model, tokenizer, head, deltas = build_model(a)

    dataset = load_from_disk(a.dataset)
    if a.limit:
        dataset = dataset.select(range(min(a.limit, len(dataset))))
    teacher_cache = None
    if a.teacher == "cache":
        if not a.teacher_cache:
            raise SystemExit("--teacher-cache is required with --teacher cache (build it with data/cache_teacher_logits.py)")
        import numpy as np
        path = a.teacher_cache if a.teacher_cache.endswith(".npz") else a.teacher_cache + ".npz"
        cached = np.load(path)
        teacher_cache = (torch.from_numpy(cached["topk_ids"]).long(), torch.from_numpy(cached["topk_logits"]).float(), cached["offsets"])
        dataset = dataset.add_column("row_index", list(range(len(dataset))))
        print(f"[train] teacher cache {path}: {teacher_cache[0].shape[0]} positions, k={teacher_cache[0].shape[1]}", flush=True)
    print(f"[train] {len(dataset)} rows", flush=True)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(a.out_dir / "adapter"), per_device_train_batch_size=a.per_device_batch_size,
        gradient_accumulation_steps=a.grad_accum, max_steps=a.max_steps, learning_rate=a.lr,
        warmup_steps=a.warmup_steps, logging_steps=a.logging_steps,
        save_strategy="steps" if a.save_steps else "no", save_steps=a.save_steps or 10**9, save_total_limit=2, bf16=True,
        optim="adamw_8bit", weight_decay=0.0, report_to="none", remove_unused_columns=False,
        dataloader_num_workers=0, seed=a.seed, gradient_checkpointing=False, lr_scheduler_type="cosine")
    trainer = DistillTrainer(model=model, args=args, train_dataset=dataset, data_collator=DistillCollator(),
                             processing_class=tokenizer,
                             callbacks=[HeadCheckpointCallback(head, deltas, {"model": a.model, "lora_r": a.lora_r,
                                                                              "head_dim": a.head_dim, "kl_weight": a.kl_weight})])
    trainer.kl_weight = a.kl_weight
    trainer.teacher_temperature = a.teacher_temperature
    trainer.kl_chunk = a.kl_chunk
    if teacher_cache is not None:
        trainer.teacher_ids, trainer.teacher_vals, trainer.teacher_offsets = teacher_cache

    start = time.time()
    trainer.train(resume_from_checkpoint=a.resume or None)
    wall = time.time() - start
    trainer.save_model(str(a.out_dir / "adapter"))
    tokenizer.save_pretrained(str(a.out_dir / "adapter"))
    torch.save({"head": head.state_dict(), "temperature": 1.0, "garbage": head.garbage,
                "deltas": {k: v.detach().cpu() for k, v in deltas.items()},
                "meta": {"model": a.model, "lora_r": a.lora_r, "lora_alpha": a.lora_alpha, "head_dim": a.head_dim,
                         "kl_weight": a.kl_weight, "teacher_temperature": a.teacher_temperature, "steps": a.max_steps}},
               a.out_dir / "head.pt")
    (a.out_dir / "train_config.json").write_text(json.dumps(vars(a), indent=2, default=str), encoding="utf-8")
    (a.out_dir / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
    print(f"[train] done in {wall / 60:.1f} min -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
