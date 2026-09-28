#!/usr/bin/env python
"""Train the external task LoRA that the few-shot stitch then imports.

Plain alpaca SFT (`SFTTrainer` on prompt+answer), NOT a klev delta: the stitch's premise is
that this adapter's knowledge lives only in its own prompt space, so it must be trained the way
a third party would -- a text LoRA over a task prompt, with no pointer head and no KL anchor.

The alpaca template and slot layout are the ones eval/eval_lora_stitch.py probes with
(`ALPACA`, Instruction/Input/Response), so training and probing see the same string.

    python training/train_ext_lora.py --base Qwen/Qwen3.5-0.8B \
        --train <rumour train.jsonl> --prompts <alpaca_train.json> \
        --out $KLEV_RUNS/ext-lora-rumour
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_BF16", "0")
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")

ALPACA = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True, help="klev labelled jsonl (for the label order)")
    ap.add_argument("--prompts", required=True, help='alpaca prompts json: conv_key -> {"prompt":...}')
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3407)
    return ap.parse_args()


def main():
    a = parse_args()
    import unsloth  # noqa: F401
    import torch
    from transformers import Trainer, TrainingArguments
    from unsloth import FastModel

    prompts = json.load(open(a.prompts, encoding="utf-8"))
    rows = [json.loads(line) for line in Path(a.train).read_text(encoding="utf-8").splitlines() if line.strip()]
    if a.limit:
        rows = rows[:a.limit]
    examples = []
    for i, r in enumerate(rows):
        key = r["_meta"]["id"].split("/")[-1]
        entry = prompts[key]
        # Take the record's single question rather than a hardcoded qid, so the same script
        # serves any builder (rumoureval uses "relationship", pubmedqa "answer").
        questions = r["questions"]
        if len(questions) != 1:
            raise SystemExit(f"{key}: expected exactly one question, got {sorted(questions)}")
        answer = next(iter(questions.values()))["label"]
        assert answer == entry["answer"], f"label/answer disagree for {key}: {answer!r} vs {entry['answer']!r}"
        examples.append({"text": ALPACA.format(entry["prompt"], "", answer) + "<|endoftext|>"})
    print(f"[ext] {len(examples)} examples from {a.train}", flush=True)

    model, processor = FastModel.from_pretrained(
        model_name=a.base, max_seq_length=a.max_seq, dtype=torch.float16,
        load_in_4bit=True, use_gradient_checkpointing="unsloth", trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = FastModel.get_peft_model(
        model, r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
        # The classic 7, matching klev's own TARGET_MODULES -- NOT kev's lora_targets="all".
        # Adding in_proj_* / out_proj (the gated-deltanet projections) makes peft emit fp32 into
        # the fla Triton kernel, which then needs 73728 B of LDS and dies on this card:
        #   OutOfResources: shared memory, Required: 73728, Hardware limit: 65536
        # (gfx1032 has 64 KB LDS). Leaving them bare keeps the deltanet in fp16, which is what
        # klev's 3,894-step run did successfully on this same box.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=a.seed)

    class SFTData(torch.utils.data.Dataset):
        def __len__(self):
            return len(tokenized)

        def __getitem__(self, i):
            return dict(tokenized[i])

    def collate(batch):
        width = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids, mask, labels = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad_n = width - n
            ids.append(b["input_ids"] + [pad] * pad_n)
            mask.append(b["attention_mask"] + [0] * pad_n)
            labels.append(b["labels"] + [-100] * pad_n)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask),
                "labels": torch.tensor(labels)}

    tokenized = []
    for ex in examples:
        enc = tokenizer(ex["text"], truncation=True, max_length=a.max_seq,
                        add_special_tokens=True)["input_ids"]
        enc = enc[: a.max_seq - 1] + [tokenizer.eos_token_id]
        tokenized.append({"input_ids": enc, "attention_mask": [1] * len(enc), "labels": list(enc)})

    args = TrainingArguments(
        output_dir=str(Path(a.out) / "trainer"), per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.accum, num_train_epochs=a.epochs, learning_rate=a.lr,
        fp16=True, bf16=False, logging_steps=5, optim="adamw_torch", weight_decay=0.01,
        lr_scheduler_type="linear", warmup_steps=max(1, int(0.03 * len(tokenized) / (a.batch * a.accum))),
        seed=a.seed, report_to="none", remove_unused_columns=False, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=SFTData(), data_collator=collate,
                      processing_class=tokenizer)
    stats = trainer.train()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    (out / "train_stats.json").write_text(json.dumps({
        "base": a.base, "examples": len(examples), "epochs": a.epochs, "lr": a.lr,
        "lora_r": a.lora_r, "lora_alpha": a.lora_alpha,
        "runtime_s": round(float(stats.metrics.get("train_runtime", 0.0)), 1),
        "train_loss": round(float(stats.training_loss), 4)}, indent=2), encoding="utf-8")
    print(f"[ext] done -> {out / 'adapter'}  loss={stats.training_loss:.4f}  {stats.metrics.get('train_runtime')}", flush=True)


if __name__ == "__main__":
    main()
