#!/usr/bin/env bash
# klev on Qwen3.5-0.8B, end to end, on the RX 6600M (gfx1032). Chained so it survives
# tool/session timeouts.
#
#   bash scripts/run_qwen_pipeline.sh
#
# Stages 1-3 are cheap to redo and are kept for reproducibility. Stage 4 is the ~2 h run.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/rocm_env.sh

# qwen3_5 is in unsloth's FORCE_FLOAT32 list: plain fp16 NaNs the grad_norm in the backward.
export UNSLOTH_FORCE_FLOAT32=1

# Which base this run trains. The repo default is e4b (the reference recipe, 16 GB class
# card); this script exists for the Qwen arm, which trained on the 8 GB RX 6600M the Gemma 4
# bases cannot fit, so it defaults to qwen35-08b rather than following that default.
# --preset sets the model AND its delimiters together, so the two cannot drift apart.
# The third preset, e2b-qat, is blocked below ~12 GB VRAM -- docs/m10-gemma4-e2b.md.
PRESET="${PRESET:-qwen35-08b}"

KLEV_RUNS="${KLEV_RUNS:-/home/feipe/Documentos/Projects/klev-runs}"
PREP="$KLEV_RUNS/prep/dv7-qwen-full"
CACHE="$KLEV_RUNS/cache/dv7-qwen-full"
RUN="$KLEV_RUNS/qwen35-08b"

# --lr-scheduler-type linear, NOT cosine. On transformers 5.6.2 the Trainer fast loop pins
# the LR at exactly 0 for every step when the scheduler is cosine and warmup_steps > 0
# (verified: 6 steps at warmup 50 all report learning_rate 0, where 1.2e-05 is correct).
# `linear` ramps correctly. This is the one recipe difference from the E4B main run, which
# used cosine; it is a bug workaround, not a choice.
SCHED=linear

echo "=== [1/5] prep full decision-v7 train (preset $PRESET) ==="
$PY data/prep_dataset.py --preset "$PRESET" --suite evals/v7/decision-v7 --split train \
    --out "$PREP" --max-seq 2048 --garbage-frac 0.1

echo "=== [2/5] teacher cache (frozen-base top-k at every content position) ==="
$PY data/cache_teacher_logits.py --preset "$PRESET" --dataset "$PREP" --out "$CACHE" --top-k 32

echo "=== [3/5] prep eval splits ==="
$PY data/prep_dataset.py --preset "$PRESET" --suite evals/v7/decision-v7 --split development \
    --out "$KLEV_RUNS/prep/dv7-qwen-dev" --max-seq 2048 --garbage-frac 0.0
$PY data/prep_dataset.py --preset "$PRESET" --suite evals/v4/transfer-v4 --split development \
    --out "$KLEV_RUNS/prep/tv4-qwen-dev" --max-seq 2048 --garbage-frac 0.0

echo "=== [4/5] train: 3,894 steps = 2 epochs, batch 1 x accum 8 (E4B main run's schedule) ==="
$PY training/train_distill.py --preset "$PRESET" --dataset "$PREP" --teacher-cache "$CACHE" \
    --out-dir "$RUN" --max-steps 3894 --lr 1e-4 --kl-weight 0.3 \
    --lora-r 16 --lora-alpha 32 --warmup-steps 50 --save-steps 500 \
    --lr-scheduler-type "$SCHED"

echo "=== [5/5] eval: decision-v7 dev + transfer-v4 dev ==="
$PY eval/eval_decisions.py --run "$RUN" --dataset "$KLEV_RUNS/prep/dv7-qwen-dev" \
    --out "$RUN-eval-dev"
$PY eval/eval_decisions.py --run "$RUN" --dataset "$KLEV_RUNS/prep/tv4-qwen-dev" \
    --out "$RUN-eval-transfer"

echo "=== PIPELINE DONE ==="
