#!/usr/bin/env bash
# klev on Qwen3.5-0.8B-Base -- the same recipe as scripts/run_qwen_pipeline.sh, on the
# checkpoint kev-0.8B uses, so the stitch comparison is base-matched.
#
# Why this exists: the two arms differed in base (IT vs -Base), and the base determines how
# strong an external task LoRA can be (tweet_eval probe 0.715 on IT vs 0.660 on -Base). The
# stitch can only import what the adapter has, so the +11.0 vs +3.0 gap was confounded. Holding
# the base fixed isolates the recipe.
#
# Needs its own teacher cache: the KL anchor is computed from the frozen base's own logits, so
# the IT-base cache does not transfer.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/rocm_env.sh
export UNSLOTH_FORCE_FLOAT32=1          # qwen3_5 is in unsloth's FORCE_FLOAT32 list

PRESET=qwen35-08b-base
KLEV_RUNS="${KLEV_RUNS:-/home/feipe/Documentos/Projects/klev-runs}"
PREP="$KLEV_RUNS/prep/dv7-qwen-base"
CACHE="$KLEV_RUNS/cache/dv7-qwen-base"
RUN="$KLEV_RUNS/qwen35-08b-base"

echo "=== [1/4] prep full decision-v7 train (preset $PRESET) ==="
$PY data/prep_dataset.py --preset "$PRESET" --suite evals/v7/decision-v7 --split train \
    --out "$PREP" --max-seq 2048 --garbage-frac 0.1

echo "=== [2/4] teacher cache (frozen -Base, must be rebuilt) ==="
$PY data/cache_teacher_logits.py --preset "$PRESET" --dataset "$PREP" --out "$CACHE" --top-k 32

echo "=== [3/4] prep dev splits ==="
$PY data/prep_dataset.py --preset "$PRESET" --suite evals/v7/decision-v7 --split development \
    --out "$KLEV_RUNS/prep/dv7-qwen-base-dev" --max-seq 2048 --garbage-frac 0.0
$PY data/prep_dataset.py --preset "$PRESET" --suite evals/v7/decision-v7 --split calibration \
    --out "$KLEV_RUNS/prep/dv7-qwen-base-cal" --max-seq 2048 --garbage-frac 0.0
$PY data/prep_dataset.py --preset "$PRESET" --suite evals/v4/transfer-v4 --split development \
    --out "$KLEV_RUNS/prep/tv4-qwen-base-dev" --max-seq 2048 --garbage-frac 0.0

echo "=== [4/4] train: 3,894 steps = 2 epochs, batch 1 x accum 8 ==="
$PY training/train_distill.py --preset "$PRESET" --dataset "$PREP" --teacher-cache "$CACHE" \
    --out-dir "$RUN" --max-steps 3894 --lr 1e-4 --kl-weight 0.3 \
    --lora-r 16 --lora-alpha 32 --warmup-steps 50 --save-steps 500 \
    --lr-scheduler-type linear

echo "=== eval: decision-v7 dev + transfer-v4 dev ==="
$PY eval/eval_decisions.py --preset "$PRESET" --run "$RUN" \
    --dataset "$KLEV_RUNS/prep/dv7-qwen-base-dev" --out "$RUN-eval-dev"
$PY eval/eval_decisions.py --preset "$PRESET" --run "$RUN" \
    --dataset "$KLEV_RUNS/prep/tv4-qwen-base-dev" --out "$RUN-eval-transfer"

echo "=== stitch on tweet_eval, reusing the -Base external LoRA already trained ==="
$PY eval/eval_lora_stitch.py --preset "$PRESET" --run "$RUN" \
    --ext "$KLEV_RUNS/ext-lora-tweet-base/adapter" --weight 1.0 \
    --train "$KLEV_RUNS/prep/tweet-train" --test "$KLEV_RUNS/prep/tweet-validation" \
    --train-prompts "$KLEV_RUNS/tweet_eval/alpaca_train.json" \
    --test-prompts "$KLEV_RUNS/tweet_eval/alpaca_validation.json" \
    --max-seq-length "$KLEV_STITCH_MAX_SEQ" --out "$RUN-stitch-tweet"

echo "=== PIPELINE DONE ==="
