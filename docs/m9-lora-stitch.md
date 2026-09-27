# M9 — few-shot stitch: making M5's head use an external LoRA's space

Date: 2026-09-27
Script: `eval/eval_lora_stitch.py` · Runs: `/mnt/f/distill_jev_runs/stitch-w1`, `stitch-w05`
External LoRA: `Trained Models/My Dataset/gemma-4-e4b_claim` (CoSt-BR, base gemma-4-e4b-it, alpaca format).
Train pool: 27 labelled CoSt-BR rows (30 selected, 3 dropped at 512-token prep; all 5 classes).

Question: with a frozen M5 + frozen external LoRA, can <=30 labelled examples teach the
pointer head to read the foreign LoRA's knowledge, instead of retraining a delta?

## Diagnostics (no training)

| | w=1.0 | w=0.5 |
|---|---|---|
| pointer head baseline (test 718) | 0.389 | 0.405 |
| NCM probe on `h_alp` (alpaca-prompt hidden, 27 shot) | **0.475** | 0.457 |
| NCM probe on `h_sys` (system-one decide state) | 0.389 | 0.329 |
| logistic probe on `h_alp` (C=0.05) | 0.421 (train 1.00) | – |

The LoRA's class information *is* linearly decodable from its own prompt format (0.46-0.48)
and is *absent* from the system-one decide state — exactly the gap the stitch has to close.

## Approach A — steering with class directions (fails)

`v_c = mean(h_alp | c) - mean(h_alp)`, decision = `argmax_c head(h_sys + α·v_c)[c]`,
α tuned on the 27 rows.

| | w=1.0 | w=0.5 |
|---|---|---|
| best α | 5.0 | 2.25 |
| test acc (tuned) | 0.370 | 0.379 |
| global prompt direction `mean(h_alp - h_sys)` | 0.394 | 0.394 |

Adding the alpaca directions to the decide state does not help; the directions are dominated
by the prompt-format shift and tilt the option scores sideways.

## Approach B — gated fusion, the prompt key in the readout (works)

`logits_final = logits_ptr + β · log p_alp`, with `p_alp = softmax(-d_c² / T)` from the
27-example NCM in the alpaca space. Two scalars trained (T, β), 27 examples.

| | w=1.0 | w=0.5 |
|---|---|---|
| probe alone | 0.475 | 0.457 |
| **fused (T,β tuned on train)** | **0.511** | **0.521** |
| gain over head | +12.2 pp | +11.6 pp |
| best cell on the grid | 0.514 | **0.535** |

The test surface is broad: at w=0.5, β in [2,16] × all temperatures stays in 0.46-0.535, so
the result is not a lucky hyperparameter cell; training accuracy saturates at 0.704 for all
selected cells (what 27 points can fit).

## Reading

- 30 examples cannot teach a new decoder, but they *can* condition the readout on the LoRA's
  own representation: the alpaca-prompt hidden state carries the class signal, and fusing it
  into the pointer logits imports roughly half of the LoRA's knowledge (0.40 → 0.52).
- The fused M5 matches the external LoRA's own generative CoSt-BR score (0.522) while keeping
  M5's decision machinery; dv7-dev retention is untouched by construction (the fusion is only
  used when the prompt key is available, and adds 2 scalars, no weight changes).
- "Steering the state" (A) does not work; "conditioning the readout" (B) does. The knowledge
  is not a shift of the system-one state but a separate linearly-readable signal in the LoRA's
  space, so it has to enter as a score, not as a perturbation.
- Follow-ups: sweep train-set size (10/20/27), replace NCM with a small LDA, and test whether
  the fused model's `none`/garbage channel stays calibrated on CoSt-BR.

## Reproduce

```bash
/home/penhfel/unsloth_uv/bin/python eval/eval_lora_stitch.py --run /mnt/f/distill_jev_runs/main \
  --ext "/home/penhfel/github/Trained Models/My Dataset/gemma-4-e4b_claim" --weight 0.5 \
  --train /mnt/f/distill_jev_runs/prep/costbr-30 --test /mnt/f/distill_jev_runs/prep/bench-costbr-test \
  --out /mnt/f/distill_jev_runs/stitch-w05
```

Feature caches (`features_w*.pt`) are written next to the report; delete to re-extract.
