# M13 — base-matching the klev/kev comparison

Date: 2026-09-28
Runs: `qwen35-08b` (IT), `qwen35-08b-base` (`-Base`), `kev-0.8b` (published)
Why: M11 compared klev against kev on **different base checkpoints**, and that turned out to
move the stitch numbers. This run holds the base fixed and re-measures.

## The confound

| | klev (M11) | kev-0.8B |
|---|---|---|
| base | `Qwen/Qwen3.5-0.8B` (IT) | `Qwen/Qwen3.5-0.8B-Base` |

The base is not neutral for the few-shot stitch. It determines how strong an *external* task
LoRA can be, and the stitch can only import what the adapter holds. Same data, same
hyper-parameters, only the base checkpoint differs:

| external adapter, tweet_eval | probe on the LoRA's prompt space |
|---|---|
| trained on the IT base | **0.715** |
| trained on `-Base` | 0.660 |

The same ordering held on RumourEval (0.490 vs 0.430). So M11's +11.0 vs +3.0 stitch gap
measured the adapter's base as much as the decision model.

## What was run

`--preset qwen35-08b-base`: the identical recipe on `Qwen/Qwen3.5-0.8B-Base`. The `-Base`
tokenizer resolves the five delimiters to the same ids (248060..248064), so the encoder needed
no change and prep produced **15,576 rows, identical to the IT arm**. The teacher cache had to
be **rebuilt** (1,413,968 positions, k=32) because the KL anchor is computed from the frozen
base's own logits.

Training: 3,894 steps, 2 epochs, batch 1 × accum 8, **195.2 min** on the RX 6600M.

## decision-v7 dev (n = 1,468), served

Each model served with its own temperature fitted on decision-v7 calibration (1,148 rows).

| | klev (IT) | klev (-Base) | kev-0.8B |
|---|---|---|---|
| accuracy | **0.8120** | 0.8065 | **0.8120** |
| Brier | **0.2590** | 0.2679 | 0.2685 |
| ECE | 0.0365 | **0.0248** | 0.0274 |
| NLL | **0.4785** | 0.4984 | 0.5102 |
| coverage @ 5 % error | **0.6158** | 0.5640 | 0.4360 |
| AURC | **0.0592** | 0.0642 | 0.0717 |
| fitted temperature | 2.2974 | 2.2449 | 1.2311 |

## transfer-v4 dev (n = 764), served

| | klev (IT) | klev (-Base) | kev-0.8B |
|---|---|---|---|
| accuracy | **0.6636** | 0.6283 | 0.6126 |
| Brier | **0.4372** | 0.4613 | 0.4542 |
| ECE | **0.0207** | 0.0398 | 0.0735 |
| NLL | 0.7538 | 0.7754 | **0.7524** |
| coverage @ 5 % error | 0.1846 | 0.1649 | **0.2395** |
| AURC | **0.1719** | 0.2007 | 0.1876 |

## Stitch on tweet_eval hate (200 rows, 32 shots)

Both `-Base` arms use the **same** external LoRA (`ext-lora-tweet-base`), so this row is
recipe against recipe.

| | klev (-Base) | kev-0.8B |
|---|---|---|
| pointer alone | 0.625 | **0.645** |
| 32-shot probe, LoRA prompt space | **0.695** | 0.660 |
| 32-shot probe, decide state | 0.665 | **0.705** |
| logreg, LoRA prompt space | 0.695 | 0.655 |
| A — steer class directions | 0.625 | 0.635 |
| A2 — steer global direction | 0.445 | 0.605 |
| **B — gated fusion** | **0.700** | 0.675 |
| gain over bare head | **+7.5 pts** | +3.0 pts |

`T = 57.25`, `beta = 16.0`, `ext_weight = 1.0`.

## Reading

**The stitch gap is real but half of it was the adapter.** Base matching takes the advantage
from +8.0 points to +4.5. klev's fusion again matches its own probe (0.700 vs 0.695) where
kev's lands *below* its decide-state probe. So: roughly half base, half recipe.

**The decision advantage is mostly recipe.** Same recipe and data, only IT → `-Base`, moves
dev accuracy by 0.6 points (0.8120 → 0.8065) and transfer by 3.5 (0.6636 → 0.6283). Against
kev that is +1.6 on transfer — real, but much smaller than the stitch gap.

**`score` prefers the `-Base` arm on dev** (0.6167 vs 0.5833), the one family that improves
when the adapter gets worse. On transfer it is 0.275 vs kev's 0.350, but n = 40, so that column
carries no weight.

**kev's decide state already contains the signal.** Its `probe, decide state` of 0.705 exceeds
its own alpaca probe (0.660) and its fused result (0.675): a linear probe on kev's *own*
decision state recovers hate-speech with no external knowledge at all. This now holds with two
different external adapters, so it looks like a property of kev's decision representation on
this task rather than an artifact of one adapter. It is also the most plausible reason its
stitch gain is small — there was little left to import.

## What this does not settle

- **One seed, one task.** Both stitch datasets are 2-class-or-4-class text classification;
  nothing here tests a generative or multi-hop task.
- **Calibration is not a function of base quality.** The `-Base` arm has a *worse* transfer ECE
  (0.0398) than the IT arm (0.0207) despite being the base kev's adapter prefers. Unexplained.
- **The kev adapter still went through my harness.** Its 372 checkpoint keys needed a prefix
  remap to load at all, and its `head.pt` was shimmed with `deltas: {}` because kev trains no
  special embeddings. Both are faithful translations of `special_embeddings: false`, but its
  numbers are not produced by kev's own evaluation code.
- **kev-0.8B is a longer-trained artifact**: warm-started from an earlier kev-0.8B over 22,539
  records including 16,539 from a separate joint mixture, against one 2-epoch pass here on
  decision-v7. It is not a recipe against a recipe on that axis.

## Reproduce

```bash
bash scripts/run_qwen_base_pipeline.sh      # preset qwen35-08b-base: prep, cache, train, eval, stitch
python scripts/calibrate.py \
  --calibration $KLEV_RUNS/qwen35-08b-base-eval-cal/rows.json \
  --rows $KLEV_RUNS/qwen35-08b-base-eval-dev/rows.json \
  --out $KLEV_RUNS/qwen35-08b-base-calib-dev
```

Artifacts: `lumierenoir/klev-0.8b` (IT arm, top level) and `lumierenoir/klev-0.8b/base`
(`-Base` arm, with the shared external LoRA and its fitted probe).
