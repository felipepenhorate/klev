# M5 — first full text run (decision-v7, 2 epochs)

Date: 2026-09-25
Run: `/mnt/f/distill_jev_runs/main` (checkpoint + `head.pt`), log `/tmp/train_main.log`
Status: baseline complete; the no-KL control arm and the knowledge probes are next.

## What was trained

| | |
|---|---|
| Base | `unsloth/gemma-4-e4b-it-unsloth-bnb-4bit` (4-bit QLoRA, bf16 compute) |
| Data | full `decision-v7` train: **15,576 rows** (12,576 records), 506 garbage rows (3.2%) |
| Adapters | LoRA r=16 α=32 on q/k/v/o/gate/up/down (42.5 M) + delimiter deltas (66 k) |
| Head | `PointerHead(d=2560, dp=256)` with the rejection candidate (`garbage_bias=-4`) |
| Loss | pointer CE over K+1 + 0.3 · cached-teacher forward-KL on content positions |
| Schedule | 3,894 steps = 2 epochs, batch 1 × accum 8, lr 1e-4 cosine, warmup 50 |
| Teacher | offline prefix top-k cache: 1,425,664 positions, k=32 (`cache/dv7-full.npz`) |
| Wall time | **509.6 min (8.5 h)** on the RTX 4080, 0.87 s/micro-batch, no OOM |

Training curve (25-step logs): loss 2.66 → 0.86 at the halfway point → 0.58–0.84 at the
end; pointer CE collapsed from ~5–7 to ~0.005–0.3; the KL stayed bounded at 1.15 → 1.53.
The garbage channel routed from the first logs: `p_garbage` ≈ 1.0 on unanswerable rows vs
≈ 1e-4–2e-3 on answerable ones.

## Results

Reads use `eval/eval_decisions.py`; option probabilities are renormalized over the K
options (`p_none` excluded), so accuracy is directly comparable to Kev's. Served numbers
apply the temperature fitted on the decision-v7 **calibration** partition (T = 1.625,
min micro-NLL over the 121-point log grid, Kev's `TEMPERATURE_FIT`).

| Read | n | raw acc | raw Brier | raw ECE | served Brier | served ECE | coverage @5% error | mean p_none |
|---|---|---|---|---|---|---|---|---|
| `decision-v7` development (trained sources) | 1,468 | **0.860** | 0.227 | 0.079 | **0.212** | **0.037** | **0.722** | 0.028 |
| `transfer-v4` development (new sources) | 764 | **0.751** | 0.360 | 0.145 | 0.323 | 0.076 | 0.550 | 0.014 |

Per question type:

| Read | choice | noul | score |
|---|---|---|---|
| decision-v7 dev | 0.902 (n=756) | 0.915 (n=472) | 0.617 (n=240) |
| transfer-v4 dev | 0.732 (n=444) | 0.807 (n=280) | 0.575 (n=40) |

Weakest transfer sources: `emotion` 0.569 and `mmlu` 0.569 (knowledge is set by the base,
as in Kev finding 8); strongest: `sciq` 0.948, `qnli` 0.850.

Calibration: in-sample fit T = 1.625; the out-of-fold estimate (group-disjoint, 5 folds)
gives ECE **0.016** against the raw 0.079 and is separated from zero, so the served
calibration is not just an in-sample fit. The temperature is written into
`runs/main/head.pt` (`scripts/calibrate.py --write-run`).

## Against the Kev family (report-only)

Different base, tokenizer and (for Kev-4B/9B) three extra delta stages, so these are not
paired comparisons — they only place the run:

| Model | trained sources | new sources (`transfer-v4`) |
|---|---|---|
| Kev-0.8B | 0.827 / 0.838 | 0.648 / 0.697 |
| Kev-4B (current, +3 deltas) | 0.873 / 0.865 | 0.817 / 0.838 |
| Kev-9B (current, +delta) | 0.872 / 0.874 | 0.822 / 0.852 |
| Jev (reference) | – | 0.857 dev |
| **distill_jev (this run, v7 only)** | **0.860** | **0.751** |

Trained-source accuracy is at the Kev-4B level after a single recipe stage. New-source
accuracy sits between Kev-0.8B and Kev-4B — expected, since the current Kev-4B adds
documents-v1, hard-v1 and devtools-v1 deltas on top of the same decision-v7 stage.

## Abstention

`p_none` is low on answerable data (mean 0.028 dev / 0.014 transfer) and does not yet
discriminate errors there: rows abstained at `p_none > 0.5` (2.5% dev, 0.7% transfer)
score 0.833 / 0.800, about the population accuracy. That is the honest reading — the
channel is trained and routes, but it has only seen synthetic "removed option" garbage;
F6's unknowable panel (removed-evidence rows, `hard-v1` missing-fact, `transfer-v9`
unknowable) is what decides whether it improves confident-error behaviour.

## Caveats

- The comparison table is not paired: different base, tokenizer, and Kev's later deltas.
- `score` questions are the weak family (0.617 dev) — ordinal readout, same as Kev.
- The run has no anchor rows yet: the KL only anchors decision states, so general-prompt
  retention (F2) is unmeasured for this checkpoint.
- The temperature was fitted on in-distribution calibration rows; the transfer read's
  served ECE (0.076) shows a single temperature does not fully transfer (Kev finding 5).

## Artifacts

```
/mnt/f/distill_jev_runs/main/            adapter/, head.pt (T=1.625), train_config.json, train_log.json
/mnt/f/distill_jev_runs/main-eval/       rows.json, report.json, calibration.json
/mnt/f/distill_jev_runs/main-cal/        rows.json, report.json
/mnt/f/distill_jev_runs/main-transfer/   rows.json, report.json, calibration.json
/mnt/f/distill_jev_runs/prep/dv7-full    prepared rows (15,576)
/mnt/f/distill_jev_runs/cache/dv7-full.npz   teacher cache (1.43 M positions, k=32)
```

## Next

1. **No-KL control arm** (λ=0, same data/schedule) for the F1/F2 retention comparison.
2. **F2 knowledge probes**: drift on a held-out general mix, MMLU/ARC-style pointer probes,
   and the anchor-row pipeline (the strongest anti-forgetting term in DuplexCascade D2).
3. **F6 unknowable panel** to score the rejection channel properly.
4. Then M2 (images) and M3 (Qwen3.5-9B synthetic) as planned.
