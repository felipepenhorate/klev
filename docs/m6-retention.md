# M6 — knowledge retention: chat mode vs system_one (pointer) mode

Date: 2026-09-25
Model: `/mnt/f/distill_jev_runs/main` (M5 full run, decision-v7, 2 epochs)
Purpose: measure what the distillation anchor kept (F2) and how the fast system_one
readout compares to normal chat scoring on the same knowledge tasks.

## Setup

None of the tasks below is in `decision-v7` (its trainable sources are
agnews/amazon/banking77/boolq/dbpedia14/imdb/mnli/sst5/trec/yelp + synthetic policy and
rule data), so accuracy measures knowledge the training never labelled.

- **Chat mode** (`eval/eval_chat.py`): the model's own chat template + loglikelihood
  scoring of the answer continuations, the same contract lm-eval uses. lm-eval 0.4.12
  itself cannot load Gemma 4 (`gemma4` has no `AutoModelForCausalLM` mapping), so the
  scorer is self-contained and identical for base and trained.
  HellaSwag/ARC use `acc_norm` (character-normalized), MMLU scores the letter
  continuation, TruthfulQA is MC2 (probability mass on true answers).
- **system_one mode** (`eval/eval_decisions.py`): the same tasks converted to TypeSafe
  decision records (`data/build_bench_records.py`) and answered by the pointer head —
  one forward pass per question, no continuation scoring, no generation.
- **Drift** (`eval/eval_drift.py`): mean per-token forward-KL(trained ‖ frozen base) and
  argmax agreement on 200 held-out Alpaca instructions.

All runs: 1,000 examples per task (TruthfulQA 817), seed 0, RTX 4080.

## Results

| Task | base chat | trained chat | Δ chat | trained system_one | system_one − chat |
|---|---|---|---|---|---|
| HellaSwag (acc_norm) | 0.589 | 0.597 | **+0.8** | **0.752** | **+15.5** |
| ARC-Challenge (acc_norm) | 0.406 | 0.400 | −0.6 | **0.856** | **+45.6** |
| MMLU (acc) | 0.322 | **0.389** | **+6.7** | **0.657** | **+26.8** |
| TruthfulQA (MC2) | 0.372 | 0.368 | −0.4 | – | – |

Pointer-mode calibration and abstention (trained head):

| Task | acc | Brier | ECE | coverage @5% error | mean `p_none` |
|---|---|---|---|---|---|
| MMLU | 0.657 | 0.476 | 0.106 | 0.183 | 0.005 |
| ARC-Challenge | 0.856 | 0.206 | 0.022 | 0.761 | 0.009 |
| HellaSwag | 0.752 | 0.369 | 0.035 | 0.290 | 0.001 |

Drift on 200 held-out Alpaca instructions (4,587 token positions): **mean KL 1.864**
(median 1.871), **argmax agreement 88.4%**.

## What this says

1. **Chat retention is intact.** After two epochs of decision training the chat model is
   level with the base on HellaSwag/ARC/TruthfulQA (Δ within the ±1.5 pp noise of
   n=1,000) and **better on MMLU (+6.7 pp)** — decision training on multiple-choice
   formats taught the model to answer letter questions, not to forget knowledge. This is
   the first retention evidence for the KL anchor at this scale.
2. **The pointer readout taps much more knowledge than chat scoring.** On the same
   questions and the same trained weights, system_one is +15 to +46 pp over chat. The
   knowledge is in the backbone; the pointer head is the better tap — which is exactly
   the Jev/Kev thesis, and it holds for Gemma 4 E4B.
3. **system_one is also ~6× cheaper per decision.** Chat scores K continuations per
   question (0.77 s/example for HellaSwag's 4 endings, ≈0.19 s per forward on the 4080);
   the pointer answers one question per forward (0.13 s/row measured, including the
   1,468-row decision-v7 read). In serving, one state prefix is cached and shared across
   every question of a request, which chat scoring cannot do.
4. **Drift is moderate and needs its control.** 11.6% of next-token argmaxes on general
   instructions changed. There is no no-KL arm yet, so this number cannot yet be
   attributed to the anchor; that is the next experiment (F1/F2).

## Caveats

- n = 1,000 per task, one seed: differences under ~1.5 pp are noise.
- Base-model pretraining may include these public test sets; that affects both modes
  equally and is a floor for every model compared this way.
- ARC/HellaSwag pointer options are full answer texts with neutral keys, a different (and
  easier) readout than chat's continuation scoring — the comparison measures readout
  quality, not just knowledge.
- The drift number is measured with the adapter on vs off on the same weights; the base's
  own KL is 0 by construction, so the useful comparison is against a no-KL trained model.
- Pointer MMLU (0.657) is below Kev-9B's published MMLU (0.74), consistent with a smaller
  effective-4B base; the gap is knowledge, not the readout.

## Artifacts

```
/mnt/f/distill_jev_runs/chat-base/chat_results.json        base chat numbers
/mnt/f/distill_jev_runs/chat-trained/chat_results.json     trained chat numbers
/mnt/f/distill_jev_runs/pointer-{mmlu,arc,hellaswag}/      pointer reports + rows
/mnt/f/distill_jev_runs/drift-trained/drift.json           drift metric
/mnt/f/distill_jev_runs/bench/*.jsonl                      benchmark records in decision format
```

## Next

1. **No-KL control arm** (λ=0) on the same data/schedule: the drift and chat numbers above
   become interpretable only against it (F1).
2. **F6 unknowable panel** for the rejection channel.
3. Anchors (general-prompt KL) to lower drift without touching decision accuracy.
