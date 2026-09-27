# M7 — external comparison: ours vs Kev-4B vs Winnow-E4B (system_one) + language scaling

Date: 2026-09-26
Models (all run locally, same records, decision mode):
- **ours** — `runs/main` (Gemma 4 E4B **it** 4-bit QLoRA + pointer head, decision-v7, 2 epochs).
- **Kev-4B** — `jaredpalmer/kev-4b` via the Kev repo (`KEV_DTYPE=bf16`; fp32 4B does not fit 16 GB).
- **Winnow-E4B** — `EldanRing/Winnow-E4B` Q8_0 GGUF, its own native server (`winnow-inference`, commit `77d1458`), text-only profile.

Chat mode is not a common path (Kev has no chat), so every number below is **system_one**
(pointer/typed-decision readout); our chat-mode numbers stay in `docs/m6-retention.md`.

## English knowledge benchmarks (1,000 items each, our bench records)

| task | ours | Kev-4B | Winnow-E4B |
|---|---|---|---|
| MMLU | 0.657 | **0.725** | 0.701 |
| ARC-Challenge | 0.856 | **0.916** | 0.896 |
| HellaSwag | 0.752 | **0.800** | 0.795 |

Kev-4B (Qwen3.5-4B base, decision-v7 recipe) leads all three. Winnow-E4B is the closest
comparison to ours — the **same Gemma 4 E4B IT base**, decision fine-tuned on a private
targeted 10K continuation — and beats ours by **+3.4 to +4.4 pp**, i.e. the gap is training
data/method, not the base model.

## Language scaling — Stanceosaurus (1,500 per language, 5-way stance) and CoSt-BR (pt-BR, 5-way)

| suite | ours | Kev-4B | Winnow-E4B | majority class |
|---|---|---|---|---|
| Stanceosaurus Arabic | 0.421 | 0.428 | **0.449** | 0.473 |
| Stanceosaurus English | 0.431 | **0.485** | 0.443 | 0.459 |
| Stanceosaurus Russian | 0.437 | 0.580 | **0.583** | 0.522 |
| Stanceosaurus Spanish | 0.596 | **0.655** | 0.653 | 0.521 |
| CoSt-BR (pt-BR) | 0.400 | 0.323 | **0.415** | 0.404 |

Observations:
- All three are weak on Stanceosaurus relative to the majority-class baseline; Kev-4B and
  Winnow clear it on Russian/Spanish, ours only on Spanish. The task (conversational stance,
  five classes, multilingual, claim-conditioned) is far outside every model's decision
  training, and the datasets are imbalanced (majority 45–52%).
- On pt-BR CoSt-BR our Gemma-based model beats Kev-4B by +7.7 pp (0.400 vs 0.323); Winnow is
  slightly ahead of ours (+1.5 pp). Gemma's Portuguese coverage shows here.
- Winnow-E4B is the most consistent across languages among the three.

## Protocol notes (make these numbers readable, not perfect)

- Our MMLU/ARC rows are 989 (11 dropped by the 384-token state cap at prep); Winnow scored the
  raw 1,000 records. HellaSwag is 1,000 for all.
- Kev-4B rejected records over its own context limits: stance 5,736/5,923, CoSt-BR 626/718.
  Our/Winnow numbers include those rows; the comparison is therefore not perfectly paired.
- Winnow Q8_0 (8.0 GB) vs ours 4-bit NF4 vs Kev-4B bf16; Winnow's probabilities come back at
  its shipped decision temperature (1.257), ours are raw (T=1) here — accuracy is unaffected,
  Brier/ECE are not directly comparable across models.
- Stanceosaurus prompts were converted to decision records from the hydrated prompt files in
  `training-conversational-stance/Datasets/Stanceossaurus/` (the public Stanceosaurus repo
  ships empty tweet text; the local hydrated prompts are the usable copy).
- Winnow's model card reports its own Kev-v9/JevBench suites (72.66 / 80.52), which are
  different records — no number there was reused here.

## Artifacts

```
/mnt/f/distill_jev_runs/bench/{mmlu,arc,hellaswag,stanceosaurus,costbr-test}.jsonl
/mnt/f/distill_jev_runs/kev4b-{mmlu,arc,hellaswag,stanceosaurus,costbr-test}/
/mnt/f/distill_jev_runs/winnow-{mmlu,arc,hellaswag,stanceosaurus,costbr-test}/
/mnt/f/distill_jev_runs/main-{stanceosaurus,costbr}/   (our model)
```

## Next

- The Winnow comparison says the gap on the same base is the training data/method: a targeted
  decision continuation (their 10K private) beats decision-v7-only. Our synthetic stage was
  meant to be that, and its first attempt failed (mixed, hedging-heavy data; see the M5b
  findings). Regenerate with shorter, policy-heavy documents and a higher synthetic share.
- Language work is untrained: a pt-BR (and optionally es/ru/ar) decision family is the cheapest
  lever if multilingual matters.
