# LinkedIn post — klev

Draft for the release announcement. Numbers are the unsloth path (the one the model cards
publish); the plain-transformers path reproduces them to ~98% identical choices
(`docs/m15-plain-inference.md`). Update the CoSt-BR / dev figures if the card re-run moves them.

---

**klev — decision models that can borrow a LoRA's knowledge**

I released **klev** (KV + Jev): small models that answer typed questions with calibrated
probabilities and an explicit "I don't know" channel. Apache-2.0, open weights.

**The arms**
- **klev-e4b** — Gemma 4 E4B, 4-bit
- **klev-0.8b** — Qwen3.5-0.8B: 0.89 GB of weights, full fine-tune in 192 min on an 8 GB card

**Where it stands** (same records, same scorer)
- English knowledge: Kev-4B leads — MMLU .725 vs our .657
- Multilingual: Winnow leads. We don't win those; the gap is training data, not architecture

**The gimmick**
- A KL anchor to the frozen base keeps klev close to it
- so any LoRA trained by someone else on that base can be loaded next to it: dev 0.860 → 0.855
- and that LoRA's knowledge can be imported with **27 examples and two scalars**:
  CoSt-BR 0.405 → 0.521 — what the LoRA scores on its own — for +251 ms, no retraining

`lumierenoir/klev-e4b` · `lumierenoir/klev-0.8b` · `github.com/felipepenhorate/klev`

#MachineLearning #LLM #OpenSource #FineTuning

---

## Even shorter (if you want one paragraph + bullets)

**klev** — small decision models, open-weights at 4B (Gemma 4) and 0.8B (Qwen3.5, 0.89 GB,
trains in 192 min on an 8 GB card). Calibrated probabilities plus an explicit "I don't know".
It doesn't beat the bigger decision models (Kev-4B leads English, Winnow multilingual) — but:

- a KL anchor to its base lets it load **any same-base LoRA without breaking its own decisions**
  (dev 0.860 → 0.855)
- and import that LoRA's knowledge with **27 examples**: 0.405 → 0.521 on CoSt-BR, no retraining

Apache-2.0 · `github.com/felipepenhorate/klev`
