# LinkedIn post — klev

Draft for the release announcement. Numbers are the unsloth path (the one the model cards
publish); the plain-transformers path reproduces them to ~98% identical choices
(`docs/m15-plain-inference.md`). The CoSt-BR row is matched: both arms scored on the same 58
pt-BR rows.

---

**klev — decision models that can borrow a LoRA's knowledge**

I released **klev** (KV + Jev): small models that answer typed questions with calibrated
probabilities and an explicit "I don't know" channel. Apache-2.0, open weights.

**The arms**
- **klev-e4b** — Gemma 4 E4B, 4-bit → https://huggingface.co/lumierenoir/klev-e4b
- **klev-0.8b** — Qwen3.5-0.8B, 0.89 GB of weights, full fine-tune in 192 min on an 8 GB card →
  https://huggingface.co/lumierenoir/klev-0.8b

**Where each one wins**
- **Portuguese**: the 4B arm. On the same 58 pt-BR rows, 0.466 vs 0.328
- **Size**: the 0.8B arm. It ties kev-0.8B in-distribution (0.812 vs 0.812, better Brier/NLL)
  and beats it by 5.1 points out of it (transfer-v4: 0.664 vs 0.613)
- **Neither** wins the 4B English/multilingual leaderboard: Kev-4B leads English
  (MMLU .725 vs our .657), Winnow leads multilingual. The gap is training data, not architecture

**The gimmick, tested on both arms**
- A KL anchor to the frozen base keeps klev close to it
- so a LoRA trained by anyone else on the same base can be loaded next to it without breaking
  its own decisions (dev 0.860 → 0.855)
- and its knowledge can be imported with ~30 labelled rows and two scalars:
  - Gemma 4 + CoSt-BR: 0.405 → **0.521** (what the LoRA scores on its own: 0.522)
  - Qwen 0.8B + tweet_eval hate detection: 0.605 → **0.715** (probe 0.715, LoRA alone 0.730)
  - Qwen 0.8B + RumourEval: 0.362 → **0.479**
- +251 ms per question, no retraining, no weight changes

Code: https://github.com/felipepenhorate/klev

#MachineLearning #LLM #OpenSource #FineTuning #LoRA

---

## Even shorter (one paragraph + bullets)

**klev** — small decision models, open-weights at 4B (Gemma 4) and 0.8B (Qwen3.5, 0.89 GB,
trains in 192 min on an 8 GB card). Calibrated probabilities plus an explicit "I don't know".
The 4B arm is the better Portuguese model (0.466 vs 0.328 on the same pt-BR rows); the 0.8B ties
kev-0.8B in-distribution and beats it by 5 points out of it. And the part I like: a KL anchor
to the base lets it **load any same-base LoRA without breaking its own decisions, and import
that LoRA's knowledge with ~30 examples** — 0.405 → 0.521 (Gemma/CoSt-BR) and 0.605 → 0.715
(Qwen/tweet_eval), no retraining.

Apache-2.0 · https://huggingface.co/lumierenoir/klev-e4b ·
https://huggingface.co/lumierenoir/klev-0.8b · https://github.com/felipepenhorate/klev

---

## Short version, lists (the one to post if the long draft is too much)

**klev — decision models that can borrow a LoRA's knowledge**

Open-weights decision models: typed questions in, calibrated probabilities plus an explicit
"I don't know" out. Apache-2.0.

- **klev-e4b** (Gemma 4 E4B, 4-bit) — the better Portuguese model: 0.466 vs 0.328 on the same
  pt-BR rows
- **klev-0.8b** (Qwen3.5-0.8B) — 0.89 GB of weights, a full fine-tune in 192 min on an 8 GB card;
  ties kev-0.8B in-distribution (0.812) and beats it by 5.1 pts out of it (0.664 vs 0.613)
- **The gimmick** — a KL anchor to its base means any LoRA trained by someone else on that same
  base can be loaded next to it without breaking its decisions (dev 0.860 → 0.855), and that
  LoRA's knowledge can be imported with ~30 labelled rows and two scalars:
  - Gemma 4 / CoSt-BR: 0.405 → **0.521**
  - Qwen 0.8B / tweet_eval: 0.605 → **0.715**
  - Qwen 0.8B / RumourEval: 0.362 → **0.479**
  - +251 ms per question, no retraining

https://huggingface.co/lumierenoir/klev-e4b · https://huggingface.co/lumierenoir/klev-0.8b ·
https://github.com/felipepenhorate/klev

#MachineLearning #LLM #OpenSource #LoRA
