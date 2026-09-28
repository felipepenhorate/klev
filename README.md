# klev

**klev** (KV + Jev) is a 4B local decision model: Gemma 4 E4B IT fine-tuned with QLoRA plus a
pointer readout, answering typed decision questions (single choice / noul / score /
multi-choice) with calibrated option probabilities and an explicit **rejection channel**. It
runs 4-bit on a single 16 GB GPU and serves the Jev-style decision protocol
(`POST /v1/systemone` shape). The name comes from the KV + Jev lineage: a Kev-lineage pointer
readout serving Jev's typed decision protocol.

It was **built with [Unsloth](https://github.com/unslothai/unsloth)** — `FastModel` for the
4-bit QLoRA run, Unsloth's gradient checkpointing and patching throughout training, teacher
caching, evals and adapter composition. See [Built with Unsloth](#built-with-unsloth).

Working name during development was **M5**; every `main*` run, doc and benchmark label refers
to this model.

## Attribution — built on Kev

klev is **built on top of [Kev](https://github.com/jaredpalmer/kev)** (Apache-2.0, Jared Palmer)
and **reuses part of its code**, vendored rather than reimplemented so klev stays
byte-compatible with the Kev family — same record format, same readout, same metrics, so
Kev-4B remains a like-for-like baseline and any Kev artifact can be read by klev:

| file | from Kev | what is reused |
|---|---|---|
| `model/head.py` | `kev/model.py` | `PointerHead` — the `q`/`k` dot-product readout over option boundary tokens (the garbage/rejection candidate is klev's addition) |
| `data/format.py` | `kev/api.py`, `kev/data.py` | the TypeSafe record format, `render()`, and the internal record builder, plus a multimodal `state` extension |
| `data/metrics.py` | `kev/metrics.py` | the scoring statistics (pure numpy, verbatim) |
| `data/suites.py` | `kev/suite.py` | frozen suite manifests and `jaredpalmer/kev-suites` mirror semantics (sha256-checked) |
| `data/encoding.py`, `data/config.py` | Kev's method | row-form encoding, delimiter-rewrite rule, and the training-context / `trainable_token_indices` recipe |

Also inherited from Kev: the **decision-v7 data** (`jaredpalmer/kev-suites`, the exact
partitions that trained Kev-0.8B/4B/9B's first stage) and the **calibration procedure**
(`scripts/calibrate.py` implements `kev.calibrate`). Kev is Apache-2.0; every vendored file
carries a header naming its Kev origin.

What is klev's own: the Gemma 4 E4B base instead of Qwen, the Unsloth QLoRA training stack
instead of peft + Modal, the KL anchor, the `<unused0>..<unused4>` delimiter embedding deltas,
the rejection channel, and the LoRA stitch. No Kev model weights are used — see `SPEC.md`
§2.1 for the full reuse map.

## Built with Unsloth

klev was **built using [Unsloth](https://github.com/unslothai/unsloth)**, which is what makes
the whole recipe fit one consumer GPU: the 4-bit QLoRA fine-tune of a 4B base, the delimiter
embedding deltas, the pointer head and the teacher-KL cache all run through Unsloth's patched
Gemma 4 stack.

| stage | Unsloth usage |
|---|---|
| base load | `unsloth.FastModel.from_pretrained(..., load_in_4bit=True)` on `unsloth/gemma-4-e4b-it-unsloth-bnb-4bit` |
| fine-tune | `FastModel.get_peft_model` for the rank-16 LoRA + `use_gradient_checkpointing="unsloth"`, driven by `training/DistillTrainer` (a `transformers.Trainer` subclass) |
| trainable tokens | delimiter rows via peft `trainable_token_indices` (Unsloth's Gemma 4 embedding path) |
| teacher KL | cached base logits, read through Unsloth's patched hidden-state/logits access |
| inference / evals | `FastModel` for every decision, chat, drift and stitch eval |
| adapter composition | `tools/merge_adapters.py` (Unsloth-loaded PeftModel) for second-adapter experiments |

Unsloth must be imported **before** `transformers` / `peft` / `trl` (every entry point does
`import unsloth  # noqa: F401` first) because it patches them at import time.

## Model

| | |
|---|---|
| Base | `unsloth/gemma-4-e4b-it-unsloth-bnb-4bit` (Gemma 4 E4B IT is Apache-2.0) |
| Stack | **Unsloth** 2026.9.7 (`FastModel`, `load_in_4bit=True`, `use_gradient_checkpointing="unsloth"`), 4-bit NF4 QLoRA, bf16 compute |
| Trainable | rank-16 LoRA (alpha 32) + 5 delimiter embedding rows + pointer head (~43.7M params) |
| Readout | Kev's `PointerHead` (vendored, see above): `q`/`k` dot-product over option boundary tokens, softmax over K options + a learned "garbage" candidate |
| Data | decision-v7, 15,576 rows, 2 epochs |
| Objective | pointer CE over K+1 + **KL anchor 0.3 to the frozen base** on decision states |
| Training | 8.5 h on one RTX 4080 16 GB, 4-bit, ~6.5 s/step |
| Artifacts | `/mnt/f/distill_jev_runs/main/{adapter,head.pt}` |

The delimiter tokens (`<unused0>..<unused4>`, Gemma reserved slots) mark state / options /
decision positions; their embedding rows are trained as deltas (no vocabulary resize), so
weight tying to `lm_head` stays intact.

## Performance vs competitors

Same records, same scorer (`eval/`), 4B-class models, single 16 GB GPU. Best per row in bold.
Kev-4B: `jaredpalmer/kev-4b` (Qwen3.5-4B base, decision-v7 recipe). Winnow-E4B: `EldanRing`
private 10K targeted decision continuation on the **same Gemma 4 E4B IT base**.

English knowledge and reading (acc):

| suite | klev | Kev-4B | Winnow-E4B |
|---|---|---|---|
| MMLU | 0.657 | **0.725** | 0.701 |
| ARC-Challenge | 0.856 | **0.916** | 0.896 |
| HellaSwag | 0.752 | **0.800** | 0.795 |

Multi-lingual decision tasks (acc):

| suite | klev | Kev-4B | Winnow-E4B |
|---|---|---|---|
| Stanceosaurus ar / en / ru / es | 0.421 / 0.431 / 0.437 / 0.596 | 0.428 / **0.485** / 0.580 / **0.655** | **0.449** / 0.443 / **0.583** / 0.653 |
| CoSt-BR (pt-BR, majority 0.404) | 0.400 | 0.323 | **0.415** |
| in-distribution multilingual (held-out of the trained sets) | 0.785 | 0.794 | **0.823** |
| OOD multilingual (XNLI, Belebele, XStoryCloze, PAWS-X) | 0.751 | 0.756 | **0.850**\* |

\* Winnow answered 3,520/4,000 OOD rows (it rejects some record shapes); per task, XNLI is
flat across the three, Belebele is where Winnow pulls ahead (+18 pp over klev), XStoryCloze
is where klev is strongest (0.924 vs 0.874 Kev-4B, 0.922 Winnow).

Its own decision suite: decision-v7 dev **0.860** (Brier 0.212), transfer-v4 dev 0.751.
Chat-mode retention sits level with the base (MMLU 0.322 → 0.389), i.e. the decision training
does not wreck general knowledge, and pointer mode beats chat mode by +15–46 pp on the same
records.

Honest summary: Kev-4B leads English knowledge, Winnow leads multilingual reading. klev's
edge is the combination of calibrated probabilities + rejection at 4-bit coordinates, and
the property below.

## The KL loss and the stitch

klev is trained with a **KL anchor to the frozen base** on decision states. Consequence:
its representation stays close to the base — drift on held-out Alpaca prompts is
**mean KL 1.864, 88.4 % next-token argmax agreement** — so a LoRA trained *outside* this
fine-tune (game rules: on the base model) can be mounted next to klev without wrecking its
decisions: with the external CoSt-BR LoRA active, decision-v7 dev moves only 0.860 → 0.855.

That alignment is what makes the **stitch** cheap. A third-party LoRA's knowledge is not a
shift of klev's decision state, but it *is* linearly readable from the hidden state of the
prompt format the LoRA was trained on. So with **27 labelled examples** and two scalars:

```
logits_final = logits_ptr(h_sys) + β · log p_alp        (p_alp = NCM in the LoRA's prompt space)
```

| CoSt-BR (test 718) | w(ext)=0.5 | w=1.0 |
|---|---|---|
| klev alone (pointer) | 0.405 | 0.389 |
| 27-shot probe in LoRA's space | 0.457 | 0.475 |
| **stitched klev** | **0.521** | 0.511 |
| external LoRA alone (generation) | 0.522 | 0.522 |

The stitched klev matches the external LoRA's own score with no retraining, while keeping
its decision head and its dev accuracy.

**Latency** (RTX 4080, 4-bit, single stream, `eval/stitch_demo.ipynb`):

| | median | p90 |
|---|---|---|
| vanilla klev (system-one decision, ~250-token prompt) | 205 ms | 231 ms |
| stitched klev (decision + LoRA-prompt forward + fusion) | **456 ms** | 495 ms |
| overhead | **+251 ms (2.23×)** | +264 ms |

The fusion itself is negligible (one NCM distance + two scalars); the cost is the extra
forward over the LoRA's own prompt format (~600 tokens), which is unavoidable — that is
where the foreign knowledge is linearly readable. Throughput-oriented serving can batch
both forwards and cache the prompt-side probe per request. Steering the state directly with the LoRA's class
directions fails (0.37–0.38); the knowledge has to enter as a score. Details and curves:
`docs/m9-lora-stitch.md`.

Why it matters: without the KL loss a decision fine-tune drifts and a foreign LoRA either
collides with it or has to be absorbed with a full delta run; with the anchor, new
capabilities are a 30-example stitching job. The price of the anchor is the same as always —
knowledge/stance gaps vs heavier fine-tunes (see the tables above).

## Install and use

klev is **not on PyPI** — install it from the repository:

```bash
pip install "klev @ git+https://github.com/felipepenhorate/klev.git"
```

On ROCm, keep your torch and let unsloth be installed separately (it would otherwise replace
the ROCm build):

```bash
pip install --no-deps "klev @ git+https://github.com/felipepenhorate/klev.git"
pip install --no-deps unsloth
source scripts/rocm_env.sh
```

Predictions do **not** need unsloth: the default install is torch + transformers + peft +
bitsandbytes, and `load_klev(..., backend="plain")` (the default when `KLEV_BACKEND=plain`, or
`--backend plain`) serves the same 4-bit checkpoint through plain transformers. Same accuracy,
98 % identical choices — [docs/m15-plain-inference.md](docs/m15-plain-inference.md). Unsloth is
now the `train` extra, for fine-tuning and teacher caching: `pip install "klev[train] @ git+..."`.

The library carries the loader, the pointer readout, the record format and both usage paths, so
consuming a checkpoint does not mean cloning this repo. `transformers >= 5.5.0` is a hard floor —
`gemma4` and `qwen3_5` do not exist before it, and on 5.3.0 the failure is misreported by
unsloth as a generic "not supported yet" ValueError. On ROCm, install unsloth with `--no-deps` so
it does not replace your ROCm torch, and `source scripts/rocm_env.sh` before importing.

A decision model is small and stateless, so the normal way to use one is **in-process** — load
it once, call it as often as you like. No server, no port, no daemon. Full detail and the
validation log in [`docs/m14-usage.md`](docs/m14-usage.md).

```python
import unsloth  # noqa: F401  — must precede transformers / peft
from klev import load_klev, answer

model, tokenizer, head = load_klev("lumierenoir/klev-0.8b")   # base read from the checkpoint

result = answer(model, tokenizer, head, {
    "state": "My card was declined twice at a supermarket and the ATM refused it too.",
    "questions": {
        "is_card": {"type": "noul", "instructions": "Is the card being declined?",
                    "criteria": {"false": "no", "true": "yes"}},
        "action":  {"type": "choice", "instructions": "What should the agent do first?",
                    "criteria": {"usage": "ask how the card was used",
                                 "atm":   "raise an ATM fault",
                                 "fraud": "run a fraud check"}},
    }})
result["answers"]["action"]["choice"]   # -> 'atm'
result["answers"]["action"]["none"]     # -> 0.196, the rejection channel
```

```bash
# same thing from a shell, with a plain-language readback
klev-system-one --ckpt lumierenoir/klev-0.8b
klev-system-one --ckpt lumierenoir/klev-0.8b --request my_request.json
```

`temperature` defaults to **1.0, the raw readout** — what the benchmark tables score. Serve at
the fitted value (**2.2974** for the IT arm, **2.2449** for `base/`) to get the calibrated
numbers in the model card; `head.pt` stores 1.0 because it is written during training.

### When you do want HTTP

Several processes sharing one warm copy of the weights, or a caller that isn't Python.

```bash
klev-serve --ckpt lumierenoir/klev-0.8b --port 8090 --temperature 2.2974
curl -s localhost:8090/v1/systemone -H 'content-type: application/json' -d @request.json
```

`POST /v1/systemone`, `GET /v1/models`, `GET /health` — the same routes and response shapes as
`kev`'s own server, so `eval/eval_systemone_http.py` scores it unmodified. Stdlib
`http.server`, so there is no fastapi/uvicorn to install. One decision per request under a lock.

## Layout

```
klev.py      the public entry point: load_klev, answer, the two console scripts
model/       pointer head, delimiter embedding deltas, loading helpers, model/infer.py
data/        encoding, formatting, suites, dataset builders, teacher cache
training/    DistillTrainer (pointer CE + cached-teacher KL), build_model, train_ext_lora
eval/        decision/chat/drift/system-one evals, stitch experiment, demo notebook
examples/    system_one.py -- inline decisions, no server
serve/       server.py -- optional TypeSafe-compatible HTTP endpoint
scripts/     calibration, ROCm environment, pipeline runners
docs/        SPEC.md and milestone reports M5-M14
evals/       decision-v7 suite (train/dev/calibration)
runs/        small local runs
data/        encoding, formatting, suites, dataset builders, teacher cache
training/    DistillTrainer (pointer CE + cached-teacher KL), build_model
eval/        decision/chat/drift/system-one evals, stitch experiment, demo notebook
scripts/     calibration and utilities
docs/        SPEC.md and milestone reports M5–M9
evals/       decision-v7 suite (train/dev/calibration)
runs/        small local runs
```

Artifacts (checkpoints, prepared datasets, benchmark reports) live outside the repo in
`/mnt/f/distill_jev_runs/`.

## Quick start

```bash
# train
python training/train_distill.py --dataset /mnt/f/distill_jev_runs/prep/dv7-full \
    --teacher-cache /mnt/f/distill_jev_runs/cache/dv7 --out /mnt/f/distill_jev_runs/main \
    --epochs 2 --lr 1e-4 --kl-weight 0.3 --lora-r 16 --lora-alpha 32

# decision eval
python eval/eval_decisions.py --run /mnt/f/distill_jev_runs/main \
    --dataset /mnt/f/distill_jev_runs/prep/dv7-dev --out /mnt/f/distill_jev_runs/main-eval

# few-shot stitch with an external LoRA (demo + latency in eval/stitch_demo.ipynb)
python eval/eval_lora_stitch.py --run /mnt/f/distill_jev_runs/main \
    --ext "/home/penhfel/github/Trained Models/My Dataset/gemma-4-e4b_claim" --weight 0.5 \
    --train /mnt/f/distill_jev_runs/prep/costbr-30 \
    --test /mnt/f/distill_jev_runs/prep/bench-costbr-test --out /mnt/f/distill_jev_runs/stitch-w05
```

## Documentation

- `SPEC.md` — design, milestones, probe gates
- `docs/m5-text-run.md` — training run (decision-v7, KL anchor, memory)
- `docs/m6-retention.md` — drift, chat/pointer retention
- `docs/m7-external-comparison.md` — klev vs Kev-4B vs Winnow-E4B
- `docs/m8-multilingual.md` — multilingual delta and OOD suite
- `docs/m9-lora-stitch.md` — few-shot stitch, steering vs gated fusion
- `docs/m14-usage.md` — in-process and HTTP usage, with the validation log
- `docs/m15-plain-inference.md` — predictions without unsloth: the plain backend and its parity
- `docs/m10-gemma4-e2b.md` — why the Gemma 4 E2B arm does not fit 8 GB
- `docs/m11-qwen35-08b.md` — the Qwen3.5-0.8B port and the ROCm environment
- `docs/m12-presets.md` — the base model as a preset (`--preset e4b` / `qwen35-08b` / …)
- `docs/m13-qwen35-08b-base.md` — base-matching the klev/kev comparison
- **`docs/m14-usage.md` — using a model: inline, and over HTTP (with the validation log)**
