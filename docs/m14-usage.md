# M14 — using a klev model: inline, and over HTTP

Date: 2026-09-28
Code: `model/infer.py` (the shared core), `examples/system_one.py`, `serve/server.py`
Validates: `SPEC.md` promised a `serve/server.py` that had never been written; both paths below
were run on this machine before being documented.

## klev is a library first

A decision model is small and stateless, so the default way to use one is **in-process** — load
it once, call it as many times as you like. There is no daemon to run, no port, no lifecycle.

```python
from model.infer import load_klev, answer

model, tokenizer, head = load_klev("lumierenoir/klev-0.8b")   # or a local dir

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
```

Verified output from exactly this request (IT arm, temperature 2.2974):

```json
{"answers": {
  "is_card": {"type": "noul", "noul": 0.9286, "none": 0.012948},
  "action":  {"type": "choice", "choice": "atm", "confidence": 0.4277,
              "probabilities": {"usage": 0.1334, "atm": 0.5708, "fraud": 0.0995},
              "none": 0.196325}}}
```

`none` is the rejection channel — the probability mass on "none of these apply", renormalised
off the K options so the returned distribution still sums to 1.

`model/infer.py` deliberately shares `data/encoding.py`, `data/format.py` and `model/head.py`
with the scorer rather than reimplementing them, so the probabilities a user sees are the ones
`eval/eval_decisions.py` measures. If those ever drift, the published numbers stop describing
the model.

### The three question types

```python
{"type": "noul",   "instructions": "...", "criteria": {"false": "...", "true": "..."}}
{"type": "choice", "instructions": "...", "criteria": {"key_a": "...", "key_b": "..."}}
{"type": "score",  "instructions": "...", "criteria": ["level 0", "level 1", ...]}
```

A request is validated by `data/format.py`'s `SystemOneRequest`, so unknown question types and
out-of-range `criteria` are rejected before any GPU work. Labels, `target` and `_meta` are
ignored, which means a labelled eval record can be passed straight through.

### Temperature

`answer(..., temperature=1.0)` is the **raw** readout — the same thing the benchmark tables
score. The model card's *served* numbers use the per-model temperature from
`scripts/calibrate.py`:

| arm | temperature |
|---|---|
| `klev-0.8b` (IT) | 2.2974 |
| `klev-0.8b/base` (`-Base`) | 2.2449 |

`head.pt` stores `1.0` because it is written during training and the head only tempers in eval
mode, so a served deployment has to supply the fitted value. That is a real footgun: the raw
readout has ECE 0.113 against 0.037 served on decision-v7 dev.

## Command line, still no server

```bash
python examples/system_one.py --ckpt lumierenoir/klev-0.8b                  # built-in demo
python examples/system_one.py --ckpt lumierenoir/klev-0.8b --request r.json # your request
echo '{"state":"...","questions":{...}}' | python examples/system_one.py --ckpt ... --request -
```

Output ends with a plain-language readback, which is the quickest way to sanity-check a
checkpoint:

```
channel_issue  noul    p(true)=0.885  -> yes   reject=0.015
next_step      choice  -> atm_fault   conf=0.371  reject=0.193
severity       score   -> 1.52        reject=0.025
```

## The server, when you actually want one

Reach for HTTP when several processes share one warm copy of the weights, or when the caller
isn't Python. Not otherwise — it adds a port, a process and a failure mode to save ~1 GB and a
second of load time.

```bash
python serve/server.py --ckpt lumierenoir/klev-0.8b --port 8090 --temperature 2.2974
```

| route | |
|---|---|
| `POST /v1/systemone` | `{"state", "questions"}` → `{"answers": {qid: typed answer}}` |
| `GET /v1/models` | loaded base, adapter, temperature, whether the rejection channel exists |
| `GET /health` | `{"ok": true, "model": ...}` |

```bash
curl -s localhost:8090/v1/systemone -H 'content-type: application/json' -d @request.json
```

Same routes and response shapes as `kev`'s own server, so **`eval/eval_systemone_http.py`
scores it on the bench records unmodified**. It is stdlib `http.server` on purpose: no
fastapi/uvicorn to install on a box that is already awkward about kernels. One decision per
request under a lock, since a forward is not re-entrant; batching would raise throughput and
would not change a single returned probability.

## What was validated, and how

| check | result |
|---|---|
| `examples/system_one.py` on the IT arm | 3 question types answered, rejection channel present |
| `GET /health`, `GET /v1/models` | `{"ok": true}`, base and temperature reported |
| `POST /v1/systemone` | full TypeSafe shape, `none` included |
| malformed body | `400` with the error, not a 500 |
| unknown route | `404` |
| 703-token decision-v7 row | `200`, correct answer, process healthy |
| `eval/eval_systemone_http.py`, 120 rows, concurrency 2 | **0 failures**, acc 0.865, Brier 0.208, ECE 0.082 |
| error path, rows over `--max-seq` | `400` naming the token count and the limit |

The last client run is the important one: it is the repo's own bench harness, unmodified,
driving the server over HTTP and scoring with the same `data/metrics.py` as everything else.

## The RX 6600M sequence ceiling is a *training* limit

`scripts/rocm_env.sh` documents a hard ceiling of ~195 tokens on this card for **training** —
Qwen 3.5's gated-deltanet attention autotunes into a bf16 kernel gfx1030 cannot compile. It is
easy to over-generalise that to inference. It does not apply: the 703-token row and the
120-row client run above were both served on this machine. So do not cap `--max-seq` on the
card out of caution; decision-v7 rows run to p100 ~742 and are served fine.

## Choosing an arm

| | `lumierenoir/klev-0.8b` | `lumierenoir/klev-0.8b` + `/base` |
|---|---|---|
| base | `Qwen/Qwen3.5-0.8B` (IT) | `Qwen/Qwen3.5-0.8B-Base` (kev's) |
| decision-v7 dev / transfer-v4 | 0.8120 / 0.6636 | 0.8065 / 0.6283 |
| temperature to serve at | 2.2974 | 2.2449 |

Both are 4-bit, 0.89 GB of weights. Pick the IT arm unless you need to be comparable with
kev-0.8B, in which case use `base/` — see `docs/m13-qwen35-08b-base.md` for why the base matters
more than it looks.
