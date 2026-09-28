# M15 — predictions without unsloth

Date: 2026-09-28
Script: `eval/check_backend_parity.py` · Code: `model/infer.py` (`backend="plain"`)

## Question

Unsloth is what makes the 4-bit *fine-tune* fit one consumer GPU. Is it needed to make
*predictions*? A decision is one forward pass, so the load path is the only thing that has to
change. `model/infer.py` grew a second base loader: `_load_base_plain` (transformers +
bitsandbytes + peft) alongside `_load_base_unsloth` (`FastModel`), selected by
`load_klev(..., backend=...)`, `KLEV_BACKEND=plain`, or `--backend plain` on `klev-system-one`
and `klev-serve`. Everything after the base load — delimiters, `head.pt`, the readout,
`answer()` — is shared, so the two paths run identical arithmetic.

## What had to be solved

1. **Pillow.** `AutoProcessor` for Gemma 4 pulls in the vision stack. A decision row is text, so
   the plain loader falls back to `AutoTokenizer` when the processor's dependency is missing.
2. **`Gemma4ClippableLinear`.** The published adapter's `target_modules` are the usual projection
   names, and unsloth's training-time model applied them everywhere those names appear —
   including Gemma 4's vision/audio towers, whose projections are wrapper modules peft refuses
   (`only nn.Linear, Linear4bit, Embedding, Conv*, MultiheadAttention are supported`). The text
   stack itself is plain `Linear4bit`. `_attach_adapter_plain` therefore pins the targets to the
   language-model modules and lets the tower weights go unused, which is what a text-only
   deployment does anyway.

## Parity

Same checkpoint, same 200 CoSt-BR records, temperature 1.0 (raw readout), 5 rows skipped in both
for the state cap. The comparison is per-row, not just an accuracy number.

| | klev-e4b (Gemma 4) | klev-0.8b (Qwen3.5) |
|---|---|---|
| records | 195 | 58 |
| accuracy, unsloth / plain | **0.410 / 0.410** | **0.317 / 0.317** |
| choice agreement | 98.5 % | 98.3 % |
| max abs probability delta | 0.043 | 0.042 |
| load time, unsloth / plain | 91.2 s / 73.2 s | 42.4 s / 20.8 s |
| answer time (200 / 60 rows) | 48.1 s / 42.6 s | 14.3 s / 10.7 s |

Both arms: identical accuracy, ~98 % identical choices, the same rows skipped, and the residual
comes from kernels rather than from the weights (unsloth's patched attention/quantised matmuls vs
plain sdpa + bnb). Plain is *faster* to load and marginally faster per answer.

## Conclusion

**Unsloth is a training dependency, not an inference one.** A consumer of a published checkpoint
needs `torch`, `transformers>=5.5`, `peft`, `bitsandbytes`, `pydantic` and nothing from unsloth.
`pyproject.toml` moves it into the `train` extra; the default install is unsloth-free, and
`pip install "klev[train] @ git+https://github.com/felipepenhorate/klev.git"` brings it back for
fine-tuning or teacher caching.

Caveat worth keeping: the 98 % agreement is not 100 %, so a published number reproduced through
the plain path can differ by a row or two per hundred. Benchmarks in the model cards were
produced with the unsloth path; the plain path is for serving and integration, and
`eval/check_backend_parity.py --compare` is the check if a number has to be reproduced exactly.

## Reproduce

```bash
# any env with unsloth
python eval/check_backend_parity.py --backend unsloth --ckpt /mnt/f/distill_jev_runs/main \
    --limit 200 --out /tmp/backends/unsloth.json
# an env without it
python eval/check_backend_parity.py --backend plain --ckpt /mnt/f/distill_jev_runs/main \
    --limit 200 --out /tmp/backends/plain.json
python eval/check_backend_parity.py --compare /tmp/backends/unsloth.json /tmp/backends/plain.json
```
