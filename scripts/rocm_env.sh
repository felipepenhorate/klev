#!/usr/bin/env bash
# ROCm environment for the gfx1032 (RX 6600M) box. Source before any unsloth run here:
#   source scripts/rocm_env.sh
#
# This is the recipe from Projects/Qwen_3_(0_6B)_Text.ipynb, which is the known-good
# reference run on this card (60 steps, batch 2x4, 72 s). It lives in one place because every
# one of these is load-bearing.
#
# HSA_OVERRIDE_GFX_VERSION=10.3.0 makes ROCm target the RX 6600M, which reports itself as
# gfx1032 but is covered by the gfx1030 code path.
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export CUDA_VISIBLE_DEVICES=0

# SEQUENCE LENGTH CEILING on this card. Not a dtype problem, and not a cache problem.
#
# Qwen3.5's gated-deltanet attention autotunes over block sizes, and past a certain length the
# search reaches a bf16 candidate that gfx1030 cannot compile, so the process aborts:
#   LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16
# The ceiling was measured, not guessed. Alpaca-prompt token lengths, same model, same script,
# same batch, same max_seq=512, empty Triton cache throughout:
#
#   2 abstracts x 45 words   p50 221  p90 253  p100 304   aborts
#   2 abstracts x 30 words   p50 190  p90 214  p100 238   aborts
#   1 abstract  x 45 words   p50 159  p90 177  p100 212   aborts
#   1 abstract  x 30 words   p50 144  p90 157  p100 177   trains
#
# The RumourEval adapter, which trains here, tops out at p100 195. So: keep the whole tokenized
# row under ~195 tokens. Nothing else moves it -- KLEV_DTYPE=fp16 does not, dropping
# UNSLOTH_FORCE_FLOAT32 does not, and clearing ~/.triton/cache does not help (a from-empty
# cache trains the short case in 119 s). Do not waste time re-testing those.
#
# Cost of the ceiling: it is a real constraint on what task this card can host. PubMedQA runs
# only with the evidence cut to 30 words of one abstract, and at that compression the adapter
# degenerates to predicting the majority class on all 200 test rows. Longer-context tasks need
# the 4080.

export ROCM_PATH=/opt/rocm
export HIP_PATH="${ROCM_PATH}/hip"
export PATH="${ROCM_PATH}/bin:${PATH}"
# `:-` on both: these run under `set -u` in the pipeline scripts, and neither is guaranteed
# to be set in a fresh shell. Mirrors os.environ.get(..., "") in the notebook.
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ROCM_PATH}/bin:${PATH}"

# Keep bf16 out of the LLVM intrinsic selection: the GPU has no bf16 fdot2, so Triton's fused
# kernels abort with
#   LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16
# This is the supported knob and it is what the notebook uses -- preferable to lying to
# torch.cuda.is_bf16_supported() the way .venv-e2b's sitecustomize does.
export TORCH_ROCM_AOTRITON_ENABLE_BF16=0

# torch.compile and unsloth's compile wrapper misbehave on this ROCm+gfx1032 stack; with
# them on, a 16-token forward took minutes of CPU-bound time at ~0% GPU utilisation.
export UNSLOTH_COMPILE_DISABLE=1

# Bypass unsloth's Triton vision patching, which does not come up cleanly on ROCm.
export UNSLOTH_DISABLE_FAST_GENERATION=1

# No flash-attn on ROCm; unsloth falls back to xformers/sdpa, which is fine.
export UNSLOTH_DISABLE_FAST_ATTENTION=1
export UNSLOTH_ENABLE_CAUSAL_CONV1D=0

# Native bf16 matmul is fine on gfx103x (measured: fwd+bwd finite). Only the fused Triton
# kernels are not, which is what TORCH_ROCM_AOTRITON_ENABLE_BF16=0 above addresses; keep
# UNSLOTH_USE_TRITON=0 anyway so unsloth does not route anything through them.
export UNSLOTH_USE_TRITON=0

# Gemma 4 and Qwen 3.5 are in unsloth's FORCE_FLOAT32 list: plain fp16 NaNs the grad_norm in
# the backward, so unsloth loads bf16 weights, computes matmuls in fp16 and holds the residual
# stream in float32. Required for those two families; must stay UNSET for Qwen3, which the
# notebook trains in plain fp16. Set it per family at the call site, not here.
# export UNSLOTH_FORCE_FLOAT32=1

# Do NOT set PYTORCH_ALLOC_CONF=expandable_segments:True here: the ROCm allocator rejects it
# ("expandable_segments not supported on this platform", torch/c10/hip/HIPAllocatorConfig.h).

export KLEV_RUNS="${KLEV_RUNS:-/home/feipe/Documentos/Projects/klev-runs}"
export PY="${PY:-/home/feipe/Documentos/Projects/.venv-e2b/bin/python}"
