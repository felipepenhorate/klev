#!/usr/bin/env python
"""Garbage-candidate mechanics probe (standalone, no model load).

Train a PointerHead(garbage=True) on synthetic hidden states: half the samples carry an
answer aligned with option c (decide = option_c * 3 + noise), half are "no option fits"
(decide = -mean(options) * 3 + noise, i.e. anti-aligned with every option). Checks:

  1. at init the garbage bias (-4) keeps p_garbage ≈ 1-2% and the option ranking;
  2. training routes the mass: p_garbage low on answerable samples, high on garbage ones;
  3. gradients reach `g` and `g_bias`;
  4. the garbage score is exactly permutation-invariant (option order permutes the K
     logits and leaves the garbage logit unchanged).

Run:  /home/penhfel/unsloth_uv/bin/python tools/probe_garbage.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from model.head import PointerHead

D, K, DP, N = 64, 4, 32, 1024
SEP, NOISE, STEPS = 3.0, 0.4, 600


def make_options(seed=1):
    return F.normalize(torch.randn(K, D, generator=torch.Generator().manual_seed(seed)), dim=-1)


def make_data(n, options, garbage_frac=0.5, seed=0):
    rng = torch.Generator().manual_seed(seed)
    is_garbage = torch.rand(n, generator=rng) < garbage_frac
    labels = torch.randint(0, K, (n,), generator=rng)
    decide = options[labels] * SEP + NOISE * torch.randn(n, D, generator=rng)
    decide[is_garbage] = -options.mean(0, keepdim=True) * SEP + NOISE * torch.randn(int(is_garbage.sum()), D, generator=rng)
    return decide, labels, is_garbage


def batch_logits(head, decide, options):
    q = decide.shape[0]
    owner = torch.arange(q).repeat_interleave(K)
    opts = options.unsqueeze(0).expand(q, K, D).reshape(q * K, D)
    z, zg = head.many_with_garbage(decide, opts, owner)
    return torch.cat([z.reshape(q, K), zg[:, None]], -1)


def main():
    torch.manual_seed(0)
    options = make_options()
    train_decide, train_y, train_g = make_data(N, options, seed=1)
    test_decide, test_y, test_g = make_data(512, options, seed=2)
    head = PointerHead(D, dp=DP, garbage=True)

    with torch.no_grad():
        probs = torch.softmax(batch_logits(head, test_decide, options), -1)
    init = {"p_garbage_mean": float(probs[:, K].mean()), "p_garbage_max": float(probs[:, K].max())}

    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-3)
    for _ in range(STEPS):
        idx = torch.randint(0, N, (64,))
        logits = batch_logits(head, train_decide[idx], options)
        target = torch.zeros(len(idx), K + 1)
        answerable = ~train_g[idx]
        target[answerable, train_y[idx][answerable]] = 1.0
        target[~answerable, K] = 1.0
        loss = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_grads = {"g": float(head.g.grad.norm()), "g_bias": float(head.g_bias.grad)}

    with torch.no_grad():
        probs = torch.softmax(batch_logits(head, test_decide, options), -1)
        p_g, argmax = probs[:, K], probs.argmax(-1)
        answerable = ~test_g
        results = {
            "init": init,
            "final_loss": float(loss),
            "p_garbage_answerable_mean": float(p_g[answerable].mean()),
            "p_garbage_garbage_mean": float(p_g[test_g].mean()),
            "answerable_accuracy": float((argmax[answerable] == test_y[answerable]).float().mean()),
            "garbage_detected": float((argmax[test_g] == K).float().mean()),
            "distribution_sums": float(probs.sum(-1).mean()),
            "grads_seen": final_grads,
        }

    # permutation invariance: permuting the option table permutes the K logits, garbage unchanged
    perm = torch.randperm(K, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        z1 = batch_logits(head, test_decide[:32], options)
        z2 = batch_logits(head, test_decide[:32], options[perm])
    results["permutation_invariant"] = bool(torch.allclose(z2[:, :K], z1[:, :K][:, perm]) and torch.allclose(z2[:, K], z1[:, K]))

    checks = {
        "init_garbage_small": init["p_garbage_max"] < 0.1,
        "routing_works": results["p_garbage_garbage_mean"] > 0.5 and results["p_garbage_answerable_mean"] < 0.1,
        "accuracy_kept": results["answerable_accuracy"] > 0.95,
        "gradients_flow": final_grads["g"] > 0 and abs(final_grads["g_bias"]) > 0,
        "permutation_invariant": results["permutation_invariant"],
    }
    payload = {"checks": checks, "results": results}
    print(json.dumps(payload, indent=2))
    out = ROOT / "runs" / "probe_garbage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    failed = [name for name, ok in checks.items() if not ok]
    print(f"[probe] {'ALL CHECKS PASSED' if not failed else 'FAILED: ' + ', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
