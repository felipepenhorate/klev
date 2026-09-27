"""Pointer head: readout over option boundary tokens, with an optional rejection candidate.

Vendored from kev/model.py (Apache-2.0, github.com/jaredpalmer/kev, PointerHead) so the
decision readout is byte-compatible with Kev's. `temperature` is applied in eval mode only;
a checkpoint carries the value fitted on its own development rows in `head.pt`.

The garbage candidate (opt-in, `garbage=True`) is the K+1-th competitor of the softmax:

    z_j = (k(h_opt_j) . q(h_decide)) * scale          # the K options, as in Kev
    z_g = (g . q(h_decide)) * scale + g_bias          # "the rest"

It is a learned direction in query space plus a bias. Because it reads only the `<decide>`
state (which attends to the state, the instruction and every option), it can raise itself
when none of the options fits, is exactly permutation-invariant (no option enters its score)
and costs one dot product. `g_bias` starts at -4 so an untrained head gives the garbage
candidate ≈1-2% of the mass and the option ranking is unchanged.

Targets (see data/format.py): a hard label is one-hot over the K options with 0 garbage;
an unanswerable record is garbage with 0 options; an ambiguous record can carry mass on
both. At serving, the K option probabilities are renormalized by `1 - p_garbage` and
`p_garbage` is returned as the `none` field, so the TypeSafe distribution still sums to 1.
"""
import math

import torch
import torch.nn as nn


class PointerHead(nn.Module):
    def __init__(self, d, dp=256, garbage=False, garbage_bias=-4.0):
        """dp = pointer dimension (head capacity knob). garbage = add the rejection candidate."""
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        self.temperature = 1.0
        self.garbage = bool(garbage)
        if self.garbage:
            self.g = nn.Parameter(torch.zeros(dp))
            self.g_bias = nn.Parameter(torch.tensor(float(garbage_bias)))

    def _temper(self, z):
        return z if self.training or self.temperature == 1.0 else z / self.temperature

    def score_garbage(self, h_decide):  # [d] or [Q,d] -> scalar or [Q] (garbage is a query-space direction)
        return (self.q(h_decide) @ self.g) * self.scale + self.g_bias

    def forward(self, h_decide, h_opts):  # [d], [K,d] -> logits [K], or [K+1] with garbage last
        z = self._temper((self.k(h_opts) @ self.q(h_decide)) * self.scale)
        if not self.garbage:
            return z
        return torch.cat([z, self._temper(self.score_garbage(h_decide)).reshape(1)])

    def many(self, h_decide, h_opts, owner):  # [Q,d], [sum K,d], question of each option [sum K] -> logits [sum K]
        z = self._temper((self.k(h_opts) * self.q(h_decide)[owner]).sum(-1) * self.scale)
        return z

    def many_with_garbage(self, h_decide, h_opts, owner):
        """> ([sum K] option logits, [Q] garbage logits) at one temperature."""
        return self.many(h_decide, h_opts, owner), self._temper(self.score_garbage(h_decide))
