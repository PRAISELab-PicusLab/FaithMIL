"""Shapley values of the patches of one patient.

The risk of any subset S of patches is well defined (the attention is re-normalised over the subset):

    v(S) = sum_{c in S} w_c s_c / sum_{c in S} w_c,    w_c = exp(g_c),    v({}) = b

with b the mean risk of the reference patients. The Shapley value phi_c is the average marginal change of the risk
when patch c joins a random coalition; sum_c phi_c = R - b exactly. Permutations are sampled with their reversed
copies (antithetic), and the prefix sums make one permutation O(n).
"""
from __future__ import annotations

import numpy as np
import torch


def shapley(scores, logits, baseline: float, n_permutations: int = 256, seed: int = 0) -> np.ndarray:
    s = torch.as_tensor(np.asarray(scores), dtype=torch.float64)
    g = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    n = s.numel()
    if n == 0:
        return np.zeros(0)
    w = torch.exp(g - g.max())
    gen = torch.Generator().manual_seed(seed)
    half = max(1, n_permutations // 2)
    perms = torch.argsort(torch.rand(half, n, generator=gen), dim=1)
    perms = torch.cat([perms, perms.flip(1)])  # antithetic pairs
    v = torch.cumsum((w * s)[perms], dim=1) / torch.cumsum(w[perms], dim=1)
    prev = torch.cat([torch.full((perms.shape[0], 1), float(baseline), dtype=torch.float64), v[:, :-1]], dim=1)
    return torch.zeros_like(v).scatter_(1, perms, v - prev).mean(0).numpy()


def first_order(scores, weights, baseline: float) -> np.ndarray:
    """a_c (s_c - b): exact first-order decomposition of R - b with the attention weights held fixed."""
    return np.asarray(weights) * (np.asarray(scores) - baseline)
