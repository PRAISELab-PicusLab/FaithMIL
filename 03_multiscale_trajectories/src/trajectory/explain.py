"""Explanations of a patient's risk in terms of trajectories and levels.

Shapley values over trajectories
--------------------------------
The model's patient risk for any SUBSET S of trajectories is well defined
(attention re-normalises over the subset):

    v(S) = sum_{k in S} w_k s_k / sum_{k in S} w_k,   w_k = exp(g_k),   v({}) = b

with b = mean training risk (the "average patient"). The Shapley value phi_k
is the average marginal change in risk when trajectory k joins a random
coalition. Properties that matter here:
  * efficiency: sum_k phi_k = R - b exactly (the attribution map explains the
    whole deviation of this patient from the average patient);
  * sign: phi_k > 0 pushes risk up, phi_k < 0 is protective -- the "positive
    and negative weights" of the design, with a game-theoretic meaning;
  * removal semantics: v(S) is literally the model run on fewer regions, the
    same operation the challenge's perturbation test (progressively removing
    top-attributed regions) performs.
Monte Carlo over permutations is O(N) per permutation with prefix sums, and
antithetic (reversed) permutations halve the variance -- ~0.1 s for 5000
trajectories x 256 permutations. For an ensemble of fold models, the Shapley
value of the averaged model is the average of the Shapley values (linearity).

Level occlusion
---------------
Delta_l = R - R(level l replaced by its training-set mean feature, for every
trajectory): how much of this patient's risk is carried by magnification l.
"""
from __future__ import annotations

import numpy as np
import torch


def shapley_trajectories(scores, logits, baseline: float, n_permutations: int = 256, seed: int = 0,
                         pooling: str = "attention") -> np.ndarray:
    s = torch.as_tensor(np.asarray(scores), dtype=torch.float64)
    g = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    n = s.numel()
    if n == 0:
        return np.zeros(0)
    w = torch.ones_like(s) if pooling == "mean" else torch.exp(g - g.max())
    gen = torch.Generator().manual_seed(seed)
    half = max(1, n_permutations // 2)
    perms = torch.argsort(torch.rand(half, n, generator=gen), dim=1)
    perms = torch.cat([perms, perms.flip(1)])  # antithetic pairs
    num = torch.cumsum((w * s)[perms], dim=1)
    den = torch.cumsum(w[perms], dim=1)
    v = num / den
    prev = torch.cat([torch.full((perms.shape[0], 1), float(baseline), dtype=torch.float64), v[:, :-1]], dim=1)
    phi = torch.zeros_like(v).scatter_(1, perms, v - prev).mean(0)
    return phi.numpy()


def attention_contributions(scores, weights, baseline: float) -> np.ndarray:
    """a_k (s_k - b): the exact first-order decomposition of R - b with the
    attention weights held fixed (cheap, deterministic companion to Shapley)."""
    return np.asarray(weights) * (np.asarray(scores) - baseline)


@torch.no_grad()
def level_occlusion(model, feats: dict[int, torch.Tensor], chunk: int = 4096) -> dict[int, float]:
    full = model.bag_risk(feats, chunk)
    out = {}
    for l in model.levels:
        mean = getattr(model, f"level_mean_L{l}").to(feats[l].device, feats[l].dtype)
        occluded = dict(feats)
        occluded[l] = mean.expand_as(feats[l])
        out[l] = full - model.bag_risk(occluded, chunk)
    return out
