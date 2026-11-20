"""Additive attention MIL.

    h_i = adapter(x_i)                      one adapter per view (LayerNorm, dropout, Linear, GELU, dropout)
    h_i = Dropout(GELU(W [h_i^1; h_i^2 (; h_i^3)]))     when there are context views (concatenation and fusion)
    s_i = v^T h_i + b                       local score of instance i
    a_i = softmax_i g(h_i)                  gated attention (Ilse et al., 2018) over the instances of the patient
    R   = sum_i a_i s_i                     patient risk, linear in the instance scores

The instance columns are split into the views by `dims` (e.g. [384, 384] = L1 | centred L2). One view gives the
single-level model of the paper (level of the view = L0, L1, L2 or L3).

Established MIL baselines (same features, folds, Cox loss and training protocol; only the aggregator changes):

    mean       mean pooling of the adapted L1 patch embeddings, linear risk head (view children)
    clam_sb    CLAM single-branch (Lu et al., 2021) over the L1 patches: gated attention pooling and an instance
               clustering loss on the k = 8 most / least attended patches of each patient (weight 0.3) (view children)
    transmil   TransMIL (Shao et al., 2021) over the L2 tiles: class token, two transformer layers and the PPEG
               convolutional position encoding on the square-padded token grid (view parents)

They do not decompose the risk into patch terms, so they are used for the C-index only.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BASELINES = {"mean": ["children"], "clam_sb": ["children"], "transmil": ["parents"]}  # arch -> required views


def adapter(D: int, d: int, p_in: float, p: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(D), nn.Dropout(p_in), nn.Linear(D, d), nn.GELU(), nn.Dropout(p))


class Gate(nn.Module):
    """Gated attention: w^T (tanh(V h) * sigmoid(U h))."""

    def __init__(self, d: int, a: int, p: float):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(d, a), nn.Tanh(), nn.Dropout(p))
        self.U = nn.Sequential(nn.Linear(d, a), nn.Sigmoid(), nn.Dropout(p))
        self.w = nn.Linear(a, 1)

    def forward(self, h):
        return self.w(self.V(h) * self.U(h)).squeeze(-1)


def seg_softmax(logits: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of the logits within each segment (patient)."""
    m = torch.full((n,), float("-inf"), device=logits.device).scatter_reduce(0, seg, logits, "amax", include_self=True)
    e = torch.exp(logits - m[seg])
    den = torch.zeros(n, device=logits.device).index_add(0, seg, e)
    return e / den[seg]


class AttentionMIL(nn.Module):
    def __init__(self, dims, hidden_dim: int = 64, attention_dim: int = 32, input_dropout: float = 0.25,
                 dropout: float = 0.25):
        super().__init__()
        self.dims = list(dims)
        d = hidden_dim
        self.adapters = nn.ModuleList([adapter(D, d, input_dropout, dropout) for D in self.dims])
        # the layers are created in the order of the original experiments (same initialisation for a given seed)
        if len(self.dims) == 1:
            self.gate = Gate(d, attention_dim, dropout)
            self.score = nn.Linear(d, 1)
            self.fuse = None
        else:
            self.fuse = nn.Sequential(nn.Linear(len(self.dims) * d, d), nn.GELU(), nn.Dropout(dropout))
            self.score = nn.Linear(d, 1)
            self.gate = Gate(d, attention_dim, dropout)

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        parts, a = [], 0
        for ad, D in zip(self.adapters, self.dims):
            parts.append(ad(X[:, a:a + D].float()))
            a += D
        return parts[0] if self.fuse is None else self.fuse(torch.cat(parts, dim=1))

    def forward(self, X: torch.Tensor, seg: torch.Tensor, n_pat: int):
        h = self.embed(X)
        g = self.gate(h)
        s = self.score(h).squeeze(-1)
        w = seg_softmax(g, seg, n_pat)
        risk = torch.zeros(n_pat, device=h.device).index_add(0, seg, w * s)
        return risk, {"g": g, "s": s, "w": w}


class MeanMIL(nn.Module):
    def __init__(self, D: int, d: int, p_in: float, p: float):
        super().__init__()
        self.ad, self.head = adapter(D, d, p_in, p), nn.Linear(d, 1)

    def forward(self, X, seg, n_pat):
        h = self.ad(X.float())
        cnt = torch.zeros(n_pat, device=h.device).index_add(0, seg, torch.ones_like(seg, dtype=torch.float))
        H = torch.zeros(n_pat, h.shape[1], device=h.device).index_add(0, seg, h) / cnt.clamp_min(1)[:, None]
        return self.head(H).squeeze(-1), {}


class CLAMSB(nn.Module):
    def __init__(self, D: int, d: int, a: int, p_in: float, p: float, inst_k: int = 8, inst_w: float = 0.3):
        super().__init__()
        self.inst_k, self.inst_w = inst_k, inst_w
        self.ad = adapter(D, d, p_in, p)
        self.gate, self.head = Gate(d, a, p), nn.Linear(d, 1)
        self.inst = nn.Linear(d, 2)

    def forward(self, X, seg, n_pat):
        h = self.ad(X.float())
        g = self.gate(h)
        w = seg_softmax(g, seg, n_pat)
        H = torch.zeros(n_pat, h.shape[1], device=h.device).index_add(0, seg, w[:, None] * h)
        out = {}
        if self.training:  # instance clustering: top-k attended patches = positive, bottom-k = negative pseudo-labels
            losses = []
            for i in range(n_pat):
                idx = torch.nonzero(seg == i).squeeze(1)
                k = min(self.inst_k, len(idx) // 2)
                if k < 1:
                    continue
                order = torch.argsort(g[idx], descending=True)
                sel = torch.cat([idx[order[:k]], idx[order[-k:]]])
                y = torch.cat([torch.ones(k), torch.zeros(k)]).long().to(h.device)
                losses.append(F.cross_entropy(self.inst(h[sel]), y))
            if losses:
                out["inst_loss"] = self.inst_w * torch.stack(losses).mean()
        return self.head(H).squeeze(-1), out


class PPEG(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.c7 = nn.Conv2d(d, d, 7, padding=3, groups=d)
        self.c5 = nn.Conv2d(d, d, 5, padding=2, groups=d)
        self.c3 = nn.Conv2d(d, d, 3, padding=1, groups=d)

    def forward(self, x, side: int):
        cls, tok = x[:, :1], x[:, 1:]
        B, N, C = tok.shape
        grid = tok.transpose(1, 2).reshape(B, C, side, side)
        grid = grid + self.c7(grid) + self.c5(grid) + self.c3(grid)
        return torch.cat([cls, grid.flatten(2).transpose(1, 2)], 1)


class TransMIL(nn.Module):
    def __init__(self, D: int, d: int, p_in: float, p: float, heads: int = 4):
        super().__init__()
        self.ad = adapter(D, d, p_in, p)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.l1 = nn.TransformerEncoderLayer(d, heads, 2 * d, p, batch_first=True, norm_first=True)
        self.l2 = nn.TransformerEncoderLayer(d, heads, 2 * d, p, batch_first=True, norm_first=True)
        self.ppeg, self.norm, self.head = PPEG(d), nn.LayerNorm(d), nn.Linear(d, 1)

    def forward(self, X, seg, n_pat):
        h = self.ad(X.float())
        risks = []
        for i in range(n_pat):
            t = h[seg == i]
            side = int(math.ceil(math.sqrt(len(t))))
            pad = side * side - len(t)  # the grid is completed by repeating the first tokens
            t = torch.cat([t, t[:pad]], 0) if pad <= len(t) else torch.cat([t, t.repeat(pad // len(t) + 1, 1)[:pad]], 0)
            x = self.l2(self.ppeg(self.l1(torch.cat([self.cls, t[None]], 1)), side))
            risks.append(self.head(self.norm(x[:, 0])).squeeze(-1))
        return torch.cat(risks), {}


def build_model(config: dict) -> nn.Module:
    arch, dims = config.get("arch", "additive"), config["dims"]
    d, a, p_in, p = config["hidden_dim"], config["attention_dim"], config["input_dropout"], config["dropout"]
    if arch == "additive":
        return AttentionMIL(dims, d, a, p_in, p)
    if arch == "mean":
        return MeanMIL(dims[0], d, p_in, p)
    if arch == "clam_sb":
        return CLAMSB(dims[0], d, a, p_in, p)
    if arch == "transmil":
        return TransMIL(dims[0], d, p_in, p)
    raise ValueError(f"unknown arch {arch!r}")
