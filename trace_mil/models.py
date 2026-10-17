"""Additive attention MIL.

    h_i = adapter(x_i)                      one adapter per view (LayerNorm, dropout, Linear, GELU, dropout)
    h_i = Dropout(GELU(W [h_i^1; h_i^2 (; h_i^3)]))     when there are context views (concatenation and fusion)
    s_i = v^T h_i + b                       local score of instance i
    a_i = softmax_i g(h_i)                  gated attention (Ilse et al., 2018) over the instances of the patient
    R   = sum_i a_i s_i                     patient risk, linear in the instance scores

The instance columns are split into the views by `dims` (e.g. [384, 384] = L1 | centred L2). One view gives the
single-level model of the paper (level of the view = L0, L1, L2 or L3).
"""
from __future__ import annotations

import torch
import torch.nn as nn


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


def build_model(config: dict) -> AttentionMIL:
    return AttentionMIL(config["dims"], config["hidden_dim"], config["attention_dim"], config["input_dropout"],
                        config["dropout"])
