"""Trajectory MIL survival model.

    per level l:   e_l (frozen feature, D_l)  --LevelAdapter_l-->  t_l (d)     "one encoder per level"
    trajectory k:  [t_L5 ... t_L0]  --aggregator (mlp | transformer | gru | lstm)-->  h_k (d)
    local score:   s_k = w^T h_k + b                                            (risk of that trajectory)
    reliability:   a_k = softmax_k( gated_attention(h_k) )                      (which trajectories to trust)
    patient risk:  R = sum_k a_k s_k                                            (higher = earlier BCR)

The encoders themselves are frozen and run once, offline (the features are
cached); the level adapters are the level-specific, trainable part. With a
linear score head, R = w^T (sum_k a_k h_k) + b, i.e. this is exactly gated
attention-MIL (Ilse et al. 2018) -- the per-trajectory scores come for free
and are what the explanations are built on (see explain.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.trajectory.wsi import MAX_LEVEL


class LevelAdapter(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, input_dropout: float, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(input_dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPAggregator(nn.Module):
    """Concatenation of the level tokens -> MLP (order is implicit in the position)."""

    def __init__(self, n_levels: int, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_levels * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d), nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (N, L, d)
        return self.net(tokens.flatten(1))


class TransformerAggregator(nn.Module):
    """[CLS] + level tokens with a learned embedding of the ABSOLUTE level id, so
    'L3' means the same thing whichever subset of levels a run uses."""

    def __init__(self, levels: list[int], d: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.register_buffer("level_ids", torch.tensor(levels, dtype=torch.long), persistent=False)
        self.level_embedding = nn.Embedding(MAX_LEVEL + 1, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=2 * d, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens + self.level_embedding(self.level_ids)[None]
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        return self.norm(self.encoder(x)[:, 0])


class RNNAggregator(nn.Module):
    """Reads the trajectory coarse -> fine like a pathologist zooming in; the
    state after the finest level summarises the whole trajectory."""

    def __init__(self, d: int, n_layers: int, dropout: float, cell: str = "gru"):
        super().__init__()
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(d, d, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(tokens)
        return self.norm(out[:, -1])


class GatedAttention(nn.Module):
    def __init__(self, d: int, attn_dim: int, dropout: float):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(d, attn_dim), nn.Tanh(), nn.Dropout(dropout))
        self.U = nn.Sequential(nn.Linear(d, attn_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.w(self.V(h) * self.U(h)).squeeze(-1)


class TrajectorySurvivalModel(nn.Module):
    def __init__(
        self,
        level_dims: dict[int, int],
        levels: list[int],
        hidden_dim: int = 128,
        aggregator: str = "transformer",
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.25,
        input_dropout: float = 0.1,
        level_dropout: float = 0.1,
        attention_dim: int = 64,
        pooling: str = "attention",
    ):
        super().__init__()
        self.levels = sorted(levels, reverse=True)  # coarse -> fine, always
        self.level_dims = {int(k): int(v) for k, v in level_dims.items()}
        self.pooling = pooling
        self.level_dropout = level_dropout
        d = hidden_dim

        self.adapters = nn.ModuleDict({f"L{l}": LevelAdapter(self.level_dims[l], d, input_dropout, dropout)
                                       for l in self.levels})
        # replaces a whole level token under level dropout (training only)
        self.mask_token = nn.Parameter(torch.zeros(d))
        if aggregator == "mlp":
            self.aggregator = MLPAggregator(len(self.levels), d, dropout)
        elif aggregator == "transformer":
            self.aggregator = TransformerAggregator(self.levels, d, n_heads, n_layers, dropout)
        elif aggregator in ("gru", "lstm"):
            self.aggregator = RNNAggregator(d, n_layers, dropout, cell=aggregator)
        else:
            raise ValueError(f"unknown aggregator '{aggregator}'")
        self.score_head = nn.Linear(d, 1)
        self.attention = GatedAttention(d, attention_dim, dropout)
        if pooling not in ("attention", "mean"):
            raise ValueError(f"unknown pooling '{pooling}'")

        # Training-set mean feature per level: the "neutral" input used by the
        # level-occlusion explanation (set by train_trajectory.py).
        for l in self.levels:
            self.register_buffer(f"level_mean_L{l}", torch.zeros(self.level_dims[l]))

    # ------------------------------------------------------------------
    def encode(self, feats: dict[int, torch.Tensor]) -> torch.Tensor:
        """feats[l]: (..., D_l) -> trajectory embeddings h: (..., d)."""
        lead = feats[self.levels[0]].shape[:-1]
        tokens = torch.stack([self.adapters[f"L{l}"](feats[l].reshape(-1, feats[l].shape[-1]).float())
                              for l in self.levels], dim=1)  # (N, L, d)
        if self.training and self.level_dropout > 0 and len(self.levels) > 1:
            drop = torch.rand(tokens.shape[:2], device=tokens.device) < self.level_dropout
            drop[drop.all(dim=1)] = False  # never drop every level of a trajectory
            tokens = torch.where(drop[..., None], self.mask_token.to(tokens.dtype), tokens)
        h = self.aggregator(tokens)
        return h.view(*lead, -1)

    def pool(self, scores: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor):
        """scores/logits/mask: (B, K). Returns (risk (B,), weights (B, K))."""
        if self.pooling == "mean":
            logits = torch.zeros_like(logits)
        logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        risk = (weights * scores.masked_fill(~mask, 0.0)).sum(dim=1)
        return risk, weights

    def forward(self, feats: dict[int, torch.Tensor], mask: torch.Tensor) -> dict:
        """feats[l]: (B, K, D_l), mask: (B, K) bool (True = real trajectory)."""
        h = self.encode(feats)
        scores = self.score_head(h).squeeze(-1)
        logits = self.attention(h)
        risk, weights = self.pool(scores, logits, mask)
        return {"risk": risk, "scores": scores, "logits": logits, "weights": weights}

    @torch.no_grad()
    def trajectory_outputs(self, feats: dict[int, torch.Tensor], chunk: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
        """Local scores s_k and attention logits g_k for ONE full bag
        (feats[l]: (N, D_l)), chunked to bound memory."""
        n = feats[self.levels[0]].shape[0]
        s, g = [], []
        for i in range(0, n, chunk):
            h = self.encode({l: f[i:i + chunk] for l, f in feats.items()})
            s.append(self.score_head(h).squeeze(-1))
            g.append(self.attention(h))
        return torch.cat(s), torch.cat(g)

    @torch.no_grad()
    def bag_risk(self, feats: dict[int, torch.Tensor], chunk: int = 4096) -> float:
        s, g = self.trajectory_outputs(feats, chunk)
        mask = torch.ones_like(s, dtype=torch.bool)
        risk, _ = self.pool(s[None], g[None], mask[None])
        return float(risk)


def build_trajectory_model(level_dims: dict[int, int], cfg: dict) -> TrajectorySurvivalModel:
    m = cfg["model"]
    return TrajectorySurvivalModel(
        level_dims=level_dims,
        levels=cfg["data"]["levels"],
        hidden_dim=m["hidden_dim"],
        aggregator=m["aggregator"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        dropout=m["dropout"],
        input_dropout=m["input_dropout"],
        level_dropout=m["level_dropout"],
        attention_dim=m["attention_dim"],
        pooling=m["pooling"],
    )
