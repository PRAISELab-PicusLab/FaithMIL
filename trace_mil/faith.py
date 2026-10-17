"""Faithfulness-aware training terms (Eq. 5 of the paper). They are used for training only: the model at test time is
the same as without them.

    L = L_Cox(R) + lambda_s L_Cox(R_top) + lambda_n Var(R_rest) / Var(R) + lambda_e H(a) / log t

R_top is the risk recomputed on the fraction `topk` of the patches with the highest attention (attention re-normalised
over them), R_rest the risk of the remaining patches; the variances are over the patients of the mini-batch, H(a) is the
entropy of the attention over the t patches of a patient. The terms can be switched on after `start_epoch` Cox-only
epochs (those epochs are excluded from checkpoint selection).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .losses import cox_ph_loss
from .models import seg_softmax


@dataclass
class FaithConfig:
    suff_cox: float = 0.0   # lambda_s: the top patches alone must rank the patients (sufficiency)
    nec_var: float = 0.0    # lambda_n: without the top patches the risk should no longer separate patients (necessity)
    ent: float = 0.0        # lambda_e: concentrated attention
    topk: float = 0.1       # fraction q of the patches ranked by attention
    start_epoch: int = 0    # Cox-only warm-up epochs

    @property
    def any(self) -> bool:
        return bool(self.suff_cox or self.nec_var or self.ent)

    def asdict(self) -> dict:
        return asdict(self)


def faith_loss(cfg: FaithConfig, g, s, w, seg, risk, n_pat: int, time, event) -> torch.Tensor:
    """g attention logits, s instance scores, w attention, seg patient of every instance, risk (n_pat,) the patient
    risks, time / event the follow-up of the patients of the batch."""
    dev = g.device
    terms = []
    if cfg.ent:
        n = torch.zeros(n_pat, device=dev).index_add(0, seg, torch.ones_like(w))
        H = torch.zeros(n_pat, device=dev).index_add(0, seg, -w * torch.log(w.clamp_min(1e-12)))
        terms.append(cfg.ent * (H / torch.log(n.clamp_min(2))).mean())
    if cfg.suff_cox or cfg.nec_var:
        key = g.detach()
        top = torch.zeros_like(w, dtype=torch.bool)
        for b in range(n_pat):  # top fraction of every patient's instances by attention
            idx = torch.nonzero(seg == b).squeeze(1)
            k = max(1, min(len(idx) - 1, int(math.ceil(cfg.topk * len(idx)))))
            top[idx[torch.topk(key[idx], k).indices]] = True
        big = torch.finfo(g.dtype).min / 4

        def sub_risk(mask):
            ww = seg_softmax(torch.where(mask, g, torch.full_like(g, big)), seg, n_pat)
            return torch.zeros(n_pat, device=dev).index_add(0, seg, ww * s)

        if cfg.suff_cox:
            terms.append(cfg.suff_cox * cox_ph_loss(sub_risk(top), time, event))
        if cfg.nec_var:
            terms.append(cfg.nec_var * sub_risk(~top).var() / (risk.var() + 1e-6))
    return sum(terms)
