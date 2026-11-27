"""TRACE L1 model for the OWL container: tile reading, DINO L1 features, the fold x seed ABMIL ensemble and its exact
attributions.

The model (trained in exp_ctx_l1l2/results/l1only_abmil_dino, 10 folds x 3 seeds on OWL RUMC + TCGA-PRAD) is gated-
attention MIL over the 0.22 mm L1 tiles. For one member m with attention logits g_i and tile scores s_i = w.h_i + b:

    R_m = sum_i a_i s_i,  a = softmax(g),   z_m = (R_m - mu_m) / sd_m   (mu, sd: risks of its training patients)
    Z   = mean_m z_m                                                      (the ensemble risk, higher = earlier BCR)

so Z = sum_i phi_i exactly, with phi_i = mean_m a_mi (s_mi - mu_m) / sd_m (the tile contribution of the report), and the
effect of removing any set of tiles can be computed exactly from (g, s): this is what ranks the attribution heatmap.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from trace_mil.layout import PATCH
from trace_mil.wsi import TiledPyramid, patch_top_left

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # as the frozen DINO feature extraction
DELETION_STEPS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)  # the fractions of the challenge faithfulness test


# ---------------------------------------------------------------------------- MIL ensemble

def _adapter(D, d):
    return nn.Sequential(nn.LayerNorm(D), nn.Dropout(0.0), nn.Linear(D, d), nn.GELU(), nn.Dropout(0.0))


class Gate(nn.Module):
    def __init__(self, d, a):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(d, a), nn.Tanh(), nn.Dropout(0.0))
        self.U = nn.Sequential(nn.Linear(d, a), nn.Sigmoid(), nn.Dropout(0.0))
        self.w = nn.Linear(a, 1)

    def forward(self, h):
        return self.w(self.V(h) * self.U(h)).squeeze(-1)


class ABMIL(nn.Module):
    """exp_combined/tree/mil_baselines.ABMIL (level "l1") in eval mode, same parameter names."""

    def __init__(self, D=384, d=64, a=32):
        super().__init__()
        self.ad, self.gate, self.head = _adapter(D, d), Gate(d, a), nn.Linear(d, 1)

    def tile_terms(self, X):
        """Attention logits g and tile scores s (risk = sum softmax(g) * s)."""
        h = self.ad(X.float())
        return self.gate(h), self.head(h).squeeze(-1)


class Ensemble:
    def __init__(self, model_dir: Path, device):
        self.device = device
        self.members, mus, sds = [], [], []
        for fp in sorted((model_dir / "ensemble").glob("fold_*/model_seed*.pt")):
            ck = torch.load(fp, map_location="cpu", weights_only=True)
            m = ABMIL(ck["D"], ck["hidden_dim"], ck["attention_dim"])
            m.load_state_dict(ck["state_dict"], strict=True)
            self.members.append(m.to(device).eval())
            mus.append(float(ck["mu"]))
            sds.append(float(ck["sd"]))
        if not self.members:
            raise FileNotFoundError(f"no ensemble members in {model_dir / 'ensemble'}")
        self.mu, self.sd = np.array(mus), np.array(sds)

    @torch.no_grad()
    def tile_terms(self, X: torch.Tensor):
        """(n_members, n_tiles) attention logits and tile scores, float64 on the CPU."""
        g, s = zip(*(m.tile_terms(X.to(self.device)) for m in self.members))
        return torch.stack(g).double().cpu().numpy(), torch.stack(s).double().cpu().numpy()


def ensemble_z(g, s, mu, sd, keep=None) -> float:
    """Ensemble risk Z of the tiles in `keep` (bool mask over tiles; all tiles if None)."""
    if keep is not None:
        g, s = g[:, keep], s[:, keep]
    a = np.exp(g - g.max(axis=1, keepdims=True))
    a /= a.sum(axis=1, keepdims=True)
    return float(np.mean(((a * s).sum(axis=1) - mu) / sd))


def contributions(g, s, mu, sd) -> np.ndarray:
    """phi_i: the exact additive share of tile i in Z (sum_i phi_i == Z)."""
    a = np.exp(g - g.max(axis=1, keepdims=True))
    a /= a.sum(axis=1, keepdims=True)
    return np.mean(a * (s - mu[:, None]) / sd[:, None], axis=0)


def leave_one_out(g, s, mu, sd) -> np.ndarray:
    """Delta_i = Z(without tile i) - Z, exact, for every tile at once."""
    e = np.exp(g - g.max(axis=1, keepdims=True))
    E, ES = e.sum(axis=1, keepdims=True), (e * s).sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_wo = (ES - e * s) / (E - e)
    r_wo = np.where(np.isfinite(r_wo), r_wo, (ES / E))  # a slide of one tile: nothing left, no change
    r = ES / E
    return np.mean((r_wo - r) / sd[:, None], axis=0)


def deletion_auc(order, g, s, mu, sd, steps=DELETION_STEPS) -> float:
    """Average |Delta Z| when the first x% of `order` is removed (trapezoid from 0, as the challenge metric)."""
    n = len(order)
    z0 = ensemble_z(g, s, mu, sd)
    ys = []
    for f in steps:
        k = int(np.ceil(round(f * n, 6)))
        keep = np.ones(n, bool)
        keep[order[:k]] = False
        ys.append(abs(ensemble_z(g, s, mu, sd, keep) - z0) if keep.any() else 0.0)
    x, y = np.r_[0.0, steps], np.r_[0.0, ys]
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2) / (x[-1] - x[0]))


def rank01(v) -> np.ndarray:
    """Ranks scaled to 0..1 (ties: average rank): the order of `v` without its heavy tail, so that a few extreme
    tiles do not make the rest of a map look flat."""
    import pandas as pd
    v = np.asarray(v, np.float64)
    if len(v) < 2:
        return np.ones(len(v))
    return (pd.Series(v).rank(method="average").to_numpy() - 1) / (len(v) - 1)


def relevance(g, s, mu, sd) -> tuple[np.ndarray, str, dict]:
    """The tile relevance of the attribution heatmap: how much removing the tile moves the prediction.

    The leave-one-out effect Delta_i is exact for one tile. Removals in opposite directions cancel, so the map ranks the
    tiles by Delta in the direction that moves this patient's prediction most when several are removed together (chosen
    on the exact deletion curve of the model), with |Delta| as the alternative. Returns (score, rule, aucs); higher score
    = more relevant."""
    d = leave_one_out(g, s, mu, sd)
    candidates = {"towards_higher_risk": d, "towards_lower_risk": -d, "absolute": np.abs(d)}
    aucs = {k: deletion_auc(np.argsort(-v, kind="stable"), g, s, mu, sd) for k, v in candidates.items()}
    rule = max(aucs, key=aucs.get)
    return candidates[rule], rule, aucs


# ---------------------------------------------------------------------------- tiles and features

class L1Reader:
    """uint8 (n, 224, 224, 3) L1 tiles of the children of a set of parents, pixel-identical to the training features:
    the children of a parent are cut from one native-L1 read of its 896 px window (as the training feature
    extraction). One TiledPyramid per thread."""

    def __init__(self, wsi_path: Path):
        self.path = wsi_path
        self._local = threading.local()

    def _rd(self) -> TiledPyramid:
        rd = getattr(self._local, "rd", None)
        if rd is None:
            rd = self._local.rd = TiledPyramid(self.path, max_native_level=4, tile_cache_tiles=64, strict=False)
        return rd

    def parent_tiles(self, pxy, cxy_list):
        rd = self._rd()
        out = []
        for (cx, cy), cxy in zip(pxy, cxy_list):
            x0, y0 = int(patch_top_left(cx, 1, 4 * PATCH)), int(patch_top_left(cy, 1, 4 * PATCH))
            crop = rd.read_region(1, x0, y0, 4 * PATCH, 4 * PATCH)
            ox = patch_top_left(cxy[:, 0], 1, PATCH) - x0
            oy = patch_top_left(cxy[:, 1], 1, PATCH) - y0
            out.extend(crop[oy[j]:oy[j] + PATCH, ox[j]:ox[j] + PATCH] for j in range(len(cxy)))
        return np.stack(out)


def make_vit():
    """trace_mil.dino.make_vit (ViT-S/14, 384-d, random initialisation; weights come only from our DINO checkpoint),
    without its torchvision import."""
    import timm
    return timm.create_model("vit_small_patch16_224", pretrained=False, patch_size=14, img_size=224, num_classes=0,
                             dynamic_img_size=True)


def load_encoder(path: Path, device):
    vit = make_vit()
    vit.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return vit.to(device).eval()


@torch.no_grad()
def extract_features(vit, wsi_path: Path, parents, children, device, threads: int = 8, chunk: int = 16,
                     batch: int = 256, log=print) -> torch.Tensor:
    """(n_children, 384) float16 features in the row order of `children`."""
    pid = children["parent_id"].to_numpy()
    order = np.argsort(pid, kind="stable")  # children grouped by parent, as in training
    counts = np.bincount(pid, minlength=len(parents))
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    cxy = children[["x", "y"]].to_numpy(np.float64)[order]
    pxy = parents[["x", "y"]].to_numpy(np.float64)
    groups = [p for p in range(len(parents)) if counts[p]]
    chunks = [groups[i:i + chunk] for i in range(0, len(groups), chunk)]
    reader = L1Reader(wsi_path)

    def read(sel):
        return reader.parent_tiles(pxy[sel], [cxy[starts[p]:starts[p] + counts[p]] for p in sel])

    use_amp = device.type == "cuda"
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    feats = torch.empty((len(children), 384), dtype=torch.float16)
    k, t0 = 0, time.time()

    def ordered_reads(ex):
        """Chunk tiles in order, with at most 2 x threads chunks read ahead: Executor.map would read the whole slide
        ahead of a slower GPU and hold it in RAM (~15 GB for the largest slides)."""
        pending, it = deque(), iter(chunks)
        for sel in islice(it, 2 * threads):
            pending.append(ex.submit(read, sel))
        while pending:
            tiles = pending.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(ex.submit(read, nxt))
            yield tiles

    with ThreadPoolExecutor(threads) as ex:
        for i, tiles in enumerate(ordered_reads(ex)):
            for b in range(0, len(tiles), batch):
                x = torch.from_numpy(tiles[b:b + batch]).to(device, non_blocking=True).permute(0, 3, 1, 2).float()
                x = (x.div_(255.0) - mean) / std
                with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                    f = vit(x)
                feats[k:k + len(f)] = f.float().cpu().half()
                k += len(f)
            if (i + 1) % 100 == 0:
                log(f"  features {k}/{len(children)} ({time.time() - t0:.0f}s)")
    out = torch.empty_like(feats)
    out[torch.from_numpy(order)] = feats  # grouped -> children row order
    return out


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
