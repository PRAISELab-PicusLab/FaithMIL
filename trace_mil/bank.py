"""In-memory feature bank: the instances of every patient with their groups, and the batching of patients.

An instance is a tile of the first view of a model. Instances are organised in groups: the L1 children of one L2 tile
(views children / ctx2 / ctx3), or single tiles (views parents / grid0 / grid3, every tile is its own group).
Training draws `max_groups` random groups per patient and uses all their instances (bag-level augmentation that keeps
the cost independent of the slide size); evaluation uses every instance.

When several views are given (children + ctx2 [+ ctx3]) their features are concatenated column-wise, so that every
instance row is [L1 | centred L2 | centred L3].
"""
from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .layout import FEATURE_VIEWS, Layout

TREE_VIEWS = ("children", "ctx2", "ctx3")


@dataclass
class CaseBank:
    X: torch.Tensor        # (n, D) float16, instances sorted by group
    starts: torch.Tensor   # (n_groups,) first instance of every group
    counts: torch.Tensor   # (n_groups,) instances per group
    order: np.ndarray      # X[k] is row order[k] of the anchor file (or of the pool)
    rows: np.ndarray | None = None  # pooled banks: rows of the full feature file that X holds (in file order)


def check_views(views):
    views = list(views)
    unknown = [v for v in views if v not in FEATURE_VIEWS]
    if unknown:
        raise ValueError(f"unknown views {unknown}; choose from {list(FEATURE_VIEWS)}")
    if len(views) > 1 and (views[0] != "children" or any(v not in TREE_VIEWS for v in views)):
        raise ValueError("several views are only supported as children + ctx2 [+ ctx3]")
    if len(views) > 1 and views != [v for v in TREE_VIEWS if v in views]:
        raise ValueError("views must be given in the order children, ctx2, ctx3")
    return views


def pool_rows(case: str, n: int, k: int, seed: int = 0) -> np.ndarray:
    """Fixed random subset of at most k rows of a case (the training pool of levels with too many tiles)."""
    if n <= k:
        return np.arange(n)
    rng = np.random.default_rng([seed, zlib.crc32(case.encode())])
    return np.sort(rng.choice(n, k, replace=False))


def load_bank(layout: Layout, cases, encoder: str, views, pool: int | None = None, pool_seed: int = 0,
              logger=None) -> tuple[dict[str, CaseBank], list[int]]:
    """case -> CaseBank. Returns the bank and the feature width of every view (to split the columns again)."""
    views = check_views(views)
    anchors = layout.anchors_of_view(views[0])
    dims = [int(json.loads((layout.features(encoder, v) / "manifest.json").read_text())["feature_dim"]) for v in views]
    bank, t0 = {}, time.time()
    for i, case in enumerate(cases):
        parts = []
        for v, d in zip(views, dims):
            f = np.load(layout.features(encoder, v) / f"{case}.npy", mmap_mode="r" if pool else None)
            if f.ndim != 2 or f.shape[1] != d:
                raise RuntimeError(f"{case}/{v}: feature shape {f.shape}, expected (*, {d})")
            parts.append(f)
        n = parts[0].shape[0]
        if any(p.shape[0] != n for p in parts):
            raise RuntimeError(f"{case}: views have different numbers of rows {[p.shape[0] for p in parts]}")
        rows = None
        if pool:
            rows = pool_rows(case, n, pool, pool_seed)
            parts = [np.ascontiguousarray(p[rows]) for p in parts]
        X = np.concatenate(parts, axis=1)
        if views[0] in TREE_VIEWS:
            gid = pd.read_csv(anchors / f"{case}.csv", usecols=["parent_id"])["parent_id"].to_numpy()
            if len(gid) != n:
                raise RuntimeError(f"{case}: {len(gid)} anchors but {n} feature rows")
            n_groups = int(gid.max()) + 1
        else:
            gid = np.arange(len(X))
            n_groups = len(X)
        order = np.argsort(gid, kind="stable")
        counts = np.bincount(gid, minlength=n_groups)
        if (counts == 0).any():
            raise RuntimeError(f"{case}: group without instances")
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        bank[case] = CaseBank(torch.from_numpy(np.ascontiguousarray(X[order])), torch.from_numpy(starts),
                              torch.from_numpy(counts), order, rows)
        if logger and (i + 1) % 100 == 0:
            logger.info(f"  loaded {i + 1}/{len(cases)} slides ({time.time() - t0:.0f}s)")
    if logger:
        gb = sum(b.X.numel() * 2 for b in bank.values()) / 1e9
        logger.info(f"bank {encoder}/{'+'.join(views)}: {len(bank)} slides, {gb:.2f} GB, instances/slide median "
                    f"{int(np.median([len(b.X) for b in bank.values()]))} ({time.time() - t0:.0f}s)")
    return bank, dims


def assemble(bank, cases, max_groups: int | None, gen, device):
    """Concatenate a batch of patients -> (X, seg): the instance features and the patient (position in `cases`) of each."""
    xs, seg = [], []
    for b, case in enumerate(cases):
        cb = bank[case]
        n = len(cb.counts)
        if max_groups and n > max_groups:
            sel = torch.randperm(n, generator=gen)[:max_groups].sort().values
            idx = torch.cat([torch.arange(int(cb.starts[j]), int(cb.starts[j] + cb.counts[j])) for j in sel.tolist()])
            xs.append(cb.X[idx])
            seg.append(torch.full((len(idx),), b, dtype=torch.long))
        else:
            xs.append(cb.X)
            seg.append(torch.full((len(cb.X),), b, dtype=torch.long))
    return torch.cat(xs).to(device, non_blocking=True), torch.cat(seg).to(device, non_blocking=True)
