"""In-memory bank of cached trajectory features + bag sampling.

All features of the requested cases/levels are loaded once (float16, CPU).
ResNet50 features for all 508 slides x 6 levels are ~16 GB, 384-d ViT-S
features ~3 GB, so a whole CV run keeps everything in RAM and an epoch costs
seconds.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from src.trajectory.io import features_dir, load_features, load_manifest


class FeatureBank:
    def __init__(self, traj_root, anchor_set: str, case_ids, levels, level_backbones: dict[int, str],
                 logger: logging.Logger | None = None):
        log = logger or logging.getLogger(__name__)
        self.levels = sorted(levels, reverse=True)
        self.case_ids = list(case_ids)
        self.level_backbones = {int(l): level_backbones[int(l)] for l in self.levels}

        missing = [(c, l) for c in self.case_ids for l in self.levels
                   if not (features_dir(traj_root, anchor_set, self.level_backbones[l], l) / f"{c}.npy").exists()]
        if missing:
            by_level: dict[int, int] = {}
            for _, l in missing:
                by_level[l] = by_level.get(l, 0) + 1
            raise FileNotFoundError(
                f"{len(missing)} missing feature files (per level: {by_level}); e.g. {missing[:3]}. "
                "Run extract_trajectory_features.py for these levels first."
            )

        self.manifests = {l: load_manifest(traj_root, anchor_set, self.level_backbones[l], l) for l in self.levels}
        self.feats: dict[int, list[torch.Tensor]] = {l: [] for l in self.levels}
        self.n_traj = []
        for c in self.case_ids:
            arrs = {l: load_features(traj_root, anchor_set, self.level_backbones[l], l, c) for l in self.levels}
            sizes = {a.shape[0] for a in arrs.values()}
            if len(sizes) != 1:
                raise RuntimeError(f"{c}: levels have different numbers of trajectories {sizes}")
            for l, a in arrs.items():
                self.feats[l].append(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float16)))
            self.n_traj.append(sizes.pop())
        self.index = {c: i for i, c in enumerate(self.case_ids)}
        self.level_dims = {l: int(self.feats[l][0].shape[1]) for l in self.levels}
        n = np.array(self.n_traj)
        gb = sum(t.numel() * 2 for l in self.levels for t in self.feats[l]) / 1e9
        log.info(f"FeatureBank: {len(self.case_ids)} cases, levels {self.levels}, dims {self.level_dims}, "
                 f"trajectories/case min={n.min()} median={int(np.median(n))} max={n.max()}, {gb:.1f} GB")

    def bag(self, case_id: str, device) -> dict[int, torch.Tensor]:
        i = self.index[case_id]
        return {l: self.feats[l][i].to(device, non_blocking=True) for l in self.levels}

    def sample_batch(self, case_ids, bag_size: int, generator: torch.Generator, device):
        """Random subset of <= bag_size trajectories per case, zero-padded.
        Returns ({level: (B, K, D_l)}, mask (B, K))."""
        idxs = []
        for c in case_ids:
            n = self.n_traj[self.index[c]]
            idxs.append(torch.randperm(n, generator=generator)[:bag_size] if n > bag_size else torch.arange(n))
        K = max(len(ix) for ix in idxs)
        mask = torch.zeros(len(case_ids), K, dtype=torch.bool)
        out = {}
        for l in self.levels:
            buf = torch.zeros(len(case_ids), K, self.level_dims[l], dtype=torch.float16)
            for b, (c, ix) in enumerate(zip(case_ids, idxs)):
                buf[b, :len(ix)] = self.feats[l][self.index[c]][ix]
            out[l] = buf.to(device, non_blocking=True)
        for b, ix in enumerate(idxs):
            mask[b, :len(ix)] = True
        return out, mask.to(device)

    def level_means(self, case_ids) -> dict[int, torch.Tensor]:
        """Mean feature per level over all trajectories of `case_ids` (float32)."""
        means = {}
        for l in self.levels:
            total = torch.zeros(self.level_dims[l], dtype=torch.float64)
            count = 0
            for c in case_ids:
                f = self.feats[l][self.index[c]]
                total += f.double().sum(0)
                count += f.shape[0]
            means[l] = (total / max(count, 1)).float()
        return means
