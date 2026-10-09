#!/usr/bin/env python3
"""Encode every trajectory patch with a FROZEN backbone and cache the features.

For each case and each requested level k, writes
    trajectories/features/<anchor_set>/<backbone>/L<k>/<case_id>.npy   (N_anchors, D) float16
row-aligned with trajectories/anchors/<anchor_set>/<case_id>.csv.

Levels are stored separately so the staged plan (L5-L4 -> +L3 -> +L2 ...) only
ever extracts the new level; a case/level whose file exists is skipped, so
jobs are resumable. Any backbone from src/backbones.py works; different levels
may use different backbones (train_trajectory.py --level-backbones).

Examples:
    # first check: 3 cases, coarse levels only
    python extract_trajectory_features.py --backbone imagenet --levels 5 4 --max-cases 3
    # SLURM array shard (shard/num-shards default to SLURM_ARRAY_TASK_ID/COUNT)
    python extract_trajectory_features.py --backbone mocov2 --levels 5 4 3 2 1 0 --shard 3 --num-shards 8
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from src.backbones import BACKBONE_NAMES, build_backbone, get_spec
from src.trajectory.io import (
    DEFAULT_IMAGES_DIR,
    DEFAULT_TRAJ_ROOT,
    ROOT,
    anchors_dir,
    atomic_save_npy,
    features_dir,
    load_anchor_config,
    wsi_path,
    write_json,
)
from src.trajectory.wsi import ALL_LEVELS, TiledPyramid
from src.utils import setup_logger


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True, choices=BACKBONE_NAMES)
    p.add_argument("--levels", type=int, nargs="+", required=True, choices=list(ALL_LEVELS))
    p.add_argument("--anchor-set", default="grid_L2_p224")
    p.add_argument("--traj-root", default=str(DEFAULT_TRAJ_ROOT))
    p.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    p.add_argument("--pretrained-dir", default=str(ROOT / "pretrained"))
    p.add_argument("--max-native-level", type=int, default=4,
                   help="Levels above this are synthesized from it (default 4: L5 synthesized for ALL slides)")
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--batch-anchors", type=int, default=64, help="anchors per GPU batch (x levels images)")
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--no-amp", dest="amp", action="store_false", help="fp32 forward (default: fp16 autocast)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data: one item = one anchor's patches at the requested levels (uint8).
# ---------------------------------------------------------------------------

_READERS: dict = {}  # per-worker-process cache of open slides (never opened in the parent)


def _reader(path: str, max_native_level: int) -> TiledPyramid:
    reader = _READERS.get(path)
    if reader is None:
        while len(_READERS) >= 2:
            _READERS.pop(next(iter(_READERS))).close()
        reader = TiledPyramid(path, max_native_level=max_native_level)
        _READERS[path] = reader
    return reader


class TrajectoryPatchDataset(Dataset):
    def __init__(self, jobs: list[dict], patch_size: int, max_native_level: int):
        self.jobs = jobs  # [{case_id, path, anchors: DataFrame, levels: [...]}]
        self.patch_size = patch_size
        self.max_native_level = max_native_level
        self.index = [(j, a) for j, job in enumerate(jobs) for a in range(len(job["anchors"]))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j, a = self.index[i]
        job = self.jobs[j]
        row = job["anchors"].iloc[a]
        try:
            patches = _reader(job["path"], self.max_native_level).read_trajectory(
                float(row["x"]), float(row["y"]), job["levels"], self.patch_size)
            ok = True
        except Exception as exc:  # reported per case by the main process, never silently kept
            logging.getLogger("extract").error(f"{job['case_id']} anchor {a}: {exc!r}")
            patches = np.zeros((len(job["levels"]), self.patch_size, self.patch_size, 3), np.uint8)
            ok = False
        return {"patches": torch.from_numpy(patches), "job": j, "anchor": a, "ok": ok}


class PerCaseBatchSampler(Sampler):
    """Consecutive anchors of ONE case per batch: raster order keeps the
    tile cache hot, and a case is complete as soon as its last batch arrives."""

    def __init__(self, dataset: TrajectoryPatchDataset, batch_anchors: int):
        self.batches = []
        start = 0
        for job in dataset.jobs:
            n = len(job["anchors"])
            for s in range(0, n, batch_anchors):
                self.batches.append(list(range(start + s, start + min(s + batch_anchors, n))))
            start += n

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _check_or_write_manifest(level_dir: Path, manifest: dict, logger) -> None:
    import json

    path = level_dir / "manifest.json"
    keys = ["backbone", "anchor_set", "level", "patch_size", "max_native_level", "feature_dim",
            "input_size", "normalization_mean", "normalization_std"]
    if path.exists():
        with open(path) as f:
            old = json.load(f)
        diff = {k: (old.get(k), manifest[k]) for k in keys if old.get(k) != manifest[k]}
        if diff:
            raise RuntimeError(f"{path} was written with different settings {diff}; refusing to mix features.")
    else:
        write_json(path, manifest)
        logger.info(f"wrote {path}")


def main():
    args = parse_args()
    shard = args.shard if args.shard is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_shards = args.num_shards if args.num_shards is not None else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    levels = sorted(set(args.levels), reverse=True)

    out_root = Path(args.traj_root) / "features" / args.anchor_set / args.backbone
    logger = setup_logger(out_root / "logs" / f"extract_L{''.join(map(str, levels))}_shard{shard}of{num_shards}.log",
                          name="extract")
    logger.info(f"args={vars(args)} shard={shard}/{num_shards}")

    anchor_cfg = load_anchor_config(args.traj_root, args.anchor_set)
    patch_size = int(anchor_cfg["patch_size"])
    adir = anchors_dir(args.traj_root, args.anchor_set)
    all_cases = args.cases or sorted(p.stem for p in adir.glob("case_*.csv"))
    if args.max_cases:
        all_cases = all_cases[: args.max_cases]
    cases = all_cases[shard::num_shards]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    spec = get_spec(args.backbone)
    backbone, feature_dim, info = build_backbone(args.backbone, args.pretrained_dir, logger)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    mean = torch.tensor(spec.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(spec.std, device=device).view(1, 3, 1, 1)
    if patch_size != spec.input_size:
        logger.warning(f"patch_size={patch_size} != {args.backbone} input size {spec.input_size}: resizing on GPU")

    for lvl in levels:
        _check_or_write_manifest(features_dir(args.traj_root, args.anchor_set, args.backbone, lvl), {
            "backbone": args.backbone, "anchor_set": args.anchor_set, "level": lvl, "patch_size": patch_size,
            "max_native_level": args.max_native_level, "feature_dim": feature_dim, "input_size": spec.input_size,
            "normalization_mean": list(spec.mean), "normalization_std": list(spec.std), "dtype": "float16",
            "amp": args.amp, "created": datetime.now().isoformat(timespec="seconds"),
            "backbone_info": {k: v for k, v in info.items() if k not in ("missing_keys", "unexpected_keys")},
        }, logger)

    jobs = []
    for case_id in cases:
        todo = [lvl for lvl in levels if args.overwrite or not (
            features_dir(args.traj_root, args.anchor_set, args.backbone, lvl) / f"{case_id}.npy").exists()]
        if not todo:
            continue
        anchors = pd.read_csv(adir / f"{case_id}.csv")
        if len(anchors) == 0:
            logger.warning(f"{case_id}: 0 anchors, skipped")
            continue
        jobs.append({"case_id": case_id, "path": str(wsi_path(case_id, args.images_dir)),
                     "anchors": anchors[["x", "y"]], "levels": todo})
    n_patches = sum(len(j["anchors"]) * len(j["levels"]) for j in jobs)
    logger.info(f"{len(cases)} cases in shard, {len(jobs)} with work left, {n_patches:,} patches to encode")
    if not jobs:
        return

    ds = TrajectoryPatchDataset(jobs, patch_size, args.max_native_level)
    loader = DataLoader(ds, batch_sampler=PerCaseBatchSampler(ds, args.batch_anchors), num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=False,
                        prefetch_factor=4 if args.num_workers > 0 else None)

    buffers: dict[int, dict] = {}
    t_start = t_case = time.time()
    done_patches = 0
    for batch in loader:
        j = int(batch["job"][0])
        job = jobs[j]
        buf = buffers.setdefault(j, {"feats": {lvl: [] for lvl in job["levels"]}, "n": 0, "failed": 0})

        x = batch["patches"].to(device, non_blocking=True)  # (B, L, P, P, 3) uint8
        B, L = x.shape[:2]
        x = x.view(B * L, patch_size, patch_size, 3).permute(0, 3, 1, 2).float().div_(255.0)
        if patch_size != spec.input_size:
            x = F.interpolate(x, size=(spec.input_size, spec.input_size), mode="bilinear", antialias=True,
                              align_corners=False)
        x = (x - mean) / std
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                             enabled=args.amp and device.type == "cuda"):
            feats = backbone(x)
        feats = feats.float().view(B, L, -1).cpu().numpy().astype(np.float16)
        if not np.isfinite(feats).all():
            raise RuntimeError(f"{job['case_id']}: non-finite features (fp16 overflow?) -- rerun with --no-amp")
        for li, lvl in enumerate(job["levels"]):
            buf["feats"][lvl].append(feats[:, li])
        buf["n"] += B
        buf["failed"] += int((~batch["ok"]).sum())
        done_patches += B * L

        if buf["n"] == len(job["anchors"]):
            if buf["failed"]:
                logger.error(f"{job['case_id']}: {buf['failed']} anchors failed to read -> NOT saved (will retry on resume)")
            else:
                for lvl in job["levels"]:
                    arr = np.concatenate(buf["feats"][lvl])
                    atomic_save_npy(features_dir(args.traj_root, args.anchor_set, args.backbone, lvl) / f"{job['case_id']}.npy", arr)
            dt = time.time() - t_case
            elapsed = time.time() - t_start
            logger.info(f"[{j + 1}/{len(jobs)}] {job['case_id']}: {buf['n']} anchors x L{job['levels']} "
                        f"in {dt:.1f}s | overall {done_patches / elapsed:.0f} patches/s, "
                        f"ETA {(n_patches - done_patches) / max(done_patches / elapsed, 1e-9) / 60:.1f} min")
            del buffers[j]
            t_case = time.time()

    logger.info(f"done: {done_patches:,} patches in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
