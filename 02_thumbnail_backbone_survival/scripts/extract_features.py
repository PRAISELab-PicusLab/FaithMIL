#!/usr/bin/env python3
"""Stage 1 of the linear-probe protocol: one frozen forward pass per thumbnail.

For a given backbone, runs all 508 thumbnails through the frozen backbone with
the SAME deterministic preprocessing train.py uses at val/test time (resize
longest side + white pad to the backbone's input size, the backbone's own
normalization) and saves the embeddings to

    features/<backbone>.npz   (case_id: str[N], features: float32[N, D])
    features/<backbone>.json  (provenance: dim, normalization, versions, ...)

Because the backbone is frozen, these embeddings are exactly what the linear
Cox head would see every epoch; computing them once lets stage 2
(scripts/linear_probe_survival.py) fit the head to convergence in seconds.

Fully offline (reads the local HF cache). Run on a GPU node:
    sbatch slurm/linear_probe.slurm <backbone>
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backbones import build_backbone, get_spec  # noqa: E402
from src.dataset import LeopardThumbnailDataset, load_folds  # noqa: E402
from src.utils import set_global_seed, setup_logger  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True)
    p.add_argument("--splits-csv", default="splits/folds.csv")
    p.add_argument("--pretrained-dir", default="pretrained")
    p.add_argument("--output-dir", default="features")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true", help="Disable fp16 autocast (on by default on GPU)")
    p.add_argument("--force", action="store_true", help="Overwrite existing features")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{args.backbone}.npz"
    meta_path = out_dir / f"{args.backbone}.json"
    if npz_path.exists() and not args.force:
        print(f"{npz_path} already exists -- pass --force to recompute.")
        return

    logger = setup_logger(out_dir / f"{args.backbone}.log", name=f"extract_{args.backbone}")
    set_global_seed(42, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    logger.info(f"backbone={args.backbone} device={device} amp={use_amp}")

    spec = get_spec(args.backbone)
    backbone, feature_dim, info = build_backbone(args.backbone, args.pretrained_dir, logger)
    backbone = backbone.to(device).eval()
    for param in backbone.parameters():
        param.requires_grad = False

    df = load_folds(args.splits_csv)
    ds = LeopardThumbnailDataset(df, image_size=spec.input_size, backbone=args.backbone, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")

    case_ids, feats = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    f = backbone(x)
            else:
                f = backbone(x)
            feats.append(f.float().cpu().numpy())
            case_ids.extend(batch["case_id"])
    features = np.concatenate(feats).astype(np.float32)
    elapsed = time.time() - t0

    if features.shape != (len(df), feature_dim):
        raise RuntimeError(f"feature matrix shape {features.shape}, expected {(len(df), feature_dim)}")
    if not np.isfinite(features).all():
        raise RuntimeError(f"{np.sum(~np.isfinite(features))} non-finite values in features (try --no-amp)")
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError("duplicate case_id in extracted features")

    np.savez_compressed(npz_path, case_id=np.array(case_ids), features=features)
    meta = {
        "backbone": args.backbone,
        "n_cases": len(case_ids),
        "feature_dim": feature_dim,
        "input_size": spec.input_size,
        "normalization_mean": list(spec.mean),
        "normalization_std": list(spec.std),
        "preprocessing": "ResizePad (longest side -> input_size, white pad), no augmentation",
        "amp_fp16": use_amp,
        "checkpoint": info.get("checkpoint"),
        "description": spec.description,
        "feature_stats": {
            "mean_abs": float(np.abs(features).mean()),
            "std_across_cases_mean": float(features.std(axis=0).mean()),
            "n_constant_dims": int((features.std(axis=0) < 1e-8).sum()),
        },
        "extraction_seconds": round(elapsed, 1),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "host": platform.node(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"saved {features.shape} -> {npz_path} in {elapsed:.1f}s")
    logger.info(f"feature stats: {meta['feature_stats']}")


if __name__ == "__main__":
    main()
