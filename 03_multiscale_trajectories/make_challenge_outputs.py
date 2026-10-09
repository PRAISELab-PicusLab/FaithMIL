#!/usr/bin/env python3
"""Produce the four OWL outputs for cases predicted by a trained run.

For every case in the chosen folds' TEST split (out-of-fold predictions, so
the outputs are what the model would say on unseen patients):

    <out>/<case_id>/time-to-recurrence.json   float, years (RMST, see src/trajectory/survival.py)
    <out>/<case_id>/risk-group.json           "low" | "medium" | "high"
    <out>/<case_id>/attribution-map.tif       uint8 multiresolution TIFF, tissue-mask spacing
    <out>/<case_id>/report.pdf                <= 5 pages
    <out>/summary.csv

The file names are placeholders until the challenge publishes its container
interface (leaderboard opens 2026-10-16); only the writer names need to change.

Examples:
    python make_challenge_outputs.py --run-dir results/trajectory/imagenet_transformer_attention_L54 --max-cases 3
    python make_challenge_outputs.py --run-dir results/trajectory/<run> --folds 0,1 --map-mode signed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.trajectory.attribution_map import encode_uint8, render_map, values_to_grid, write_pyramid_tiff
from src.trajectory.io import DEFAULT_IMAGES_DIR, DEFAULT_MASKS_DIR, anchors_dir, mask_path, wsi_path
from src.trajectory.report import CaseExplanation, write_report
from src.trajectory.survival import BreslowModel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--folds", default="all", help="'all', 'holdout' or comma-separated fold indices")
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--out-dir", default=None, help="default: <run-dir>/challenge_outputs")
    p.add_argument("--map-mode", choices=["abs", "positive", "signed"], default="abs")
    p.add_argument("--map-values", choices=["shapley", "contributions"], default="shapley")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--no-map", action="store_true")
    p.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    p.add_argument("--masks-dir", default=str(DEFAULT_MASKS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_root = Path(args.out_dir) if args.out_dir else run_dir / "challenge_outputs"
    if args.folds == "all":
        split_dirs = sorted(run_dir.glob("fold_*")) or [run_dir / "holdout"]
    elif args.folds == "holdout":
        split_dirs = [run_dir / "holdout"]
    else:
        split_dirs = [run_dir / f"fold_{f}" for f in args.folds.split(",")]

    rows, n_done = [], 0
    for sdir in split_dirs:
        if not (sdir / "model.pt").exists():
            print(f"skip {sdir}: no model.pt", file=sys.stderr)
            continue
        ckpt = torch.load(sdir / "model.pt", map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        breslow = BreslowModel.from_dict(ckpt["breslow"])
        adir = anchors_dir(cfg["data"]["traj_root"], cfg["data"]["anchor_set"])
        with open(adir / "config.json") as f:
            patch_size = int(json.load(f)["patch_size"])
        summary = pd.read_csv(adir / "summary.csv").set_index("case_id")
        preds = pd.read_csv(sdir / "test_predictions.csv")
        if args.cases:
            preds = preds[preds["case_id"].isin(args.cases)]

        for _, p in preds.iterrows():
            if args.max_cases and n_done >= args.max_cases:
                break
            c = p["case_id"]
            npz_path = sdir / "trajectory_outputs" / f"{c}.npz"
            if not npz_path.exists():
                print(f"skip {c}: no {npz_path}", file=sys.stderr)
                continue
            z = np.load(npz_path)
            anchors = pd.read_csv(adir / f"{c}.csv")
            info = summary.loc[c]
            out = out_root / c
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "time-to-recurrence.json", "w") as f:
                json.dump(float(p["predicted_time_years"]), f)
            with open(out / "risk-group.json", "w") as f:
                json.dump(str(p["risk_group"]), f)

            values = z["shapley"] if args.map_values == "shapley" else z["contributions"]
            if not args.no_map:
                grid = encode_uint8(values_to_grid(anchors["gx"], anchors["gy"], values,
                                                   int(info["grid_rows"]), int(info["grid_cols"])), args.map_mode)
                img, pages = render_map(grid, int(info["stride_l0"]), mask_path(c, args.masks_dir))
                write_pyramid_tiff(out / "attribution-map.tif", img, pages)
                del img
            if not args.no_report:
                ex = CaseExplanation(
                    case_id=c, risk=float(z["risk"]), baseline=float(z["baseline"]),
                    predicted_time=float(p["predicted_time_years"]), risk_group=str(p["risk_group"]),
                    anchors=anchors, scores=z["scores"], weights=z["weights"], shapley=z["shapley"],
                    occlusion={int(l): float(d) for l, d in zip(z["occlusion_levels"], z["occlusion_delta"])},
                    levels=list(ckpt["levels"]), breslow=breslow, train_risks=np.asarray(ckpt["train_risks"]),
                    risk_group_thresholds=ckpt["risk_group_thresholds"], patch_size=patch_size,
                    stride_l0=int(info["stride_l0"]),
                    n_train_cases=ckpt.get("n_train_cases"), n_train_events=ckpt.get("n_train_events"),
                    run_info={"run": cfg["run_name"], "fold": ckpt["fold"], "encoders": ckpt["level_backbones"]},
                )
                write_report(ex, wsi_path(c, args.images_dir), out / "report.pdf")
            rows.append({"case_id": c, "fold": p["fold"], "risk_score": p["risk_score"],
                         "time_to_recurrence_years": p["predicted_time_years"], "risk_group": p["risk_group"],
                         "event": p["event"], "follow_up_years": p["follow_up_years"]})
            n_done += 1
            print(f"[{n_done}] {c}: time={p['predicted_time_years']:.2f}y group={p['risk_group']} -> {out}", flush=True)

    if rows:
        out_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_root / "summary.csv", index=False)
    print(f"{n_done} cases written to {out_root}")


if __name__ == "__main__":
    main()
