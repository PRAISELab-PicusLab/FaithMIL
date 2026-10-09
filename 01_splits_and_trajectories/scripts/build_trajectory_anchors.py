#!/usr/bin/env python3
"""Build the anchor set (tissue-filtered grid of trajectory centres) per slide.

CPU only, no torch: reads WSI headers + one tissue-mask level per slide
(~1 s/slide), so it is light enough to run anywhere; the per-case CSVs are
resumable (existing ones are skipped unless --force).

Examples:
    # staged: 5 cases + QC figures first
    python scripts/build_trajectory_anchors.py --max-cases 5 --qc 5
    # then everything
    python scripts/build_trajectory_anchors.py --workers 16
    # denser grid (L1/L0 sampled 4x more often), capped per slide
    python scripts/build_trajectory_anchors.py --anchor-set grid_s1792_p224 --stride-l0 1792 --max-anchors 8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trajectory.anchors import AnchorConfig, generate_anchors  # noqa: E402
from src.trajectory.io import (  # noqa: E402
    DEFAULT_IMAGES_DIR,
    DEFAULT_MASKS_DIR,
    DEFAULT_TRAJ_ROOT,
    ROOT,
    anchors_dir,
    mask_path,
    wsi_path,
    write_json,
)
from src.trajectory.wsi import ALL_LEVELS, level_fov_um  # noqa: E402


def parse_min_tissue(text: str) -> dict:
    out = {}
    for item in filter(None, text.split(",")):
        lvl, thr = item.split(":")
        out[int(lvl)] = float(thr)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anchor-set", default=None, help="Name of the anchor set (default: derived from the grid)")
    p.add_argument("--patch-size", type=int, default=224)
    p.add_argument("--grid-level", type=int, default=2)
    p.add_argument("--stride-l0", type=int, default=None)
    p.add_argument("--min-tissue", default="0:0.5,1:0.25,2:0.1", help="level:fraction,... (others: > 0)")
    p.add_argument("--max-anchors", type=int, default=None)
    p.add_argument("--mask-level", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--splits-csv", default=str(ROOT / "splits" / "folds.csv"))
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    p.add_argument("--masks-dir", default=str(DEFAULT_MASKS_DIR))
    p.add_argument("--traj-root", default=str(DEFAULT_TRAJ_ROOT))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--qc", type=int, default=0, help="Write QC figures for the first N processed cases")
    p.add_argument("--force", action="store_true", help="Recompute cases and allow a changed config")
    return p.parse_args()


def _run_case(case_id, images_dir, masks_dir, cfg_dict, out_dir, force):
    out_csv = Path(out_dir) / f"{case_id}.csv"
    info_json = Path(out_dir) / "info" / f"{case_id}.json"
    if out_csv.exists() and info_json.exists() and not force:
        with open(info_json) as f:
            return json.load(f), "skipped"
    cfg = AnchorConfig.from_dict(cfg_dict)
    df, info = generate_anchors(case_id, wsi_path(case_id, images_dir), mask_path(case_id, masks_dir), cfg)
    df.to_csv(out_csv, index=False)
    write_json(info_json, info)
    return info, "done"


def write_qc_figure(case_id: str, images_dir, out_dir: Path, patch_size: int, seed: int = 0) -> None:
    """Anchors over the L4 image + one full trajectory (L5 -> L0) with the
    footprint of each finer patch drawn inside the coarser one."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from src.trajectory.wsi import TiledPyramid, level_downsample

    df = pd.read_csv(out_dir / f"{case_id}.csv")
    with TiledPyramid(wsi_path(case_id, images_dir)) as pyr:
        H4, W4 = pyr.shapes[4]
        thumb = pyr.read_region(4, 0, 0, W4, H4)
        rng = np.random.default_rng(seed)
        row = df.iloc[int(rng.integers(len(df)))] if len(df) else None
        traj = pyr.read_trajectory(row["x"], row["y"], list(reversed(ALL_LEVELS)), patch_size) if row is not None else None
        mpp = pyr.mpp_l0

    fig = plt.figure(figsize=(18, 7))
    ax = fig.add_axes([0.01, 0.05, 0.36, 0.9])
    ax.imshow(thumb)
    sc = ax.scatter(df["x"] / 256, df["y"] / 256, c=df["tissue_L0"], s=4, cmap="viridis", vmin=0, vmax=1)
    if row is not None:
        ax.plot(row["x"] / 256, row["y"] / 256, "r*", ms=14)
    ax.set_title(f"{case_id}: {len(df)} anchors (colour = L0 tissue fraction)")
    ax.axis("off")
    fig.colorbar(sc, ax=ax, fraction=0.03)

    if traj is not None:
        levels = list(reversed(ALL_LEVELS))
        for i, lvl in enumerate(levels):
            axp = fig.add_axes([0.39 + i * 0.1, 0.3, 0.095, 0.4])
            axp.imshow(traj[i])
            if i + 1 < len(levels):  # footprint of the next (finer) patch: centred, 1/4 of the side
                side = patch_size / 4
                axp.add_patch(mpatches.Rectangle((patch_size / 2 - side / 2, patch_size / 2 - side / 2), side, side,
                                                 fill=False, ec="red", lw=1.2))
            fov = level_fov_um(lvl, patch_size, mpp)
            axp.set_title(f"L{lvl}\n{fov / 1000:.2f} mm" if fov >= 1000 else f"L{lvl}\n{fov:.0f} um", fontsize=9)
            axp.axis("off")
        fig.text(0.39, 0.2, f"trajectory at anchor {int(row['anchor_id'])} (x={int(row['x'])}, y={int(row['y'])} L0 px); "
                 "tissue " + ", ".join(f"L{l}={row[f'tissue_L{l}']:.2f}" for l in levels), fontsize=9)
    (out_dir / "qc").mkdir(exist_ok=True)
    fig.savefig(out_dir / "qc" / f"{case_id}.png", dpi=90)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = AnchorConfig(
        patch_size=args.patch_size,
        grid_level=args.grid_level,
        stride_l0=args.stride_l0,
        min_tissue=parse_min_tissue(args.min_tissue),
        max_anchors=args.max_anchors,
        mask_level=args.mask_level,
        seed=args.seed,
    )
    name = args.anchor_set or (f"grid_s{cfg.stride}_p{cfg.patch_size}" if args.stride_l0 else f"grid_L{cfg.grid_level}_p{cfg.patch_size}")
    out_dir = anchors_dir(args.traj_root, name)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            old = AnchorConfig.from_dict(json.load(f)).to_dict()  # JSON turns int keys into str
        if old != cfg.to_dict() and not args.force:
            sys.exit(f"{cfg_path} exists with a DIFFERENT config:\n  old={old}\n  new={cfg.to_dict()}\n"
                     "Use a new --anchor-set name (or --force to rebuild this one).")
    write_json(cfg_path, cfg.to_dict())

    if args.cases:
        cases = args.cases
    else:
        cases = sorted(pd.read_csv(args.splits_csv)["case_id"].tolist())
    if args.max_cases:
        cases = cases[: args.max_cases]
    print(f"anchor set '{name}': stride={cfg.stride} L0 px ({cfg.stride * 0.2425:.0f} um), "
          f"patch={cfg.patch_size}, min_tissue={cfg.min_tissue}, max_anchors={cfg.max_anchors}; {len(cases)} cases")

    t0 = time.time()
    infos, n_done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_case, c, args.images_dir, args.masks_dir, cfg.to_dict(), str(out_dir), args.force): c
                for c in cases}
        for i, fut in enumerate(as_completed(futs), 1):
            info, status = fut.result()
            infos.append(info)
            n_done += status == "done"
            if i % 50 == 0 or i == len(cases):
                print(f"  {i}/{len(cases)} ({time.time() - t0:.0f}s)", flush=True)

    summary_path = out_dir / "summary.csv"
    summary = pd.DataFrame(infos)
    if summary_path.exists():  # merge with cases processed by earlier (partial) runs
        old = pd.read_csv(summary_path)
        summary = pd.concat([old[~old["case_id"].isin(summary["case_id"])], summary])
    summary = summary.sort_values("case_id").reset_index(drop=True)
    summary.to_csv(summary_path, index=False)

    n = summary.set_index("case_id").loc[cases, "n_anchors"]
    print(f"\nprocessed {n_done} new cases in {time.time() - t0:.0f}s")
    print(f"anchors per slide: min={n.min()} median={int(n.median())} mean={n.mean():.0f} max={n.max()} total={n.sum()}")
    print(f"capped slides: {int(summary.set_index('case_id').loc[cases, 'capped'].sum())}")
    for dim in (384, 2048):
        gb = n.sum() * len(ALL_LEVELS) * dim * 2 / 1e9
        print(f"feature store for these cases, 6 levels, D={dim} fp16: {gb:.1f} GB")

    for case_id in cases[: args.qc]:
        write_qc_figure(case_id, args.images_dir, out_dir, cfg.patch_size, seed=args.seed)
    if args.qc:
        print(f"QC figures: {out_dir / 'qc'}")


if __name__ == "__main__":
    main()
