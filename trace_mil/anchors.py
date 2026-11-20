"""Tile anchors: which tiles of which level are kept, from the tissue mask.

Tree (the instances of the proposed model and its L2 / L3 contexts)
    A level-2 tile (224 px = 3584 px of level 0, 0.87 mm) contains 4 x 4 non-overlapping level-1 tiles (224 px = 896
    px of level 0, 0.22 mm). A parent is kept if its tissue fraction is >= 0.1 and at least one of its children is
    kept; a child is kept if its tissue fraction is >= 0.5. Parents without children are dropped.

        anchors/tree/parents/<case>.csv    anchor_id, x, y, gx, gy, tissue_L2, n_children      (x, y = centre, level-0 px)
        anchors/tree/children/<case>.csv   anchor_id, x, y, parent_id, ci, cj, tissue_L1        (raster order)

Grid (single-level models at other levels)
    Cells of 224 * 4**k level-0 px aligned at (0, 0), kept if the tissue fraction is >= 0.5 for k <= 1 and >= 0.1
    otherwise; a slide without any kept cell keeps its most tissue-rich cell.

        anchors/grid_L<k>/<case>.csv       anchor_id, x, y, gx, gy, tissue

The tissue fraction is read on level 1 of the mask (~8 um/px); area outside the slide counts as background.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .layout import PATCH, Layout
from .wsi import TissueMask, read_wsi_header

S2 = PATCH * 16  # side of a level-2 tile in level-0 px
S1 = PATCH * 4   # side of a level-1 tile in level-0 px


def _mask(layout: Layout, case: str, mask_level: int):
    header = read_wsi_header(layout.slide(case))
    return header, TissueMask(layout.mask(case), level=mask_level, wsi_res_l0=header["res_l0"])


def tree_case(layout: Layout, case: str, t2: float, t1: float, mask_level: int):
    header, mask = _mask(layout, case, mask_level)
    W0, H0 = header["width_l0"], header["height_l0"]
    n_cols, n_rows = int(np.ceil(W0 / S2)), int(np.ceil(H0 / S2))
    gy, gx = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()
    x0, y0 = gx * S2, gy * S2
    tissue2 = mask.fraction(x0, y0, x0 + S2, y0 + S2)
    par = np.flatnonzero((tissue2 > 0) & (tissue2 >= t2))

    ci, cj = np.meshgrid(np.arange(4), np.arange(4), indexing="xy")
    ci, cj = ci.ravel(), cj.ravel()
    cx0 = (x0[par][:, None] + ci[None, :] * S1).ravel()
    cy0 = (y0[par][:, None] + cj[None, :] * S1).ravel()
    tissue1 = mask.fraction(cx0, cy0, cx0 + S1, cy0 + S1)
    keep = (tissue1 > 0) & (tissue1 >= t1)
    par_of_child = np.repeat(par, 16)

    kept_par, counts = np.unique(par_of_child[keep], return_counts=True)
    parent_id = {p: i for i, p in enumerate(kept_par)}
    parents = pd.DataFrame({
        "anchor_id": np.arange(len(kept_par)), "x": x0[kept_par] + S2 // 2, "y": y0[kept_par] + S2 // 2,
        "gx": gx[kept_par], "gy": gy[kept_par], "tissue_L2": np.round(tissue2[kept_par], 4), "n_children": counts})
    children = pd.DataFrame({
        "x": cx0[keep] + S1 // 2, "y": cy0[keep] + S1 // 2,
        "parent_id": [parent_id[p] for p in par_of_child[keep]],
        "ci": np.tile(ci, len(par))[keep], "cj": np.tile(cj, len(par))[keep], "tissue_L1": np.round(tissue1[keep], 4)})
    children = children.sort_values(["y", "x"]).reset_index(drop=True)  # raster order: neighbours share image tiles
    children.insert(0, "anchor_id", np.arange(len(children)))
    info = {"case_id": case, "width_l0": W0, "height_l0": H0, "n_parents": len(parents), "n_children": len(children)}
    return parents, children, info


def grid_case(layout: Layout, case: str, level: int, t: float, mask_level: int):
    header, mask = _mask(layout, case, mask_level)
    S = PATCH * 4 ** level
    n_cols, n_rows = int(np.ceil(header["width_l0"] / S)), int(np.ceil(header["height_l0"] / S))
    gy, gx = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()
    x0, y0 = gx * S, gy * S
    tissue = mask.fraction(x0, y0, x0 + S, y0 + S)
    keep = np.flatnonzero((tissue > 0) & (tissue >= t))
    fallback = len(keep) == 0
    if fallback:
        keep = np.array([int(np.argmax(tissue))])
    df = pd.DataFrame({"anchor_id": np.arange(len(keep)), "x": x0[keep] + S // 2, "y": y0[keep] + S // 2,
                       "gx": gx[keep], "gy": gy[keep], "tissue": np.round(tissue[keep], 4)})
    return df, {"case_id": case, "n_anchors": len(keep), "fallback": bool(fallback)}


def _tree_job(root, case, t2, t1, mask_level):
    return tree_case(Layout(root), case, t2, t1, mask_level)


def _grid_job(root, case, level, t, mask_level):
    return grid_case(Layout(root), case, level, t, mask_level)


def build(layout: Layout, cases, mode: str, level: int | None = None, min_tissue: float | None = None,
          mask_level: int = 1, workers: int = 8) -> pd.DataFrame:
    """Write the anchor CSVs of `cases`; returns the per-case summary."""
    t0, infos = time.time(), []
    if mode == "tree":
        t2, t1 = (0.1, 0.5) if min_tissue is None else (0.1, min_tissue)
        dp, dc = layout.tree_anchors("parents"), layout.tree_anchors("children")
        for d in (dp, dc):
            d.mkdir(parents=True, exist_ok=True)
        cfg = {"patch_size": PATCH, "min_tissue_l2": t2, "min_tissue_l1": t1, "mask_level": mask_level}
        (dp.parent / "config.json").write_text(json.dumps(cfg, indent=2))
        jobs = lambda ex: [ex.submit(_tree_job, str(layout.root), c, t2, t1, mask_level) for c in cases]  # noqa: E731
    else:
        t = (0.5 if level <= 1 else 0.1) if min_tissue is None else min_tissue
        out = layout.grid_anchors(level)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps({"patch_size": PATCH, "level": level, "min_tissue": t,
                                                     "mask_level": mask_level}, indent=2))
        jobs = lambda ex: [ex.submit(_grid_job, str(layout.root), c, level, t, mask_level) for c in cases]  # noqa: E731
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = jobs(ex)
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            if mode == "tree":
                parents, children, info = res
                parents.to_csv(dp / f"{info['case_id']}.csv", index=False)
                children.to_csv(dc / f"{info['case_id']}.csv", index=False)
            else:
                df, info = res
                df.to_csv(out / f"{info['case_id']}.csv", index=False)
            infos.append(info)
            if i % 50 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} slides ({time.time() - t0:.0f}s)", flush=True)
    summary = pd.DataFrame(infos).sort_values("case_id")
    (dp.parent if mode == "tree" else out).joinpath("summary.csv").write_text(summary.to_csv(index=False))
    return summary
