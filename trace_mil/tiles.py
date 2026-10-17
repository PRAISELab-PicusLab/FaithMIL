"""Raw 224-px tiles of one level, for the self-supervised pre-training of that level's encoder.

    tiles/L<k>/<case>.npy   (n, 224, 224, 3) uint8, written last (the file's existence marks the case as done)

  level 2   every tile of anchors/tree/parents (anchor order)
  level 1   at most `cap` random tiles of anchors/tree/children (the chosen rows are stored in <case>_idx.npy)
  level 0   at most `cap` random tiles of anchors/grid_L0
  level 3   at most `cap` crops of 224 px at level 3, centred on random points of the grid_L3 cells (at most
            `per_cell` per cell), so that the crops are not aligned to the grid

The random choices are seeded with the case id.
"""
from __future__ import annotations

import os
import time
import zlib
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .layout import PATCH, Layout
from .wsi import TiledPyramid, level_downsample


def centres(layout: Layout, case: str, level: int, cap: int, per_cell: int):
    """(x, y) level-0 centres of the tiles to extract, and the anchor rows they come from (None for level 3)."""
    rng = np.random.default_rng(zlib.crc32(f"{case}_L{level}".encode()))
    if level == 2:
        a = pd.read_csv(layout.tree_anchors("parents") / f"{case}.csv", usecols=["x", "y"]).to_numpy(np.float64)
        return a, np.arange(len(a))
    if level in (0, 1):
        src = layout.tree_anchors("children") if level == 1 else layout.grid_anchors(0)
        a = pd.read_csv(src / f"{case}.csv", usecols=["x", "y"]).to_numpy(np.float64)
        sel = np.sort(rng.choice(len(a), cap, replace=False)) if len(a) > cap else np.arange(len(a))
        return a[sel], sel
    a = pd.read_csv(layout.grid_anchors(3) / f"{case}.csv", usecols=["x", "y"]).to_numpy(np.float64)
    k = min(cap, per_cell * len(a))
    side = PATCH * level_downsample(3)
    xy = a[rng.integers(0, len(a), k)] + rng.uniform(-side / 2, side / 2, (k, 2))
    return xy[np.lexsort((xy[:, 0], xy[:, 1]))], None  # raster order: neighbouring crops share image tiles


def extract_case(case: str, root: str, level: int, cap: int, per_cell: int) -> str:
    lay = Layout(root)
    out = lay.tiles(level)
    dst = out / f"{case}.npy"
    if dst.exists():
        return f"{case}: exists"
    t0 = time.time()
    xy, sel = centres(lay, case, level, cap, per_cell)
    tmp = out / f"{case}.tmp.npy"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(xy), PATCH, PATCH, 3))
    rd = TiledPyramid(lay.slide(case), max_native_level=4, tile_cache_tiles=256)
    try:
        if rd.source_level(level) != level:
            raise RuntimeError(f"{case}: level {level} is not a native level of this slide")
        for i, (cx, cy) in enumerate(xy):
            arr[i] = rd.read_patch(cx, cy, level, PATCH)
    finally:
        rd.close()
    arr.flush()
    del arr
    if sel is not None and level == 1:
        np.save(out / f"{case}_idx.npy", sel)
    os.replace(tmp, dst)
    return f"{case}: {len(xy)} tiles, {time.time() - t0:.0f}s"


def run(layout: Layout, level: int, cases, cap: int = 3000, per_cell: int = 25, workers: int = 8):
    layout.tiles(level).mkdir(parents=True, exist_ok=True)
    fn = partial(extract_case, root=str(layout.root), level=level, cap=cap, per_cell=per_cell)
    t0 = time.time()
    with Pool(workers) as pool:
        for k, msg in enumerate(pool.imap_unordered(fn, cases), 1):
            print(f"[{k}/{len(cases)}] {msg} | {(time.time() - t0) / 60:.1f} min", flush=True)
