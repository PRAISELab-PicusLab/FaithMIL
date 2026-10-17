"""Pyramid geometry, tile-level region reading and tissue masks (tifffile only).

Facts verified on the slides of both cohorts:
  * WSI level k has exactly 4**k x the L0 pixel spacing: every level's
    XResolution tag is (10555189, 256 * 4**k) px/cm (L0 = 0.2425 um/px ...
    L5 = 248.4 um/px). Level SHAPES are floor-rounded at every step, so
    coordinates are always mapped with 4**k, never with shape ratios.
  * The scanner built the pyramid by NEAREST-neighbour decimation at phase 0:
    native L5 == L4[::4, ::4] up to JPEG noise (Pearson 0.98 on two 6-level
    slides, vs 0.73 for 4x4 box averaging and <= 0.58 for any other phase).
    So level-k pixel i IS L0 pixel 4**k * i, and the L5 missing from the 233
    five-level slides can be reproduced with the scanner's own rule.
  * All WSI levels are 512x512 JPEG (YCbCr) tiles. Decoding single tiles with
    TiffPage.decode is pixel-identical to tifffile's zarr store and ~2.5x
    faster; cold reads are GPFS-latency bound (~35 ms/tile), so parallel
    DataLoader workers matter more than decode speed.
  * Tissue-mask L0 == WSI L2 (16 L0 px per mask px, from the tags), 2x per
    mask level, stored as consecutive LZW pages.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tifffile

WSI_LEVEL_FACTOR = 4
MAX_LEVEL = 5  # coarsest level that can be read (native or synthesized)
ALL_LEVELS = tuple(range(MAX_LEVEL + 1))
BACKGROUND = 255  # slides have a white background (verified in the thumbnail work)
MASK_L0_DOWNSAMPLE = 16  # mask L0 == WSI L2; re-derived from the tags and cross-checked


def level_downsample(level: int) -> int:
    return WSI_LEVEL_FACTOR ** level


def resolution_px_per_unit(page) -> float:
    value = page.tags["XResolution"].value
    return value[0] / value[1] if isinstance(value, tuple) else float(value)


def page_mpp(page) -> float:
    unit = int(page.tags["ResolutionUnit"].value)
    microns_per_unit = {2: 25400.0, 3: 10000.0}[unit]
    return microns_per_unit / resolution_px_per_unit(page)


def level_fov_um(level: int, patch_size: int, mpp_l0: float) -> float:
    return patch_size * level_downsample(level) * mpp_l0


def patch_top_left(center_l0, level: int, patch_size: int) -> np.ndarray:
    """Top-left corner (level-k pixels) of the patch_size window centred on an
    L0 coordinate. Works elementwise on arrays."""
    center = np.asarray(center_l0, dtype=np.float64)
    return np.floor(center / level_downsample(level) - patch_size / 2 + 0.5).astype(np.int64)


def patch_box_l0(cx, cy, level: int, patch_size: int):
    """L0 box [x0, x1) x [y0, y1) covered by the level-k patch centred on (cx, cy)."""
    ds = level_downsample(level)
    tx = patch_top_left(cx, level, patch_size)
    ty = patch_top_left(cy, level, patch_size)
    return tx * ds, ty * ds, (tx + patch_size) * ds, (ty + patch_size) * ds


class TiledPyramid:
    """Random-access patch reader for one WSI.

    Levels above `max_native_level` (and levels the file does not have) are
    synthesized from the finest allowed native level by phase-0 nearest
    decimation -- the same rule the scanner used. Default 4: L5 is ALWAYS
    synthesized from L4, so the 5-level and 6-level slides get
    identically-produced L5 patches and the acquisition artifact (pyramid
    depth) cannot leak into the L5 encoder.

    Not thread/fork-safe across processes: build one instance per DataLoader
    worker.
    """

    def __init__(self, path: Path | str, max_native_level: int = 4, tile_cache_tiles: int = 256):
        self.path = Path(path)
        self._tif = tifffile.TiffFile(self.path)
        levels = self._tif.series[0].levels
        self.pages = [lvl.pages[0] for lvl in levels]
        self.n_levels = len(self.pages)
        self.shapes = [(int(p.imagelength), int(p.imagewidth)) for p in self.pages]
        self.max_native_level = max_native_level
        self.mpp_l0 = page_mpp(self.pages[0])
        self.res_l0 = resolution_px_per_unit(self.pages[0])

        for k, page in enumerate(self.pages):
            if not page.is_tiled:
                raise RuntimeError(f"{self.path.name} L{k} is not tiled")
            if page.samplesperpixel < 3 or page.planarconfig != 1:
                raise RuntimeError(f"{self.path.name} L{k}: expected contiguous RGB, got spp={page.samplesperpixel}")
            ratio = self.res_l0 / resolution_px_per_unit(page)
            if abs(ratio - level_downsample(k)) > 1e-3 * level_downsample(k):
                raise RuntimeError(
                    f"{self.path.name} L{k}: resolution-tag downsample {ratio:.4f} != 4**{k}. "
                    "The pyramid geometry assumed by TRACE does not hold."
                )

        self._cache: OrderedDict = OrderedDict()
        self._cache_size = tile_cache_tiles
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def source_level(self, level: int) -> int:
        return min(level, self.max_native_level, self.n_levels - 1)

    def _tile(self, level: int, index: int) -> np.ndarray:
        key = (level, index)
        tile = self._cache.get(key)
        if tile is not None:
            self._cache.move_to_end(key)
            return tile

        page = self.pages[level]
        th, tw = page.tilelength, page.tilewidth
        offset, count = page.dataoffsets[index], page.databytecounts[index]
        if count == 0:  # sparse tile -> background
            tile = np.full((th, tw, 3), BACKGROUND, np.uint8)
        else:
            with self._lock:
                fh = self._tif.filehandle
                fh.seek(offset)
                data = fh.read(count)
            decoded, _, _ = page.decode(data, index, jpegtables=page.jpegtables)
            tile = np.ascontiguousarray(np.asarray(decoded).reshape(th, tw, -1)[..., :3])

        self._cache[key] = tile
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return tile

    def read_region(self, level: int, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        """RGB uint8 (h, w, 3) of a NATIVE level; out-of-image pixels are white."""
        out = np.full((h, w, 3), BACKGROUND, np.uint8)
        H, W = self.shapes[level]
        xs, xe = max(x0, 0), min(x0 + w, W)
        ys, ye = max(y0, 0), min(y0 + h, H)
        if xs >= xe or ys >= ye:
            return out

        page = self.pages[level]
        th, tw = page.tilelength, page.tilewidth
        n_tiles_x = math.ceil(W / tw)
        for ty in range(ys // th, (ye - 1) // th + 1):
            for tx in range(xs // tw, (xe - 1) // tw + 1):
                tile = self._tile(level, ty * n_tiles_x + tx)
                iy0, iy1 = max(ys, ty * th), min(ye, ty * th + th)
                ix0, ix1 = max(xs, tx * tw), min(xe, tx * tw + tw)
                out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = tile[iy0 - ty * th:iy1 - ty * th, ix0 - tx * tw:ix1 - tx * tw]
        return out

    def read_patch(self, cx: float, cy: float, level: int, patch_size: int) -> np.ndarray:
        """(patch_size, patch_size, 3) patch at `level` centred on L0 point (cx, cy)."""
        src = self.source_level(level)
        f = WSI_LEVEL_FACTOR ** (level - src)
        tx = int(patch_top_left(cx, level, patch_size))
        ty = int(patch_top_left(cy, level, patch_size))
        region = self.read_region(src, tx * f, ty * f, patch_size * f, patch_size * f)
        return region[::f, ::f] if f > 1 else region

    def close(self) -> None:
        self._tif.close()
        self._cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TissueMask:
    """Binary tissue mask at one mask pyramid level, with a summed-area table
    for O(1) tissue fractions of arbitrary L0 boxes.

    mask_level=1 (7.8 um/px) resolves even the L0 patch (54 um = 7 px) and
    keeps the int32 integral image small (~135 MB for the largest slide).
    """

    def __init__(self, path: Path | str, level: int = 1, wsi_res_l0: float | None = None):
        with tifffile.TiffFile(path) as tif:
            levels = tif.series[0].levels
            self.level = min(level, len(levels) - 1)
            page = levels[self.level].pages[0]
            # maxworkers=1: tifffile's default thread pool thrashes when several processes decode at once.
            arr = levels[self.level].asarray(maxworkers=1)
            res = resolution_px_per_unit(page)
            res_mask0 = resolution_px_per_unit(levels[0].pages[0])
        if arr.ndim == 3:
            arr = arr[..., 0]

        expected = MASK_L0_DOWNSAMPLE * 2 ** self.level
        if wsi_res_l0 is not None:
            ds = wsi_res_l0 / res
            if abs(ds - expected) > 1e-3 * expected:
                raise RuntimeError(
                    f"{Path(path).name}: mask level {self.level} is {ds:.3f} L0 px/px, expected {expected}."
                )
            ds0 = wsi_res_l0 / res_mask0
            if abs(ds0 - MASK_L0_DOWNSAMPLE) > 1e-3 * MASK_L0_DOWNSAMPLE:
                raise RuntimeError(f"{Path(path).name}: mask L0 is {ds0:.3f} L0 px/px, expected 16 (== WSI L2).")
        self.downsample = expected
        self.binary = arr > 0
        self.shape = self.binary.shape
        H, W = self.shape
        sat = np.zeros((H + 1, W + 1), dtype=np.int32)
        np.cumsum(self.binary, axis=0, dtype=np.int32, out=sat[1:, 1:])
        np.cumsum(sat[1:, 1:], axis=1, out=sat[1:, 1:])
        self._sat = sat

    def fraction(self, x0, y0, x1, y1) -> np.ndarray:
        """Tissue fraction of L0 boxes [x0, x1) x [y0, y1) (arrays OK).
        Area outside the slide counts as background."""
        ds = self.downsample
        mx0 = np.floor(np.asarray(x0) / ds).astype(np.int64)
        my0 = np.floor(np.asarray(y0) / ds).astype(np.int64)
        mx1 = np.ceil(np.asarray(x1) / ds).astype(np.int64)
        my1 = np.ceil(np.asarray(y1) / ds).astype(np.int64)
        area = np.maximum((mx1 - mx0) * (my1 - my0), 1)
        H, W = self.shape
        cx0, cx1 = np.clip(mx0, 0, W), np.clip(mx1, 0, W)
        cy0, cy1 = np.clip(my0, 0, H), np.clip(my1, 0, H)
        s = self._sat
        tissue = s[cy1, cx1] - s[cy0, cx1] - s[cy1, cx0] + s[cy0, cx0]
        return tissue / area


def read_wsi_header(path: Path | str) -> dict:
    """Level-0 shape, level count and spacing without decoding any pixels."""
    with tifffile.TiffFile(path) as tif:
        levels = tif.series[0].levels
        page0 = levels[0].pages[0]
        return {
            "height_l0": int(page0.imagelength),
            "width_l0": int(page0.imagewidth),
            "n_levels_wsi": len(levels),
            "res_l0": resolution_px_per_unit(page0),
            "mpp_l0": page_mpp(page0),
        }
