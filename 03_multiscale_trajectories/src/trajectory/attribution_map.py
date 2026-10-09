"""Spatial attribution map in the challenge format.

"8-bit unsigned integer multiresolution TIFF, with the spacing of the provided
tissue masks": the map has exactly the shape, pyramid layout (consecutive
pages, reduced levels with subfiletype=1), 512 tiles, LZW compression and
XResolution/YResolution/ResolutionUnit tags of the case's own tissue mask.

Each trajectory owns its grid cell (stride x stride L0 px, e.g. 0.87 mm) and
paints it with its attribution; pixels outside the tissue mask are 0.
Encodings (`mode`):
  abs      1 + 254 * |phi| / scale   (default: relevance magnitude, what a
                                      removal-based faithfulness test probes)
  positive 1 + 254 * max(phi, 0) / scale   (risk-increasing evidence only)
  signed   128 + 127 * phi / scale  (128 = neutral, >128 risk-increasing)
with scale = 99th percentile of |phi| on the slide (robust to one outlier).
0 is reserved for background / tissue that has no analysed trajectory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from src.trajectory.wsi import MASK_L0_DOWNSAMPLE


def values_to_grid(gx, gy, values, n_rows: int, n_cols: int) -> np.ndarray:
    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    grid[np.asarray(gy), np.asarray(gx)] = values
    return grid


def encode_uint8(grid: np.ndarray, mode: str = "abs") -> np.ndarray:
    has = np.isfinite(grid)
    v = np.where(has, grid, 0.0)
    scale = np.percentile(np.abs(v[has]), 99) if has.any() else 1.0
    scale = scale if scale > 0 else 1.0
    if mode == "abs":
        u = 1 + 254 * np.clip(np.abs(v) / scale, 0, 1)
    elif mode == "positive":
        u = 1 + 254 * np.clip(v / scale, 0, 1)
    elif mode == "signed":
        u = 128 + 127 * np.clip(v / scale, -1, 1)
    else:
        raise ValueError(f"unknown map mode '{mode}'")
    return np.where(has, np.rint(u), 0).astype(np.uint8)


def render_map(grid_u8: np.ndarray, stride_l0: int, mask_path: Path | str) -> tuple[np.ndarray, list]:
    """Paint the per-cell uint8 grid at mask-L0 resolution and zero the
    background. Returns (image, mask pages) -- the pages carry the layout to copy."""
    cell = stride_l0 / MASK_L0_DOWNSAMPLE
    if abs(cell - round(cell)) > 1e-9:
        raise ValueError(f"stride {stride_l0} is not a multiple of {MASK_L0_DOWNSAMPLE} L0 px")
    cell = int(round(cell))
    with tifffile.TiffFile(mask_path) as tif:
        levels = tif.series[0].levels
        out = levels[0].asarray(maxworkers=1)  # reused in place as the output buffer
        pages = [{
            "shape": lvl.shape[:2],
            "xres": lvl.pages[0].tags["XResolution"].value,
            "yres": lvl.pages[0].tags["YResolution"].value,
            "unit": int(lvl.pages[0].tags["ResolutionUnit"].value),
        } for lvl in levels]
    if out.ndim == 3:
        out = np.ascontiguousarray(out[..., 0])
    H, W = out.shape
    # one grid row (= one band of `cell` mask rows) at a time: peak memory is
    # ~1x the mask, which matters for the largest slides (135 MP at mask L0)
    for r in range(int(np.ceil(H / cell))):
        band = out[r * cell:(r + 1) * cell]
        if r < grid_u8.shape[0]:
            row = np.zeros(W, np.uint8)
            vals = np.repeat(grid_u8[r], cell)[:W]
            row[:len(vals)] = vals
        else:
            row = np.zeros(W, np.uint8)
        np.copyto(band, np.where(band > 0, row[None, :], 0).astype(np.uint8))
    return out, pages


def write_pyramid_tiff(path: Path | str, image: np.ndarray, pages: list) -> None:
    """Write `image` (== mask L0 shape) + reduced levels with the exact shapes and
    resolution tags of the reference mask pages."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.shape != tuple(pages[0]["shape"]):
        raise ValueError(f"map shape {image.shape} != mask L0 shape {pages[0]['shape']}")
    unit = {2: "INCH", 3: "CENTIMETER"}.get(pages[0]["unit"], "NONE")
    level_img = image
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        for j, pg in enumerate(pages):
            if j > 0:
                h, w = pg["shape"]
                level_img = np.asarray(Image.fromarray(level_img).resize((w, h), Image.BOX))
            tif.write(level_img, tile=(512, 512), compression="lzw", photometric="minisblack",
                      resolution=(pg["xres"], pg["yres"]), resolutionunit=unit,
                      subfiletype=0 if j == 0 else 1, metadata=None)
