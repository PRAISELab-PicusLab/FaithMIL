"""Anchor points for trajectories: a tissue-filtered regular grid.

Anchors are centres of a regular grid in L0 pixels. The default stride equals
one patch at `grid_level=2` (224 px * 4**2 = 3584 L0 px = 0.87 mm), i.e. the
L2 patches tile the tissue without overlap and gaps; L1/L0 patches are a zoomed
sample at the centre of each L2 tile, L3-L5 add surrounding context.

A trajectory is kept only if the patch at EVERY level has tissue, with a
per-level minimum tissue fraction (`min_tissue`). The finest patch is the
binding constraint (it is contained in all coarser ones), so coarse levels
default to "> 0": thresholding them would drop the tissue border, where
surgical margins -- a known BCR factor -- live.

The anchor set is defined ONCE for all six levels, independently of which
levels a model later uses, so "L5-L4" vs "L5-L4-L3" vs ... experiments differ
only in the levels, never in the sampled tissue.
"""
from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.trajectory.wsi import ALL_LEVELS, TissueMask, level_downsample, patch_box_l0, read_wsi_header


@dataclass
class AnchorConfig:
    patch_size: int = 224
    grid_level: int = 2
    stride_l0: int | None = None  # overrides grid_level when set
    min_tissue: dict = field(default_factory=lambda: {0: 0.5, 1: 0.25, 2: 0.1})
    max_anchors: int | None = None  # random spatially-uniform cap per slide (None = keep all)
    mask_level: int = 1
    seed: int = 0

    @property
    def stride(self) -> int:
        return int(self.stride_l0) if self.stride_l0 else self.patch_size * level_downsample(self.grid_level)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["min_tissue"] = {int(k): float(v) for k, v in self.min_tissue.items()}
        d["stride_effective_l0"] = self.stride
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AnchorConfig":
        d = dict(d)
        d.pop("stride_effective_l0", None)
        d["min_tissue"] = {int(k): float(v) for k, v in d["min_tissue"].items()}
        return cls(**d)


def case_seed(case_id: str, seed: int) -> int:
    # crc32, not hash(): stable across processes regardless of PYTHONHASHSEED
    return (zlib.crc32(case_id.encode()) + 1_000_003 * seed) % (2**32)


def generate_anchors(case_id: str, wsi_path, mask_path, cfg: AnchorConfig) -> tuple[pd.DataFrame, dict]:
    header = read_wsi_header(wsi_path)
    H0, W0 = header["height_l0"], header["width_l0"]
    mask = TissueMask(mask_path, level=cfg.mask_level, wsi_res_l0=header["res_l0"])

    S = cfg.stride
    n_cols, n_rows = int(np.ceil(W0 / S)), int(np.ceil(H0 / S))
    gy, gx = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()  # raster order (row-major) -> tile-cache locality
    cx = S // 2 + gx * S
    cy = S // 2 + gy * S

    tissue = {}
    for lvl in ALL_LEVELS:
        x0, y0, x1, y1 = patch_box_l0(cx, cy, lvl, cfg.patch_size)
        tissue[lvl] = mask.fraction(x0, y0, x1, y1)

    keep = np.ones(len(cx), dtype=bool)
    for lvl in ALL_LEVELS:
        keep &= tissue[lvl] > 0.0  # no patch without tissue, at any level
        keep &= tissue[lvl] >= cfg.min_tissue.get(lvl, 0.0)
    n_candidates = int(keep.sum())

    idx = np.flatnonzero(keep)
    capped = False
    if cfg.max_anchors is not None and len(idx) > cfg.max_anchors:
        rng = np.random.default_rng(case_seed(case_id, cfg.seed))
        idx = np.sort(rng.choice(idx, size=cfg.max_anchors, replace=False))
        capped = True

    df = pd.DataFrame({
        "anchor_id": np.arange(len(idx)),
        "x": cx[idx],
        "y": cy[idx],
        "gx": gx[idx],
        "gy": gy[idx],
        **{f"tissue_L{lvl}": np.round(tissue[lvl][idx], 4) for lvl in ALL_LEVELS},
    })
    info = {
        "case_id": case_id,
        "n_levels_wsi": header["n_levels_wsi"],
        "width_l0": W0,
        "height_l0": H0,
        "mpp_l0": header["mpp_l0"],
        "stride_l0": S,
        "grid_cols": n_cols,
        "grid_rows": n_rows,
        "n_grid": int(len(cx)),
        "n_candidates": n_candidates,
        "n_anchors": int(len(df)),
        "capped": capped,
    }
    return df, info
