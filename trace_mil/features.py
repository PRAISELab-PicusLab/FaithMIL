"""Frozen features of the tiles of one view, written once per case and cached.

    features/<encoder>/<view>/<case>.npy   (n_tiles, D) float16, rows in the order of the view's anchor file
    features/<encoder>/<view>/manifest.json

Encoders
    dino         the DINO encoder of the level of the view (encoders/dino_L<k>/teacher_backbone.pth), 384-d
    prost40m     Prost40M (ViT-S/14 pre-trained on prostate slides), 384-d      (downloaded from the Hugging Face hub)
    hoptimus0    H-optimus-0 (ViT-g/14), 1536-d                                  (gated: accept the licence on the hub)
    quiltnet     QuiltNet-B-16 image tower (projected embedding), 512-d          (concepts of the report, see report.py)

Views (trace_mil.layout.FEATURE_VIEWS): children / parents are the L1 / L2 tiles of the tree; ctx2 / ctx3 are the 224-px
tiles of level 2 / 3 centred on every L1 child; grid0 / grid3 are the tiles of the level-0 / level-3 grids.
Tiles are read at the native level of the pyramid; out-of-slide pixels are white.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .dino import MEAN, STD, make_vit
from .layout import FEATURE_VIEWS, PATCH, Layout
from .wsi import TiledPyramid

HOPTIMUS_MEAN, HOPTIMUS_STD = (0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
ENCODERS = ("dino", "prost40m", "hoptimus0", "quiltnet")
CHUNK = 256  # tiles per item of the loader


def load_encoder(layout: Layout, encoder: str, level: int, device):
    """(model, mean, std, feature dim) of an encoder, ready for 224-px tiles."""
    if encoder == "dino":
        path = layout.encoder(level) / "teacher_backbone.pth"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (run scripts/03_train_dino.py --level {level})")
        model = make_vit()
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        mean, std, dim = MEAN, STD, 384
    elif encoder == "prost40m":
        import timm
        model = timm.create_model("hf-hub:waticlems/Prost40M", pretrained=True)
        mean, std, dim = MEAN, STD, 384
    elif encoder == "hoptimus0":
        import timm
        model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5,
                                  dynamic_img_size=False)
        mean, std, dim = HOPTIMUS_MEAN, HOPTIMUS_STD, 1536
    elif encoder == "quiltnet":
        import open_clip

        class Image(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m.encode_image(x)

        model = Image(open_clip.create_model_from_pretrained("hf-hub:wisdomik/QuiltNet-B-16")[0])
        mean, std, dim = CLIP_MEAN, CLIP_STD, 512
    else:
        raise ValueError(f"unknown encoder {encoder!r}; choose from {ENCODERS}")
    model = model.to(device).eval()
    return (model, torch.tensor(mean, device=device).view(1, 3, 1, 1),
            torch.tensor(std, device=device).view(1, 3, 1, 1), dim)


class TileChunks(torch.utils.data.Dataset):
    """uint8 tiles of one case in chunks; each loader worker opens its own slide reader."""

    def __init__(self, slide: str, xy: np.ndarray, level: int):
        self.slide, self.xy, self.level = slide, xy, level
        self.starts = list(range(0, len(xy), CHUNK))
        self._rd = None

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        if self._rd is None:
            self._rd = TiledPyramid(self.slide, max_native_level=4, tile_cache_tiles=512)
            if self._rd.source_level(self.level) != self.level:
                raise RuntimeError(f"{self.slide}: level {self.level} is not a native level of this slide")
        s = self.starts[i]
        part = self.xy[s:s + CHUNK]
        out = np.empty((len(part), PATCH, PATCH, 3), np.uint8)
        for j, (cx, cy) in enumerate(part):
            out[j] = self._rd.read_patch(cx, cy, self.level, PATCH)
        return torch.from_numpy(out)


@torch.no_grad()
def encode(model, u8, mean, std, device, batch: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(u8), batch):
        x = u8[i:i + batch].to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model((x - mean) / std).float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


def run(layout: Layout, encoder: str, view: str, cases, workers: int = 8, device: str | None = None):
    desc, level = FEATURE_VIEWS[view]
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = layout.features(encoder, view)
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases if not (out / f"{c}.npy").exists()]
    print(f"{encoder}/{view} ({desc}): {len(cases)} cases, {len(todo)} to do", flush=True)
    if not todo:
        return
    model, mean, std, dim = load_encoder(layout, encoder, level, dev)
    (out / "manifest.json").write_text(json.dumps(
        {"encoder": encoder, "view": view, "level": level, "feature_dim": dim, "dtype": "float16", "patch_size": PATCH,
         "anchors": str(layout.anchors_of_view(view).relative_to(layout.root))}, indent=2))
    t0 = time.time()
    for k, case in enumerate(todo, 1):
        xy = pd.read_csv(layout.anchors_of_view(view) / f"{case}.csv", usecols=["x", "y"]).to_numpy(np.float64)
        ds = TileChunks(str(layout.slide(case)), xy, level)
        kw = {"prefetch_factor": 2, "multiprocessing_context": "spawn"} if workers else {}
        dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers, **kw)
        feats = np.concatenate([encode(model, u8, mean, std, dev) for u8 in dl]) if len(xy) else np.zeros((0, dim), np.float16)
        if len(feats) != len(xy):
            raise RuntimeError(f"{case}: {len(feats)} feature rows for {len(xy)} anchors")
        tmp = out / f"{case}.tmp.npy"
        np.save(tmp, feats)
        os.replace(tmp, out / f"{case}.npy")  # written last = done marker
        print(f"[{k}/{len(todo)}] {case}: {len(feats)} tiles | {(time.time() - t0) / 60:.1f} min", flush=True)
