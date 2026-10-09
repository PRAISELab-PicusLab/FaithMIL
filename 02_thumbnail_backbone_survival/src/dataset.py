"""Dataset + preprocessing for the LEOPARD global-thumbnail baseline.

Thumbnails are already-extracted low-resolution WSI PNGs (see
thumbnails/nativeLast/*.png, one per case, at whichever pyramid level ("L4" or
"L5") was available for that slide). Images are NOT re-read from the original
multi-gigabyte WSIs anywhere in this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# Background of these pathology thumbnails is white (verified empirically on
# real files), so padding to square must use white, not black, to avoid
# introducing a high-contrast artifact border the network could latch onto.
PAD_FILL = (255, 255, 255)

def normalization_for_backbone(backbone: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Each backbone's own published normalization statistics (ImageNet for
    the torchvision/UNI/Prost40M family, Lunit's for the TCGA SSL ResNets,
    H-optimus-0's own, OpenAI CLIP's for the CLIP-family models).
    """
    from src.backbones import get_spec

    spec = get_spec(backbone)
    return spec.mean, spec.std


def input_size_for_backbone(backbone: str) -> int:
    from src.backbones import get_spec

    return get_spec(backbone).input_size


class ResizePad:
    """Resize preserving aspect ratio (longest side -> target), then pad to
    a square of (target, target) with white fill. This intentionally avoids
    cropping: these are already global, low-resolution WSI thumbnails, and
    cropping would discard tissue rather than downsample it.
    """

    def __init__(self, target_size: int):
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)

        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        left, top = pad_w // 2, pad_h // 2
        canvas = Image.new("RGB", (self.target_size, self.target_size), PAD_FILL)
        canvas.paste(img, (left, top))
        return canvas


class RandomRot90:
    """Random rotation by 0/90/180/270 degrees (lossless, no interpolation
    artifacts), appropriate for a global thumbnail with no canonical
    orientation.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        k = np.random.randint(0, 4)
        if k == 0:
            return img
        return img.transpose([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270][k - 1])


class ToNormalizedTensor:
    def __init__(self, mean: tuple, std: tuple):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        return (t - self.mean) / self.std


class Compose:
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def build_transform(image_size: int, backbone: str, train: bool) -> Compose:
    """Deterministic resize+pad+normalize for val/test; adds only flips and
    90-degree rotations for train. No aggressive color augmentation and no
    random cropping (see module docstring / task spec).
    """
    mean, std = normalization_for_backbone(backbone)
    ops: list[Callable] = [ResizePad(image_size)]
    if train:
        ops += [
            lambda img: img.transpose(Image.FLIP_LEFT_RIGHT) if np.random.rand() < 0.5 else img,
            lambda img: img.transpose(Image.FLIP_TOP_BOTTOM) if np.random.rand() < 0.5 else img,
            RandomRot90(),
        ]
    ops.append(ToNormalizedTensor(mean, std))
    return Compose(ops)


class LeopardThumbnailDataset(Dataset):
    """Wraps a subset of splits/folds.csv rows.

    Expects columns: case_id, event, follow_up_years, thumbnail_level,
    thumbnail_path (see scripts/create_folds.py).
    """

    def __init__(self, df: pd.DataFrame, image_size: int, backbone: str, train: bool):
        self.df = df.reset_index(drop=True)
        self.transform = build_transform(image_size, backbone, train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img = Image.open(row["thumbnail_path"]).convert("RGB")
        img = self.transform(img)
        return {
            "image": img,
            "case_id": row["case_id"],
            "event": torch.tensor(float(row["event"])),
            "follow_up_years": torch.tensor(float(row["follow_up_years"])),
            "fold": int(row["fold"]),
        }


def load_folds(splits_csv: Path | str) -> pd.DataFrame:
    df = pd.read_csv(splits_csv)
    required = {"case_id", "fold", "event", "follow_up_years", "thumbnail_level", "thumbnail_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{splits_csv} is missing required columns: {missing}")
    return df


def make_train_val_split(
    train_val_df: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified (by event) train/val split of the 9 non-test folds, used
    only for early stopping / model selection within an outer CV fold. Never
    touches the held-out outer test fold.
    """
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_val_df["event"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
