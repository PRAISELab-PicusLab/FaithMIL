"""On-disk layout of the trajectory pipeline (one place, so scripts agree).

All paths are relative to the data root (see ROOT below), never to the code.

    <traj_root>/anchors/<anchor_set>/config.json         AnchorConfig
    <traj_root>/anchors/<anchor_set>/<case_id>.csv       one row per trajectory
    <traj_root>/anchors/<anchor_set>/summary.csv         per-case counts
    <traj_root>/features/<anchor_set>/<backbone>/L<k>/manifest.json
    <traj_root>/features/<anchor_set>/<backbone>/L<k>/<case_id>.npy   (N, D) float16,
                                                     row i <-> anchor_id i
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Every default path is relative to the data root: $OWL_DATA_ROOT when set,
# otherwise the directory commands are launched from (see the README).
ROOT = Path(os.environ.get("OWL_DATA_ROOT", os.getcwd())).resolve()
DEFAULT_IMAGES_DIR = ROOT / "images"
DEFAULT_MASKS_DIR = ROOT / "tissue-masks"
DEFAULT_TRAJ_ROOT = ROOT / "trajectories"


def wsi_path(case_id: str, images_dir: Path | str = DEFAULT_IMAGES_DIR) -> Path:
    return Path(images_dir) / f"{case_id}.tif"


def mask_path(case_id: str, masks_dir: Path | str = DEFAULT_MASKS_DIR) -> Path:
    return Path(masks_dir) / f"{case_id}_tissue.tif"


def anchors_dir(traj_root: Path | str, anchor_set: str) -> Path:
    return Path(traj_root) / "anchors" / anchor_set


def features_dir(traj_root: Path | str, anchor_set: str, backbone: str, level: int) -> Path:
    return Path(traj_root) / "features" / anchor_set / backbone / f"L{level}"


def load_anchor_config(traj_root: Path | str, anchor_set: str) -> dict:
    with open(anchors_dir(traj_root, anchor_set) / "config.json") as f:
        return json.load(f)


def load_anchors(traj_root: Path | str, anchor_set: str, case_id: str) -> pd.DataFrame:
    return pd.read_csv(anchors_dir(traj_root, anchor_set) / f"{case_id}.csv")


def load_features(traj_root, anchor_set: str, backbone: str, level: int, case_id: str, mmap: bool = False) -> np.ndarray:
    path = features_dir(traj_root, anchor_set, backbone, level) / f"{case_id}.npy"
    return np.load(path, mmap_mode="r" if mmap else None)


def load_manifest(traj_root, anchor_set: str, backbone: str, level: int) -> dict:
    with open(features_dir(traj_root, anchor_set, backbone, level) / "manifest.json") as f:
        return json.load(f)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Write-then-rename, so an interrupted job never leaves a truncated
    file that a resumed job would mistake for a finished case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
