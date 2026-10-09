#!/usr/bin/env python3
"""Extract aligned WSI/tissue-mask thumbnails at pyramid levels L4, L5 and nativeLast.

For every case this reads the WSI and its tissue mask with tifffile (no OpenSlide),
picks the requested pyramid level(s) of the WSI, finds the mask pyramid level that
should be pixel-aligned with it, and writes an RGB PNG (WSI) + single-channel 0/255
PNG (mask) pair per output folder, plus one metadata CSV per folder.

Level correspondence
---------------------
The WSI pyramid downsamples by 4x per level; the tissue-mask pyramid downsamples by
2x per level and its level 0 already corresponds to WSI level 2. So WSI level k
should correspond to mask level j = 2*k - 4. This holds exactly for k=4 (j=4) on
every case. For k=5, j=6 does not exist (masks only go up to level 5), so the
nearest available mask level is used and rescaled to the WSI level's shape with
nearest-neighbor interpolation (never bilinear/Lanczos on a binary mask).

The predicted level is never trusted blindly: the actual decoded shapes are always
compared (within a 1-pixel tolerance) and any mismatch is logged rather than
silently patched over.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from PIL import Image

CASE_RE = re.compile(r"^(case_radboud_\d+)\.tif$")
SHAPE_TOLERANCE_PX = 1
METADATA_FIELDS = [
    "case_id",
    "level_used",
    "n_levels_wsi",
    "n_levels_mask",
    "width",
    "height",
    "mpp_effective",
    "mask_level_used",
    "mask_resized",
    "shapes_match",
    "tissue_fraction",
]

DEFAULT_IMAGES_DIR = "/leonardo_scratch/large/userexternal/vmoscato/wacv_challenge/images"
DEFAULT_MASKS_DIR = "/leonardo_scratch/large/userexternal/vmoscato/wacv_challenge/tissue-masks"
DEFAULT_OUTPUT_DIR = "/leonardo_scratch/large/userexternal/vmoscato/wacv_challenge/thumbnails"


@dataclass
class CaseOutcome:
    case_id: str
    folder_status: dict[str, str]  # folder -> "ok" | "skip" | "error"
    records: dict[str, dict]  # folder -> metadata row (only for ok/skip)
    warnings: list[str]
    error: Optional[str]


def rational_float(value) -> float:
    if isinstance(value, tuple):
        return float(value[0]) / float(value[1])
    return float(value)


def level_mpp(level) -> float:
    """Average X/Y microns-per-pixel read from this pyramid level's own resolution tags."""
    page = level.pages[0]
    xres = rational_float(page.tags["XResolution"].value)
    yres = rational_float(page.tags["YResolution"].value)
    unit = int(page.tags["ResolutionUnit"].value)
    microns_per_unit = {2: 25400.0, 3: 10000.0}.get(unit)
    if microns_per_unit is None:
        return float("nan")
    return (microns_per_unit / xres + microns_per_unit / yres) / 2.0


def predicted_mask_level(k: int) -> int:
    """WSI level k <-> mask level j = 2k - 4 (mask x2/level, mask L0 == WSI L2)."""
    return 2 * k - 4


def shapes_close(a_hw: tuple[int, int], b_hw: tuple[int, int], tol: int = SHAPE_TOLERANCE_PX) -> bool:
    return abs(a_hw[0] - b_hw[0]) <= tol and abs(a_hw[1] - b_hw[1]) <= tol


def align_mask(mask_arr: np.ndarray, target_hw: tuple[int, int]) -> tuple[np.ndarray, bool]:
    """Return a binary (0/1) mask resized to target_hw with nearest-neighbor if needed."""
    binary = (mask_arr > 0).astype(np.uint8)
    if binary.shape[:2] == target_hw:
        return binary, False
    img = Image.fromarray(binary * 255)
    resized = img.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8), True


def decode_level(
    case_id: str,
    k: int,
    wsi_levels,
    mask_levels,
    warnings: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Decode WSI level k as RGB uint8 and its aligned tissue mask (0/255 uint8)."""
    wsi_level = wsi_levels[k]
    rgb = wsi_level.asarray()
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    rgb = rgb[..., :3].astype(np.uint8)
    target_hw = rgb.shape[:2]

    n_mask_levels = len(mask_levels)
    j_pred = predicted_mask_level(k)
    j_used = min(max(j_pred, 0), n_mask_levels - 1)
    raw_mask = mask_levels[j_used].asarray()
    if raw_mask.ndim == 3:
        raw_mask = raw_mask[..., 0]

    matched = shapes_close(raw_mask.shape[:2], target_hw)
    mask01, resized = align_mask(raw_mask, target_hw)

    if not matched:
        msg = (
            f"{case_id} L{k}: mask shape mismatch beyond {SHAPE_TOLERANCE_PX}px "
            f"(predicted mask level={j_pred}, used level={j_used}/{n_mask_levels - 1}, "
            f"wsi_shape={target_hw}, mask_shape={raw_mask.shape[:2]}) -> nearest-neighbor resized"
        )
        warnings.append(msg)

    meta = {
        "width": target_hw[1],
        "height": target_hw[0],
        "mpp_effective": round(level_mpp(wsi_level), 4),
        "mask_level_used": j_used,
        "mask_resized": resized,
        "shapes_match": matched,
        "tissue_fraction": round(float(mask01.mean()), 6),
    }
    return rgb, mask01 * 255, meta


def output_paths(output_root: Path, folder: str, case_id: str) -> tuple[Path, Path]:
    img_out = output_root / folder / f"{case_id}.png"
    mask_out = output_root / folder / "masks" / f"{case_id}.png"
    return img_out, mask_out


def process_case(case_id: str, images_dir: Path, masks_dir: Path, output_root: Path, force: bool) -> CaseOutcome:
    warnings: list[str] = []
    wsi_path = images_dir / f"{case_id}.tif"
    mask_path = masks_dir / f"{case_id}_tissue.tif"

    if not wsi_path.exists():
        return CaseOutcome(case_id, {}, {}, warnings, f"missing WSI file: {wsi_path}")
    if not mask_path.exists():
        return CaseOutcome(case_id, {}, {}, warnings, f"missing mask file: {mask_path}")

    try:
        with tifffile.TiffFile(wsi_path) as wt, tifffile.TiffFile(mask_path) as mt:
            wsi_levels = wt.series[0].levels
            mask_levels = mt.series[0].levels
            n_wsi = len(wsi_levels)
            n_mask = len(mask_levels)

            if n_wsi < 5:
                return CaseOutcome(
                    case_id, {}, {}, warnings, f"unexpected WSI level count {n_wsi} (<5), skipping case"
                )

            native_k = n_wsi - 1
            folder_targets: list[tuple[str, int]] = [("L4", 4)]
            if n_wsi >= 6:
                folder_targets.append(("L5", 5))
            folder_targets.append(("nativeLast", native_k))

            decoded_cache: dict[int, tuple[np.ndarray, np.ndarray, dict]] = {}
            folder_status: dict[str, str] = {}
            records: dict[str, dict] = {}

            for folder, k in folder_targets:
                img_out, mask_out = output_paths(output_root, folder, case_id)
                if k not in decoded_cache:
                    decoded_cache[k] = decode_level(case_id, k, wsi_levels, mask_levels, warnings)
                rgb, mask255, meta = decoded_cache[k]

                wrote = force or not (img_out.exists() and mask_out.exists())
                if wrote:
                    img_out.parent.mkdir(parents=True, exist_ok=True)
                    mask_out.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb).save(img_out, format="PNG")
                    Image.fromarray(mask255).save(mask_out, format="PNG")

                folder_status[folder] = "ok" if wrote else "skip"
                records[folder] = {
                    "case_id": case_id,
                    "level_used": k,
                    "n_levels_wsi": n_wsi,
                    "n_levels_mask": n_mask,
                    **meta,
                }

            return CaseOutcome(case_id, folder_status, records, warnings, None)
    except Exception as exc:  # noqa: BLE001 - one bad case must not stop the batch
        tb = traceback.format_exc()
        return CaseOutcome(case_id, {}, {}, warnings, f"{type(exc).__name__}: {exc}\n{tb}")


def discover_cases(images_dir: Path, masks_dir: Path) -> list[str]:
    cases = []
    for f in sorted(images_dir.glob("case_radboud_*.tif")):
        m = CASE_RE.match(f.name)
        if not m:
            continue
        case_id = m.group(1)
        if (masks_dir / f"{case_id}_tissue.tif").exists():
            cases.append(case_id)
    return cases


def write_metadata_csv(output_root: Path, folder: str, rows: list[dict]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["case_id"])
    csv_path = output_root / folder / f"{folder}_metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, type=Path)
    parser.add_argument("--masks-dir", default=DEFAULT_MASKS_DIR, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes (I/O bound on GPFS; default 8, higher values can degrade throughput).")
    parser.add_argument("--force", action="store_true", help="Re-extract even if output PNGs already exist.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N discovered cases (for dry runs).")
    parser.add_argument("--cases", type=str, default=None, help="Comma-separated list of case_ids to process instead of auto-discovery.")
    parser.add_argument("--log-file", type=Path, default=None, help="Path to log file (default: <output-dir>/logs/extract_<timestamp>.log).")
    args = parser.parse_args()

    output_root: Path = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    log_file = args.log_file or (output_root / "logs" / f"extract_{time.strftime('%Y%m%d_%H%M%S')}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("extract_thumbnails")

    if args.cases:
        case_list = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        case_list = discover_cases(args.images_dir, args.masks_dir)
        if args.limit:
            case_list = case_list[: args.limit]

    logger.info("Processing %d cases with %d workers (force=%s)", len(case_list), args.workers, args.force)
    logger.info("images_dir=%s masks_dir=%s output_dir=%s", args.images_dir, args.masks_dir, output_root)

    folder_status_counts: dict[str, dict[str, int]] = {
        "L4": {"ok": 0, "skip": 0, "error": 0},
        "L5": {"ok": 0, "skip": 0, "error": 0},
        "nativeLast": {"ok": 0, "skip": 0, "error": 0},
    }
    folder_rows: dict[str, list[dict]] = {"L4": [], "L5": [], "nativeLast": []}
    failed_cases: list[tuple[str, str]] = []

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_case, case_id, args.images_dir, args.masks_dir, output_root, args.force): case_id
            for case_id in case_list
        }
        for future in as_completed(futures):
            case_id = futures[future]
            outcome: CaseOutcome = future.result()
            done += 1

            for w in outcome.warnings:
                logger.warning(w)

            if outcome.error:
                logger.error("%s: FAILED - %s", case_id, outcome.error)
                failed_cases.append((case_id, outcome.error.splitlines()[0]))
                for folder in folder_status_counts:
                    folder_status_counts[folder]["error"] += 1
            else:
                for folder, status in outcome.folder_status.items():
                    folder_status_counts[folder][status] += 1
                    folder_rows[folder].append(outcome.records[folder])

            if done % 25 == 0 or done == len(case_list):
                elapsed = time.time() - t0
                logger.info("Progress: %d/%d cases (%.1fs elapsed)", done, len(case_list), elapsed)

    for folder, rows in folder_rows.items():
        write_metadata_csv(output_root, folder, rows)

    logger.info("=" * 60)
    logger.info("FINAL REPORT")
    for folder, counts in folder_status_counts.items():
        logger.info("  %-12s ok=%d skip=%d error=%d", folder, counts["ok"], counts["skip"], counts["error"])
    logger.info("  Failed cases: %d", len(failed_cases))
    for case_id, err in failed_cases:
        logger.info("    %s: %s", case_id, err)
    logger.info("Log file: %s", log_file)

    return 1 if failed_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
