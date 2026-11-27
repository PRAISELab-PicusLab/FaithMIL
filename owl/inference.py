"""FaithMIL for the OWL challenge: Grand Challenge "invoke" algorithm.

  init_model() -- called once at server startup (before /health returns 200): encoder, 30-member ensemble, calibration.
  run(model)   -- called for every /invoke: reads /input, writes the 4 outputs to /output.

Inputs
  /input/images/prostatectomy-wsi/<uuid>.tif            pyramidal WSI (levels x4)
  /input/images/prostatectomy-tissue-mask/<uuid>.tif    binary tissue mask, L0 == WSI L2
Outputs
  /output/overall-survival-years.json                   float, predicted median years to BCR (higher = lower risk)
  /output/prostate-cancer-risk-classification.json      "low" | "medium" | "high"
  /output/images/prostatectomy-overall-survival-attribution-heatmap/attribution.tif
                                                        uint8 0-255, pyramidal, grid and spacing of the mask L0
  /output/pathology-report.pdf                          3 pages

Pipeline: L1 tiles (0.22 mm) from the tissue mask exactly as in training -> DINO L1 features -> ensemble risk z ->
years (Weibull AFT median) and class (training tertiles) -> exact tile attributions -> heatmap and report.
Nothing is cached between invocations except the model weights.

Also usable as the black box of the challenge's faithfulness_blackbox.ipynb: get_args_parser() and run(args), with
args.wsi_dir, args.mask_dir, args.output_json (see owl/README.md).
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import tifffile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # trace_mil/ when run from the repo checkout
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_mil.anchors import S1, tree_anchors  # noqa: E402
from trace_mil.wsi import TissueMask, read_wsi_header, resolution_px_per_unit  # noqa: E402

import owl_model as om  # noqa: E402

INPUT_PATH = Path(os.environ.get("OWL_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("OWL_OUTPUT", "/output"))
MODEL_PATH = Path(os.environ.get("OWL_MODEL", "/opt/ml/model"))

WSI_SLUG = "prostatectomy-tissue-whole-slide-image"
MASK_SLUG = "prostatectomy-tissue-mask"
HEATMAP_DIR = "images/prostatectomy-overall-survival-attribution-heatmap"

_MODEL = None  # notebook mode: loaded once per process


def _log(msg):
    print(f"[trace-owl {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def init_model():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"torch {torch.__version__}, device {device}"
         + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    t0 = time.time()
    cal = om.load_json(MODEL_PATH / "calibration.json")
    import pandas as pd
    model = {"device": device, "vit": om.load_encoder(MODEL_PATH / "dino_L1.pth", device),
             "ensemble": om.Ensemble(MODEL_PATH, device), "cal": cal,
             "cohort": pd.read_csv(MODEL_PATH / "cohort_ref.csv"), "deploy": om.load_json(MODEL_PATH / "deploy.json")}
    _log(f"model loaded from {MODEL_PATH} in {time.time() - t0:.1f}s: {len(model['ensemble'].members)} members")
    return model


def run(model):
    if isinstance(model, argparse.Namespace):  # faithfulness notebook: run(args)
        return _run_notebook(model)
    interface_key = get_interface_key()
    handler = {(MASK_SLUG, WSI_SLUG): interf0_handler}[interface_key]
    return handler(model)


def interf0_handler(model):
    wsi_path = find_image(INPUT_PATH / "images/prostatectomy-wsi")
    mask_path = find_image(INPUT_PATH / "images/prostatectomy-tissue-mask")
    predict_case(model, wsi_path, mask_path, OUTPUT_PATH)
    return 0


def predict_case(model, wsi_path: Path, mask_path: Path, out_dir: Path, outputs=("years", "class", "heatmap", "report")):
    t0 = time.time()
    _log(f"WSI {wsi_path.name}, mask {mask_path.name}")
    header = read_wsi_header_safe(wsi_path)
    mask = open_tissue_mask(mask_path, header)
    mask_l0 = read_mask_l0_geometry(mask_path, header)
    _log(f"WSI L0 {header['width_l0']}x{header['height_l0']} ({header['mpp_l0']:.4f} um/px); "
         f"mask L0 {mask_l0['shape'][1]}x{mask_l0['shape'][0]}, 1 mask px = {mask_l0['downsample']:.3f} WSI px")

    parents, children = tree_anchors(mask, header["width_l0"], header["height_l0"])
    if len(children) == 0:  # very little tissue: relax the L1 tissue threshold before giving up
        parents, children = tree_anchors(mask, header["width_l0"], header["height_l0"], t2=0.0, t1=0.1)
        warnings.warn(f"no L1 tile with >= 50% tissue; {len(children)} tiles with >= 10%")
    _log(f"{len(children)} L1 tiles in {len(parents)} L2 tiles ({time.time() - t0:.0f}s)")

    cal, ens = model["cal"], model["ensemble"]
    if len(children):
        X = om.extract_features(model["vit"], wsi_path, parents, children, model["device"], log=_log)
        _log(f"features {tuple(X.shape)} ({time.time() - t0:.0f}s)")
        g, s = ens.tile_terms(X)
        z = om.ensemble_z(g, s, ens.mu, ens.sd)
        phi = om.contributions(g, s, ens.mu, ens.sd)
        rel, rule, aucs = om.relevance(g, s, ens.mu, ens.sd) if {"heatmap", "report"} & set(outputs) else (None, "", {})
    else:  # no tissue at all: the cohort median, documented in the report
        warnings.warn("no tissue tile: predicting the training-cohort median")
        z, phi, rel, rule, aucs = float(np.median(model["cohort"].z)), np.zeros(0), np.zeros(0), "none", {}
    years, risk_class = years_and_class(z, cal)
    _log(f"z {z:+.4f} -> {years:.3f} years, {risk_class}; relevance rule {rule} {aucs}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if "years" in outputs:
        write_json_file(location=out_dir / "overall-survival-years.json", content=float(years))
    if "class" in outputs:
        write_json_file(location=out_dir / "prostate-cancer-risk-classification.json", content=risk_class)
    if "heatmap" in outputs:
        # ranks: the same order as the relevance (all the faithfulness test uses), readable as an image
        heatmap = rasterize_tile_scores(children["x"].to_numpy(), children["y"].to_numpy(), om.rank01(rel), S1, mask_l0)
        write_pyramidal_tiff(out_dir / HEATMAP_DIR / "attribution.tif", heatmap, mask_l0)
        del heatmap
    if "report" in outputs:
        import owl_report
        owl_report.make_report(out_dir / "pathology-report.pdf", wsi_path, children, phi,
                               rel if len(rel) else np.zeros(0), z, years, risk_class, cal, model["cohort"], S1,
                               header["mpp_l0"], rule)
    _log(f"done in {time.time() - t0:.1f}s")
    return {"z": z, "years": years, "risk_class": risk_class, "n_tiles": len(children), "rule": rule,
            "children": children, "phi": phi, "relevance": rel, "mask_l0": mask_l0}


def years_and_class(z: float, cal: dict):
    y = cal["years"]
    years = float(np.exp(y["log_scale_intercept"] + y["log_scale_slope"] * z) * np.log(2) ** (1 / y["rho"]))
    rc = cal["risk_class"]
    cls = "low" if years >= rc["low_if_years_at_least"] else ("medium" if years >= rc["high_if_years_below"] else "high")
    return years, cls


# ---------------------------------------------------------------------------- faithfulness notebook interface

def get_args_parser():
    p = argparse.ArgumentParser("FaithMIL OWL inference (faithfulness notebook mode)")
    p.add_argument("--wsi-dir", type=Path, default=Path("/tmp/faith/slides"))
    p.add_argument("--mask-dir", type=Path, default=Path("/tmp/faith/tissue_masks"))
    p.add_argument("--output-json", type=Path, default=Path("/tmp/faith/output/overall-survival-years.json"))
    p.add_argument("--all-outputs", action="store_true", help="also write class, heatmap and report next to the json")
    return p


def _run_notebook(args):
    global _MODEL
    if _MODEL is None:
        _MODEL = init_model()
    wsi_path, mask_path = find_image(args.wsi_dir), find_image(args.mask_dir)
    outputs = ("years", "class", "heatmap", "report") if args.all_outputs else ("years",)
    res = predict_case(_MODEL, wsi_path, mask_path, args.output_json.parent, outputs)
    if args.output_json.name != "overall-survival-years.json":
        write_json_file(location=args.output_json, content=float(res["years"]))
    return 0


# ---------------------------------------------------------------------------- inputs

def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["socket"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    with open(location) as f:
        return json.loads(f.read())


def find_image(location: Path) -> Path:
    """The single image of a socket folder; its name is a UUID, the suffix is not guaranteed."""
    files = sorted(p for ext in ("*.tif", "*.tiff", "*.mha") for p in location.glob(ext))
    if not files:
        raise FileNotFoundError(f"no image in {location}: {sorted(os.listdir(location))}")
    if len(files) > 1:
        warnings.warn(f"{len(files)} images in {location}, using {files[0].name}")
    return files[0]


def read_wsi_header_safe(path: Path) -> dict:
    try:
        return read_wsi_header(path)
    except KeyError as e:  # no resolution tags: assume the OWL scanner spacing
        warnings.warn(f"WSI resolution tags missing ({e}); assuming 0.2425 um/px")
        with tifffile.TiffFile(path) as tif:
            page0 = tif.series[0].levels[0].pages[0]
            return {"height_l0": int(page0.imagelength), "width_l0": int(page0.imagewidth),
                    "n_levels_wsi": len(tif.series[0].levels), "res_l0": None, "mpp_l0": 0.2425}


def open_tissue_mask(path: Path, header: dict, level: int = 1) -> TissueMask:
    """TissueMask on mask level 1, as in training. If the tags do not match the pyramid geometry FaithMIL expects
    (the files on Grand Challenge are re-encoded by panimg), fall back to the downsample implied by the shapes."""
    if header["res_l0"] is not None:
        try:
            return TissueMask(path, level=level, wsi_res_l0=header["res_l0"])
        except (RuntimeError, KeyError) as e:
            warnings.warn(f"tissue mask tags: {e}; using the shape ratio instead")
    mask = TissueMask(path, level=level)
    mask.downsample = header["width_l0"] / mask.shape[1]
    _log(f"mask level {mask.level}: downsample from shapes {mask.downsample:.3f}")
    return mask


def read_mask_l0_geometry(path: Path, header: dict) -> dict:
    """Shape, resolution tags and WSI downsample of the mask L0: the grid of the attribution heatmap."""
    with tifffile.TiffFile(path) as tif:
        page = tif.series[0].levels[0].pages[0]
        shape = (int(page.imagelength), int(page.imagewidth))
        try:
            res = resolution_px_per_unit(page)
            unit = int(page.tags["ResolutionUnit"].value)
        except KeyError:
            res, unit = None, None
    return {"shape": shape, "res": res, "unit": unit, "downsample": header["width_l0"] / shape[1]}


# ---------------------------------------------------------------------------- outputs

def write_json_file(*, location, content):
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def rasterize_tile_scores(x, y, scores, side_l0: int, mask_l0: dict) -> np.ndarray:
    """Paint one score per square tile (centre x, y and side in WSI L0 px) on the mask L0 grid.

    Scores are min-max scaled to 1..255 over the tiles; 0 is left for no tile (background or unused tissue),
    so 255 marks the most relevant tiles. Tiles do not overlap, so the painting order does not matter."""
    H, W = mask_l0["shape"]
    ds = mask_l0["downsample"]
    out = np.zeros((H, W), np.uint8)
    if len(scores) == 0:
        return out
    s = np.asarray(scores, np.float64)
    lo, hi = s.min(), s.max()
    v = np.full(len(s), 255, np.uint8) if hi <= lo else np.rint(1 + 254 * (s - lo) / (hi - lo)).astype(np.uint8)
    half = side_l0 / 2
    x0 = np.clip(np.floor((np.asarray(x) - half) / ds).astype(int), 0, W)
    x1 = np.clip(np.ceil((np.asarray(x) + half) / ds).astype(int), 0, W)
    y0 = np.clip(np.floor((np.asarray(y) - half) / ds).astype(int), 0, H)
    y1 = np.clip(np.ceil((np.asarray(y) + half) / ds).astype(int), 0, H)
    for i in np.argsort(v, kind="stable"):  # higher values last: on shared border pixels the higher score wins
        out[y0[i]:y1[i], x0[i]:x1[i]] = v[i]
    return out


def write_pyramidal_tiff(path: Path, image: np.ndarray, mask_l0: dict, tile: int = 512):
    """Tiled, LZW, consecutive-page pyramid (2x per level) with the resolution tags of the mask L0: the same
    layout as the tissue masks the challenge provides."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(tile=(tile, tile), compression="lzw", photometric="minisblack")
    if mask_l0["res"] is not None:
        kw["resolutionunit"] = mask_l0["unit"]
    level, res = image, mask_l0["res"]
    with tifffile.TiffWriter(path, bigtiff=image.nbytes > 2**31) as tw:
        k = 0
        while True:
            extra = {"resolution": (res, res)} if res is not None else {}
            tw.write(level, subfiletype=0 if k == 0 else 1, **kw, **extra)
            if max(level.shape) <= tile:
                break
            level = level[::2, ::2]  # nearest, phase 0 (like the provided masks)
            res = res / 2 if res is not None else None
            k += 1


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))
