"""Local re-implementation of the challenge's faithfulness_blackbox.ipynb (tifffile instead of ASAP), with the
container's predict_case as the black box. Every perturbed mask is a full, independent run of the model.

Method (as the notebook): 256 px patches at 0.5 um/px on the mask grid (32 mask px), kept if >= 10% tissue; ranked by
the median of the map over their tissue pixels (ties: one seeded shuffle, TIE_SEED 0); the top x% (or a random x%,
seed 33, cumulative) set to 0 in the mask; y = |z_perturbed - z_unperturbed| on the predicted years, z with the mean and
SD of the unperturbed predictions; AUC = trapezoid over the % removed from (0, 0), divided by the x range;
gap = AUC_attribution - AUC_random.

    python FaithMIL/owl/faithfulness_local.py --cases case_radboud_0012 ... --work owl_tests/faith
Maps compared: "relevance" (the submitted heatmap) and, for reference, "abs_contribution" (|phi| of each tile).
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

OWL = Path(__file__).resolve().parent
PERCENTAGES = [1, 2, 4, 8, 16, 32, 64]
SEED, TIE_SEED, TILE_MIN = 33, 0, 0.10


def tissue_patches(tissue, tile):
    h, w = tissue.shape
    counts = np.add.reduceat(np.add.reduceat(tissue.astype(np.int64), np.arange(0, h, tile), axis=0),
                             np.arange(0, w, tile), axis=1)
    rows, cols = np.nonzero((counts > 0) & (counts >= TILE_MIN * tile ** 2))
    return rows, cols


def medians(values, tissue, rows, cols, tile):
    out = np.empty(len(rows))
    for i, (r, c) in enumerate(zip(rows, cols)):
        win = (slice(r * tile, (r + 1) * tile), slice(c * tile, (c + 1) * tile))
        out[i] = np.median(values[win][tissue[win]])
    return out


def n_remove(pct, n):
    return math.ceil(round(pct * n / 100, 6))


def curve_auc(values):
    x = np.r_[0.0, np.asarray(PERCENTAGES, float) / 100]
    y = np.r_[0.0, values]
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2) / (x[-1] - x[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("."))
    a = ap.parse_args()
    os.environ.setdefault("OWL_MODEL", str(OWL / "model"))
    sys.path.insert(0, str(OWL))
    import inference

    model = inference.init_model()
    a.work.mkdir(parents=True, exist_ok=True)
    gaps_fp = a.work / "slide_gaps.csv"
    rows_out = pd.read_csv(gaps_fp).to_dict("records") if gaps_fp.exists() else []
    done = {r["case_id"] for r in rows_out}  # resume: finished slides are kept
    a.cases = [c for c in a.cases if c not in done]
    print(f"{len(done)} slides already done, {len(a.cases)} to do")
    base = {}
    for case in a.cases:  # unperturbed predictions first: their mean / SD define z
        wsi, mask = a.data_root / "images" / f"{case}.tif", a.data_root / "tissue-masks" / f"{case}_tissue.tif"
        base[case] = inference.predict_case(model, wsi, mask, a.work / case / "unperturbed", ("years", "heatmap"))
    yrs = np.array([b["years"] for b in base.values()])
    cohort_years = model["cohort"].years.to_numpy()
    # few slides: their SD is noisy, so z uses the OOF predictions of the training cohort (same population)
    z_mean, z_sd = (yrs.mean(), yrs.std(ddof=1)) if len(yrs) >= 20 else (cohort_years.mean(), cohort_years.std(ddof=1))
    print(f"z statistics: mean {z_mean:.3f}, SD {z_sd:.3f} years")

    for case in a.cases:
        b = base[case]
        wsi, mask_fp = a.data_root / "images" / f"{case}.tif", a.data_root / "tissue-masks" / f"{case}_tissue.tif"
        with tifffile.TiffFile(mask_fp) as t:
            page = t.series[0].levels[0].pages[0]
            mask = t.series[0].levels[0].asarray()
            geo = {"shape": mask.shape, "res": inference.resolution_px_per_unit(page),
                   "unit": int(page.tags["ResolutionUnit"].value), "downsample": b["mask_l0"]["downsample"]}
        tile = int(round(256 * 2 * 1 / b["mask_l0"]["downsample"]))  # 0.5 um at WSI L0 (0.2425 um) -> factor 2
        tissue = mask == 1
        rows, cols = tissue_patches(tissue, tile)
        ch = b["children"]
        maps = {"relevance": inference.rasterize_tile_scores(ch.x.values, ch.y.values, b["relevance"], inference.S1, geo),
                "abs_contribution": inference.rasterize_tile_scores(ch.x.values, ch.y.values, np.abs(b["phi"]),
                                                                    inference.S1, geo)}
        orders = {}
        for name, m in maps.items():
            med = medians(m, tissue, rows, cols, tile)
            sh = np.random.default_rng(TIE_SEED).permutation(len(med))
            orders[name] = sh[np.argsort(-med[sh], kind="stable")]
        orders["random"] = np.random.default_rng(SEED).permutation(len(rows))
        z0 = (b["years"] - z_mean) / z_sd
        curves = {}
        for name, order in orders.items():
            ys = []
            for pct in PERCENTAGES:
                sel = order[:n_remove(pct, len(rows))]
                pm = mask.copy()
                for r, c in zip(rows[sel], cols[sel]):
                    pm[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = 0
                fp = a.work / case / "masks" / f"{name}_pct{pct:02d}.tif"
                inference.write_pyramidal_tiff(fp, pm, geo)
                t0 = time.time()
                r_ = inference.predict_case(model, wsi, fp, a.work / case / f"{name}_{pct}", ("years",))
                ys.append(abs((r_["years"] - z_mean) / z_sd - z0))
                print(f"  {case} {name:16s} {pct:2d}%: {r_['n_tiles']:6d} tiles, |dz| {ys[-1]:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                fp.unlink()
            curves[name] = np.array(ys)
        row = {"case_id": case, "years": b["years"], "n_patches": len(rows), "rule": b["rule"],
               "auc_random": curve_auc(curves["random"])}
        for name in maps:
            row[f"auc_{name}"] = curve_auc(curves[name])
            row[f"gap_{name}"] = row[f"auc_{name}"] - row["auc_random"]
        rows_out.append(row)
        pd.DataFrame(rows_out).to_csv(gaps_fp, index=False)
        print(f"{case}: gap relevance {row['gap_relevance']:+.4f}, gap |phi| {row['gap_abs_contribution']:+.4f} "
              f"(AUC random {row['auc_random']:.4f})", flush=True)
    df = pd.DataFrame(rows_out)
    print("MEAN over slides:", df[[c for c in df.columns if c.startswith(("gap_", "auc_"))]].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
