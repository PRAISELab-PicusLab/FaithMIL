"""Behavioural checks of the submission checklist on one training case (GPU node, from the data root):

  determinism   two runs on the same input give |dz| <= 0.01 (z of the years, cohort SD as in the faithfulness test)
  mask respect  removing half of the tissue mask changes the prediction; an empty mask does not crash

    python FaithMIL/owl/selftest.py --case case_radboud_0466 --work owl_tests/selftest
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tifffile

OWL = Path(__file__).resolve().parent


def write_mask_like(src: Path, dst: Path, arr: np.ndarray, inference):
    with tifffile.TiffFile(src) as t:
        page = t.series[0].levels[0].pages[0]
        geo = {"shape": arr.shape, "res": inference.resolution_px_per_unit(page),
               "unit": int(page.tags["ResolutionUnit"].value), "downsample": 16.0}
    inference.write_pyramidal_tiff(dst, arr.astype(np.uint8), geo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("."))
    a = ap.parse_args()
    os.environ.setdefault("OWL_MODEL", str(OWL / "model"))
    sys.path.insert(0, str(OWL))
    import inference

    wsi = a.data_root / "images" / f"{a.case}.tif"
    mask = a.data_root / "tissue-masks" / f"{a.case}_tissue.tif"
    model = inference.init_model()
    sd_years = float(model["cohort"].years.std(ddof=1))
    a.work.mkdir(parents=True, exist_ok=True)
    fail = []

    r1 = inference.predict_case(model, wsi, mask, a.work / "run1", ("years",))
    r2 = inference.predict_case(model, wsi, mask, a.work / "run2", ("years",))
    dz = abs(r1["years"] - r2["years"]) / sd_years
    print(f"determinism: {r1['years']:.6f} vs {r2['years']:.6f} years, |dz| {dz:.2e}")
    if dz > 0.01:
        fail.append(f"two identical runs differ by |dz| {dz:.4f}")

    with tifffile.TiffFile(mask) as t:
        m0 = t.series[0].levels[0].asarray()
    half = m0.copy()
    half[:, : m0.shape[1] // 2] = 0
    write_mask_like(mask, a.work / "half_mask.tif", half, inference)
    rh = inference.predict_case(model, wsi, a.work / "half_mask.tif", a.work / "half", ("years",))
    dzh = abs(rh["years"] - r1["years"]) / sd_years
    print(f"half mask: {rh['n_tiles']} of {r1['n_tiles']} tiles, {rh['years']:.4f} years, |dz| {dzh:.3f}")
    if rh["n_tiles"] >= r1["n_tiles"] or dzh == 0:
        fail.append("removing half of the mask did not change the tiles or the prediction")

    write_mask_like(mask, a.work / "empty_mask.tif", np.zeros_like(m0), inference)
    try:
        re = inference.predict_case(model, wsi, a.work / "empty_mask.tif", a.work / "empty")
        print(f"empty mask: {re['n_tiles']} tiles, {re['years']:.3f} years, {re['risk_class']} (cohort median)")
    except Exception as e:  # noqa: BLE001
        fail.append(f"empty mask crashed: {e!r}")
    print("SELFTEST: " + ("OK" if not fail else "FAILED\n  - " + "\n  - ".join(fail)))
    return int(bool(fail))


if __name__ == "__main__":
    raise SystemExit(main())
