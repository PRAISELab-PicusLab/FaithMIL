"""Run the OWL algorithm on one training case without Docker, then check the 4 outputs.

The case is staged as Grand Challenge does it: symlinks named by a UUID under <work>/input/images/<socket>/, plus
inputs.json. Outputs go to <work>/output/.

    python owl/run_local.py --wsi images/case_radboud_0466.tif --mask tissue-masks/case_radboud_0466_tissue.tif \
        --work /path/to/scratch/owl_test
    python owl/run_local.py --check-only --work /path/to/scratch/owl_test   # re-check an existing output
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import tifffile

OWL = Path(__file__).resolve().parent
SOCKETS = {
    "prostatectomy-tissue-whole-slide-image": "images/prostatectomy-wsi",
    "prostatectomy-tissue-mask": "images/prostatectomy-tissue-mask",
}


def stage(wsi: Path, mask: Path, inp: Path, copy: bool = False) -> None:
    """Grand Challenge input layout in `inp`: images/<socket>/<uuid>.tif and inputs.json. Symlinks, or copies
    (for owl/test/input/interf0 of do_test_run.sh: Docker does not follow symlinks out of the mount)."""
    shutil.rmtree(inp, ignore_errors=True)
    entries = []
    for (slug, rel), src in zip(SOCKETS.items(), (wsi, mask)):
        d = inp / rel
        d.mkdir(parents=True)
        dst = d / f"{uuid.uuid4()}{src.suffix}"
        if copy:
            shutil.copyfile(src, dst)
        else:
            dst.symlink_to(src.resolve())
        entries.append({"socket": {"slug": slug, "relative_path": rel, "is_image_kind": True, "is_panimg_kind": True},
                        "file": None, "image": {"name": src.name}, "value": None})
    (inp / "inputs.json").write_text(json.dumps(entries, indent=4))


def check(out: Path, mask: Path) -> list[str]:
    """The output checks of the submission checklist that can be done on files. Returns the failures."""
    fail = []
    years = json.loads((out / "overall-survival-years.json").read_text())
    if not (isinstance(years, float) and np.isfinite(years) and years > 0):
        fail.append(f"overall-survival-years.json: {years!r} is not a positive float")
    cls = json.loads((out / "prostate-cancer-risk-classification.json").read_text())
    if cls not in ("low", "medium", "high"):
        fail.append(f"prostate-cancer-risk-classification.json: {cls!r}")

    hm = sorted((out / "images/prostatectomy-overall-survival-attribution-heatmap").iterdir())
    if len(hm) != 1 or hm[0].suffix != ".tif":
        fail.append(f"heatmap folder must hold one .tif, has {[p.name for p in hm]}")
    else:
        with tifffile.TiffFile(hm[0]) as t, tifffile.TiffFile(mask) as m:
            hl, ml = t.series[0].levels, m.series[0].levels
            h0, m0 = hl[0].pages[0], ml[0].pages[0]
            if hl[0].dtype != np.uint8:
                fail.append(f"heatmap dtype {hl[0].dtype}")
            if hl[0].shape != ml[0].shape:
                fail.append(f"heatmap shape {hl[0].shape} != mask L0 {ml[0].shape}")
            for tag in ("XResolution", "YResolution", "ResolutionUnit"):
                hv = h0.tags[tag].value if tag in h0.tags else None
                mv = m0.tags[tag].value if tag in m0.tags else None
                if hv != mv:
                    fail.append(f"heatmap {tag} {hv} != mask {mv}")
            if len(hl) < 2 or not h0.is_tiled:
                fail.append(f"heatmap is not a tiled pyramid ({len(hl)} levels)")
            k = min(2, len(hl) - 1)  # statistics on a reduced level: L0 of a large slide is ~0.5 GB
            a = hl[k].asarray()
            print(f"  heatmap: {len(hl)} levels, L0 {hl[0].shape}, values {a.min()}..{a.max()}, "
                  f"{(a > 0).mean():.1%} of pixels > 0 (on L{k})")

    pdf = (out / "pathology-report.pdf").read_bytes()
    n_pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf))
    if not pdf.startswith(b"%PDF") or not 1 <= n_pages <= 5:
        fail.append(f"pathology-report.pdf: header {pdf[:5]!r}, {n_pages} pages")
    print(f"  years {years}, class {cls!r}, report {n_pages} page(s), {len(pdf) / 1e3:.0f} kB")
    return fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", type=Path)
    ap.add_argument("--mask", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=OWL / "model")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--stage-only", action="store_true",
                    help="only copy the case into --work as an interface folder (e.g. owl/test/input/interf0)")
    a = ap.parse_args()

    if a.stage_only:
        stage(a.wsi, a.mask, a.work, copy=True)
        print(f"staged {a.wsi.name} + {a.mask.name} in {a.work}")
        return 0
    if not a.check_only:
        shutil.rmtree(a.work / "output", ignore_errors=True)
        stage(a.wsi, a.mask, a.work / "input")
        os.environ.update(OWL_INPUT=str(a.work / "input"), OWL_OUTPUT=str(a.work / "output"), OWL_MODEL=str(a.model))
        sys.path.insert(0, str(OWL))
        import inference

        t0 = time.time()
        model = inference.init_model()
        t1 = time.time()
        inference.run(model)
        print(f"init {t1 - t0:.1f}s, run {time.time() - t1:.1f}s")

    fail = check(a.work / "output", a.mask)
    print("OUTPUT CHECK: " + ("OK" if not fail else "FAILED\n  - " + "\n  - ".join(fail)))
    return int(bool(fail))


if __name__ == "__main__":
    raise SystemExit(main())
