# Thumbnail extraction

Extracts aligned WSI/tissue-mask thumbnail pairs at pyramid levels `L4`, `L5`
and `nativeLast` for the 508-case OWL cohort. Uses `tifffile` + `imagecodecs`
only (no OpenSlide).

## Requirements

`requirements.txt` (tifffile, imagecodecs, numpy, pandas, scipy, Pillow).

## 1. Dry run on a handful of cases

Test on a small sample before the full batch:

```bash
python3 extract_thumbnails.py \
    --output-dir /path/to/thumbnails_test \
    --cases case_radboud_0000,case_radboud_0001,case_radboud_0002,case_radboud_0003,case_radboud_0007
```

`case_radboud_0000/0001/0002` have 6 WSI pyramid levels (L5 present);
`case_radboud_0003/0007` have 5 (no L5) — this mix exercises both branches of
the mask level-correspondence logic. Inspect
`<output-dir>/{L4,L5,nativeLast}/*_metadata.csv` and the log file under
`<output-dir>/logs/` before trusting the full run.

## 2. Full batch

```bash
python3 extract_thumbnails.py --images-dir <images> --masks-dir <tissue-masks> \
    --output-dir <thumbnails> --workers 8
python3 contingency_report.py --metadata-csv <thumbnails>/nativeLast/nativeLast_metadata.csv \
    --labels-csv <training_labels.csv>
```

`extract_thumbnails.py` processes all discovered cases; `contingency_report.py`
then checks for a scale confound between `n_levels_wsi` and `event`.

`--workers` controls multiprocessing parallelism; extraction is I/O bound on
the shared GPFS filesystem, so the default of 8 is deliberately conservative —
pushing it much higher can *reduce* throughput rather than increase it.

## 3. Re-running / idempotency

Re-running `extract_thumbnails.py` skips any case/folder whose PNG pair
already exists (status `skip` in the final report and log). Pass `--force` to
re-extract everything regardless. A single case's failure is logged and does
not stop the batch; failed cases are listed at the end of the log.

## Output layout

```
<output-dir>/
  L4/
    case_radboud_XXXX.png          # WSI level 4, all 508 cases
    masks/case_radboud_XXXX.png
    L4_metadata.csv
  L5/
    case_radboud_XXXX.png          # WSI level 5, only the 275 cases that have it
    masks/case_radboud_XXXX.png
    L5_metadata.csv
  nativeLast/
    case_radboud_XXXX.png          # L5 if the slide has 6 levels, else L4 (no resize)
    masks/case_radboud_XXXX.png
    nativeLast_metadata.csv
  logs/
    extract_<timestamp>.log
```

Metadata CSV columns: `case_id, level_used, n_levels_wsi, n_levels_mask,
width, height, mpp_effective, mask_level_used, mask_resized, shapes_match,
tissue_fraction`.

`mpp_effective` is read directly from each pyramid level's own TIFF resolution
tags (not assumed from a clean 4^level formula), so it's the ground truth for
that specific level/case. In `nativeLast` it takes one of two values (~62.09
or ~248.36 µm/px) depending on how many pyramid levels the slide has — this is
an acquisition artifact, not a biological signal, and must stay visible in any
downstream analysis. Run `contingency_report.py` to test whether it correlates
with `event`; if it does, any model trained on `nativeLast` risks learning
that confound instead of real tissue signal.

## Mask alignment logic

WSI levels downsample 4x/level; mask levels downsample 2x/level with mask L0
== WSI L2, so WSI level `k` should align with mask level `j = 2k - 4`. This
holds exactly for `k=4` (mask L4) on all 508 cases. For `k=5`, `j=6` doesn't
exist (masks only go up to L5), so the nearest available mask level (L5) is
used and rescaled to the WSI level's exact shape with nearest-neighbor
interpolation (`mask_resized=True`).

The predicted level is never trusted blindly — decoded shapes are compared
directly (1px tolerance) and any mismatch is logged as a warning instead of
being silently patched, per `shapes_match` in the metadata.

## Files

- `extract_thumbnails.py` — main extraction script (argparse, multiprocessing)
- `contingency_report.py` — n_levels_wsi x event contingency table + chi-square test
- `requirements.txt` — pinned dependency versions
