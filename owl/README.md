# TRACE for the OWL challenge (Grand Challenge container)

Container of the TRACE submission to the [OWL challenge](https://owl.grand-challenge.org/) (WACV 2027): from one
prostatectomy H&E slide and its tissue mask it predicts the time to biochemical recurrence (BCR), a risk class, an
attribution heatmap and a pathology report.

## The model

- **Instances**: the 0.22 mm tissue tiles of level 1 (224 px at about 0.97 um/px), selected from the provided tissue
  mask only (a tile is kept if it is at least 50% tissue), exactly as in training (`trace_mil.anchors.tree_anchors`).
- **Encoder**: a ViT-S/14 trained from random initialisation with DINO on the L1 tiles of the OWL training slides
  (no labels, no external pre-trained weights).
- **Aggregator**: gated-attention MIL with a linear score head, risk `R = sum_i a_i s_i` (exactly additive over tiles).
  Ensemble of 30 members (10 cross-validation folds x 3 seeds) trained with the Cox loss on OWL Radboudumc (508) and
  TCGA-PRAD (298); each member is z-scored with the risks of its own training patients and the z are averaged.
- **Outputs**
  - `overall-survival-years.json`: Weibull AFT median time to BCR fitted on the out-of-fold risks of the 508 OWL
    patients; it decreases with the risk, so the C-index of the years is the C-index of the model (0.666 out-of-fold).
  - `prostate-cancer-risk-classification.json`: tertiles of the out-of-fold predicted years (low / medium / high).
  - heatmap: the exact effect on the prediction of removing each tile (leave-one-out, in the direction that moves the
    prediction most on the exact deletion curve of the model), on the mask L0 grid, uint8, pyramidal TIFF.
  - `pathology-report.pdf`: 3 pages; prediction and cohort context, the tiles that raise / lower the risk with their
    exact contributions, and how to read the report.
- No Gleason / ISUP / Grade Group information is used anywhere (input, target, intermediate step or encoder).

## Files

| File | What it is |
|---|---|
| `Dockerfile`, `app.py`, `do_build.sh`, `do_test_run.sh`, `do_save.sh` | from the challenge starter kit; `app.py` unchanged; the build context is the repo root (the image needs `trace_mil/`) |
| `inference.py` | `init_model()` / `run()` of the invoke API; also `get_args_parser()` / `run(args)` for `faithfulness_blackbox.ipynb` |
| `owl_model.py` | tile reading, encoder, ensemble, exact attributions |
| `owl_report.py` | the PDF report |
| `export_deploy.py` | trained run -> `model/` (weights, calibration, cohort reference) |
| `run_local.py`, `selftest.py`, `verify_port.py`, `faithfulness_local.py` | checks without Docker (see below) |
| `model/` | becomes `model.tar.gz` (not in git: weights and OWL training labels) |

## Build and submit

On a machine with Docker (a GPU is optional for the local test), with `owl/model/` and `owl/test/input/interf0/`
copied next to the code:

```bash
cd owl
./do_test_run.sh     # builds, starts the server on an isolated network, /health then /invoke on test/input/interf0
python run_local.py --check-only --mask test/input/interf0/images/prostatectomy-tissue-mask/*.tif --work test/check
./do_save.sh         # -> trace_owl_<date>.tar.gz (image) and model.tar.gz (weights)
```

`run_local.py --check-only` expects the outputs in `<work>/output`: copy `test/output/interf0/` there, or use the
checks listed in the submission checklist directly on `test/output/interf0/`.

On Grand Challenge: the image goes to *Algorithm > Containers*, `model.tar.gz` to *Algorithm > Models* (extracted to
`/opt/ml/model`). Run a sanity check before the single test submission, with the same image.

## Checks without Docker (cluster)

```bash
python owl/export_deploy.py --data-root /path/to/data                 # model/ from the trained run
python owl/verify_port.py --data-root . --cases case_radboud_0466     # exported model == trained model
python owl/run_local.py --wsi images/X.tif --mask tissue-masks/X_tissue.tif --work /tmp/x   # 4 outputs + checks
python owl/selftest.py --case case_radboud_0466 --work /tmp/st         # determinism, half / empty mask
python owl/faithfulness_local.py --cases ... --work /tmp/faith          # the challenge faithfulness test, locally
python owl/run_local.py --stage-only --wsi ... --mask ... --work owl/test/input/interf0   # test case for Docker
```

## Faithfulness notebook

`inference.py` can be the black box of `faithfulness_blackbox.ipynb` directly: set `INFERENCE_PY` to it, and
`MODEL_ARGS = ["--wsi-dir", str(MODEL_WSI_DIR), "--mask-dir", str(MODEL_MASK_DIR), "--output-json", str(MODEL_OUTPUT_JSON)]`,
with `OWL_MODEL` pointing at the extracted `model/`. The model is loaded once per process; no state is kept between runs.
