# TRACE: Tree-Based Regional Attribution with Context Enrichment for Prostate Cancer Recurrence Prediction

TRACE predicts biochemical recurrence (BCR) after radical prostatectomy from H&E whole-slide images using a context-enriched, additive multiple-instance learning (MIL) model. It combines low-magnification tissue regions with higher-resolution child patches and decomposes patient risk into regional contributions.
The paper evaluates the model using patient-level 10-fold cross-validation across 508 RUMC patients and 298 TCGA-PRAD patients. Both cohorts participate in training and testing across the folds; TCGA-PRAD is a second test population, rather than an external validation cohort.

## Framework

The proposed L2–L1 tree combines L2 parent regions (approximately 0.87 mm, 3.9 µm/px) with their L1 child patches (approximately 0.22 mm, 0.97 µm/px). A shared frozen encoder embeds both scales. Trainable adapters and gated attention pool the child features, which are fused with the parent features to form a context-enriched regional representation.
Each region receives a local score and a patient-level attention weight. Patient risk is their weighted sum, trained with the Cox partial likelihood, and therefore has an exact decomposition into regional contributions. The explanation pipeline compares regional attribution methods through deletion tests and uses region-level Shapley values for the attribution map and key-region selection. Patient reports distinguish model contribution, recurring cohort tissue pattern, and histological concept match, with abstention when concept evidence is insufficient.

## Repository structure
The repository retains the numbered directories used during development. Thumbnail and trajectory experiments accompany the multi-scale pipeline described in the paper.

| Folder | What it does |
|---|---|
| `01_splits_and_trajectories/` | Patient-level 10-fold stratified splits (`scripts/create_folds.py`, fold summaries and plots) and the trajectory anchors: a tissue-filtered grid of points on each slide, with the pyramid reader and tissue-mask utilities (`scripts/build_trajectory_anchors.py`, `src/trajectory/{wsi,anchors,io}.py`) |
| `02_thumbnail_backbone_survival/` | Survival from the lowest-resolution whole-slide thumbnail: thumbnail extraction, a 14-backbone registry, fine-tuning (`train.py`, `run_cv.py`) and frozen linear probe (`scripts/extract_features.py`, `scripts/linear_probe_survival.py`), backbone comparison |
| `03_multiscale_trajectories/` | Multi-scale trajectories (patches at L5 … L0 centred on the same point): per-level feature extraction, trajectory MIL (`train_trajectory.py`), per-level pooled linear probe, permutation null, challenge outputs (attribution map + PDF report). See its `README.md` |

## How paths are resolved

Every default input/output path in the scripts (`splits/folds.csv`,
`training_labels.csv`, `images/`, `tissue-masks/`, `thumbnails/`, `pretrained/`,
`trajectories/`, `results/`, ...) is **relative to the directory the command is
launched from**. Launch from the directory that holds the data and call the
scripts by path:

```bash
cd <data root>
python owl_challenge_wacv2027/01_splits_and_trajectories/scripts/build_trajectory_anchors.py --max-cases 5
python owl_challenge_wacv2027/03_multiscale_trajectories/train_trajectory.py --folds all --levels 5 4
```

- `OWL_DATA_ROOT` overrides the launch directory for `src/trajectory/io.py`.
- Default configs are read from the folder of the script (`<folder>/configs/`).

## Shared code

`03_multiscale_trajectories` reuses modules from the other two folders through
relative symlinks, so each file exists once and imports stay `from src... import ...`:

- from `01`: `src/trajectory/{__init__,wsi,anchors,io}.py` (pyramid reader, anchors, on-disk layout);
- from `02`: `src/{__init__,backbones,dataset,losses,metrics,utils}.py` (backbone
  registry, train/val split, Cox loss, metrics) and `scripts/linear_probe_survival.py`.

Edit the original file in `01` or `02`.

## Environments

`02_thumbnail_backbone_survival/requirements-train.txt` (torch, timm, open_clip,
transformers, lifelines, tifffile, ...) runs every script. Thumbnail extraction
alone also works with `02_thumbnail_backbone_survival/thumbnail_extraction/requirements.txt`.
