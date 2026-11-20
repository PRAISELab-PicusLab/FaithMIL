#!/usr/bin/env bash
# The sequence of commands of the paper, in order. Every step is resumable (finished cases / epochs are skipped), so the
# script can be started again after an interruption. Run it from the repository root; the data root is $TRACE_DATA_ROOT
# (images/, tissue-masks/ and the label files of the README must be in place). Steps 3-4 need one GPU, step 5 benefits from it.
set -euo pipefail
export TRACE_DATA_ROOT="${TRACE_DATA_ROOT:?set TRACE_DATA_ROOT to the data root}"
PY=python
S=scripts

# 0. cases and the folds of the paper
$PY $S/00_prepare_data.py --labels "$TRACE_DATA_ROOT"/labels/*.csv

# 1. tissue anchors: the L2 / L1 tree and the grids of levels 0 and 3
$PY $S/01_build_anchors.py --mode tree
$PY $S/01_build_anchors.py --mode grid --level 3
$PY $S/01_build_anchors.py --mode grid --level 0

# 2. raw tiles for the self-supervised pre-training of each level
$PY $S/02_extract_tiles.py --level 2
$PY $S/02_extract_tiles.py --level 1 --cap 3000
$PY $S/02_extract_tiles.py --level 3 --cap 1500 --per-cell 25
$PY $S/02_extract_tiles.py --level 0 --cap 3000

# 3. one DINO encoder per level (123 / 18 / 68 / 16 epochs for levels 2 / 1 / 3 / 0)
for L in 2 1 3 0; do $PY $S/03_train_dino.py --level $L; done

# 4. frozen features
for V in children parents ctx2 ctx3 grid3 grid0; do $PY $S/04_extract_features.py --encoder dino --view $V; done
for E in hoptimus0 prost40m; do                      # foundation-model baselines of Table 1
  for V in children parents grid3 grid0; do $PY $S/04_extract_features.py --encoder $E --view $V; done
done
$PY $S/04_extract_features.py --encoder quiltnet --view children    # concepts of the report

# 5. Table 1: single levels, with the three encoders
for E in dino hoptimus0 prost40m; do
  $PY $S/05_train.py --run-name L1_$E --views children --encoder $E --save-models
  $PY $S/05_train.py --run-name L2_$E --views parents  --encoder $E --save-models
  $PY $S/05_train.py --run-name L3_$E --views grid3    --encoder $E
  $PY $S/05_train.py --run-name L0_$E --views grid0    --encoder $E --pool 8192
done

# 6. Table 2: context-enriched models, without and with the faithfulness-aware terms (two trainings each)
$PY $S/05_train.py --run-name ctx2_dino --views children ctx2      --save-models
$PY $S/05_train.py --run-name ctx3_dino --views children ctx2 ctx3 --save-models
FAITH="--suff-cox 0.5 --nec-var 0.5 --ent 0.3 --topk 0.1 --n-seeds 3 --save-models"
for R in "" "_r2"; do
  SEED=$([ -z "$R" ] && echo 42 || echo 1042)
  $PY $S/05_train.py --run-name faith_L1_dino$R   --views children          --seed $SEED $FAITH
  $PY $S/05_train.py --run-name faith_L2_dino$R   --views parents           --seed $SEED $FAITH
  $PY $S/05_train.py --run-name faith_ctx2_dino$R --views children ctx2     --seed $SEED $FAITH
  $PY $S/05_train.py --run-name faith_ctx3_dino$R --views children ctx2 ctx3 --seed $SEED $FAITH
  $PY $S/05_train.py --run-name faith_ctx3_warm_dino$R --views children ctx2 ctx3 --seed $SEED $FAITH --faith-start-epoch 10
done

# 6b. Table 2: established MIL baselines (same features, folds and protocol; C-index only)
$PY $S/05_train.py --run-name mean_dino     --arch mean     --views children
$PY $S/05_train.py --run-name clam_sb_dino  --arch clam_sb  --views children
$PY $S/05_train.py --run-name transmil_dino --arch transmil --views parents

# 7. Table 3: deletion / insertion at equal tissue area
for R in L1_dino L2_dino ctx2_dino ctx3_dino \
         faith_L1_dino faith_L2_dino faith_ctx2_dino faith_ctx3_dino faith_ctx3_warm_dino \
         faith_L1_dino_r2 faith_L2_dino_r2 faith_ctx2_dino_r2 faith_ctx3_dino_r2 faith_ctx3_warm_dino_r2; do
  $PY $S/06_faithfulness.py --run $R
done

# 8. report: ontology terms and concepts, patterns, one patient (edit --cases)
$PY $S/07_ontology_concepts.py --ncit "$TRACE_DATA_ROOT"/ontology/Thesaurus.FLAT.zip --ncit-version 26.09d \
    --uberon "$TRACE_DATA_ROOT"/ontology/uberon-basic.obo
$PY $S/08_explain.py --run faith_ctx3_dino
$PY $S/09_make_reports.py --run faith_ctx3_dino --cases TCGA-KK-A8IC --figures

# 9. tables
$PY $S/10_tables.py
