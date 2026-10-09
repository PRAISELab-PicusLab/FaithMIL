# Multi-scale trajectory MIL for BCR survival

A **trajectory** is the sequence of patches centred on the same tissue point,
from coarse to fine (L5 → L0). With 224 px patches and the 4× pyramid, a
trajectory covers 55 mm → 14 mm → 3.5 mm → 0.87 mm → 217 µm → 54 µm, which is
what a pathologist sees while zooming in. Each patient is a **bag of
trajectories**.

```
anchors (tissue grid) ──► patches L5..L0 ──► frozen encoder (cached, per level)
    ──► level adapters (trainable, one per level)
    ──► trajectory aggregator: mlp | transformer | gru | lstm   → h_k
    ──► local score s_k  +  reliability weight a_k = softmax(gated attention)
    ──► patient risk R = Σ a_k s_k  ──► Cox loss
    ──► Breslow → time to recurrence (RMST), risk group; Shapley → attribution map + report
```

## Design decisions (and why)

| Decision | Reason |
|---|---|
| **Encoders frozen, features cached once** | 508 patients / 87 events: fine-tuning 6 encoders end-to-end overfits (the thumbnail baseline already overfit with one image per patient), and every epoch would re-read ~4M patches from 2 TB. Caching makes an epoch take seconds, so the full **10-fold CV stays affordable**: no need to fall back to a single noisy holdout. |
| **Trainable: per-level adapters + aggregator + attention + head** (~0.3–1.5 M params) | The per-level adapter is the level-specific "encoder". Different frozen backbones per level are supported (`--level-backbones 0:prost40m,1:prost40m`). |
| **L5 synthesized from L4 for all 508 slides** | Verified: the scanner builds the pyramid by nearest decimation at phase 0 (native L5 vs `L4[::4, ::4]`: r = 0.98; 4×4 box averaging: 0.73). So L5 exists for every patient, the L5 adapter trains on all data, and pyramid depth (an acquisition artifact) cannot leak into the model. `--max-native-level 5` uses the native L5 where it exists. |
| **Grid anchors, stride = one L2 patch (0.87 mm)** | L2 patches tile the tissue; L1/L0 are zoomed samples at each tile centre. About 1,230 trajectories per slide, ~650k in total. The trajectory is kept only if every patch has tissue (L0 ≥ 50%, L1 ≥ 25%, L2 ≥ 10%, coarser > 0). Coarse levels are not thresholded, so tissue borders (surgical margins) are kept. |
| **One anchor set for every level configuration** | `L5-L4` vs `+L3` vs … differ only in the levels, never in the sampled tissue. |
| **Score × softmax reliability** | With a linear score head, `Σ a_k s_k = wᵀ(Σ a_k h_k) + b`, i.e. exactly gated attention-MIL. The per-trajectory scores come for free. `--pooling mean` is the "no reliability" ablation. |
| **Shapley values over trajectories** | `v(S)` = the model's risk on a subset of trajectories. The values sum exactly to `R − mean training risk`, their sign says risk-increasing vs protective, and they probe the same "remove a region" operation as the challenge's faithfulness test. They cost ~0.1 s per slide (prefix sums + antithetic permutations). |
| **RMST as time to recurrence** | 5-year BCR-free survival is 86%, so the median survival time is undefined for most patients. RMST is finite and strictly monotone in risk, so C-index(time) = C-index(risk). |

## Staged run (commands)

Launched from the data root (see the main README).

```bash
# 1. anchors for all slides (CPU). QC figures in trajectories/anchors/<set>/qc/
python owl_challenge_wacv2027/01_splits_and_trajectories/scripts/build_trajectory_anchors.py --workers 16 --qc 10

# 2. frozen features, one level at a time (resumable; --shard/--num-shards split the slides)
python owl_challenge_wacv2027/03_multiscale_trajectories/extract_trajectory_features.py --backbone imagenet --levels 5 4
python owl_challenge_wacv2027/03_multiscale_trajectories/extract_trajectory_features.py --backbone imagenet --levels 3 2
python owl_challenge_wacv2027/03_multiscale_trajectories/extract_trajectory_features.py --backbone imagenet --levels 1 0 --shard 0 --num-shards 16

# 3. trajectory MIL, 10-fold CV (all folds in one process), adding levels step by step
python owl_challenge_wacv2027/03_multiscale_trajectories/train_trajectory.py --folds all --levels 5 4
python owl_challenge_wacv2027/03_multiscale_trajectories/train_trajectory.py --folds all --levels 5 4 3 2 --aggregator gru
python owl_challenge_wacv2027/03_multiscale_trajectories/train_trajectory.py --split-mode holdout --levels 5 4                # folds 0-6 / 7 / 8-9
python owl_challenge_wacv2027/03_multiscale_trajectories/scripts/summarize_trajectory_results.py                             # pooled OOF C-index per run

# 4. per-level linear probe: pool each slide's trajectories, then nested-CV ridge Cox
python owl_challenge_wacv2027/03_multiscale_trajectories/scripts/pool_trajectory_features.py --backbone imagenet --levels 2 1
python owl_challenge_wacv2027/02_thumbnail_backbone_survival/scripts/linear_probe_survival.py --backbone imagenet_L1_mean \
    --features-dir features_traj --results-dir results_probe_traj

# 5. permutation null for a probe configuration: (event, time) shuffled within each fold
python owl_challenge_wacv2027/03_multiscale_trajectories/scripts/make_permuted_splits.py --n 30 --out-dir splits_perm
python owl_challenge_wacv2027/02_thumbnail_backbone_survival/scripts/linear_probe_survival.py --backbone imagenet_L1_mean --features-dir features_traj \
    --splits-csv splits_perm/perm_000.csv --results-dir results_probe_perm/perm_000   # ... one per permutation

# 6. the 4 challenge outputs for out-of-fold cases
python owl_challenge_wacv2027/03_multiscale_trajectories/make_challenge_outputs.py --run-dir results/trajectory/<run> --max-cases 5
```

## Outputs of a run (`results/trajectory/<run>/fold_k/`)

`all_predictions.csv` / `test_predictions.csv` (risk_score, predicted_time_years,
risk_group, attention_eff_n), `metrics.json`, `history.csv`, `model.pt` (weights +
Breslow baseline + group thresholds), `level_occlusion.csv`, and
`trajectory_outputs/<case>.npz` (local scores, weights, Shapley values).
`make_challenge_outputs.py` turns these into `time-to-recurrence.json`,
`risk-group.json`, `attribution-map.tif` (uint8 pyramid with the mask's exact
layout and spacing) and `report.pdf` (5 pages).

## Files

| File | Role |
|---|---|
| `extract_trajectory_features.py` | reads the patches of every trajectory at the requested levels and encodes them with a frozen backbone; one `(N_trajectories, D)` array per slide and level |
| `train_trajectory.py` | trajectory MIL (CV or holdout), then Breslow/RMST, risk groups, Shapley values and level occlusion for the test cases |
| `make_challenge_outputs.py` | time-to-recurrence and risk-group JSON, attribution-map TIFF, PDF report |
| `configs/trajectory.yaml` | defaults of `train_trajectory.py` (every field can be overridden from the command line) |
| `src/trajectory/bank.py` | in-memory feature bank, bag sampling |
| `src/trajectory/model.py` | level adapters, aggregators, gated attention, patient risk |
| `src/trajectory/survival.py` | Breslow baseline, RMST, risk groups |
| `src/trajectory/explain.py` | Shapley values over trajectories, attention contributions, level occlusion |
| `src/trajectory/attribution_map.py` | uint8 pyramidal TIFF with the tissue mask's shape, levels and spacing |
| `src/trajectory/report.py` | 5-page PDF report |
| `scripts/pool_trajectory_features.py` | per-slide mean / max / top-k pooling of one level's features, in the linear-probe input format (+ random-feature controls) |
| `scripts/make_permuted_splits.py` | copies of `folds.csv` with (event, time) shuffled within each fold |
| `scripts/summarize_trajectory_results.py` | table of pooled and per-fold C-index across runs |

Symlinked from `01_splits_and_trajectories` (edit them there):
`src/trajectory/{__init__,wsi,anchors,io}.py`. The anchors themselves are built by
`01_splits_and_trajectories/scripts/build_trajectory_anchors.py`.

Symlinked from `02_thumbnail_backbone_survival` (edit them there):
`src/{__init__,backbones,dataset,losses,metrics,utils}.py` and
`scripts/linear_probe_survival.py` (the nested-CV ridge-Cox probe reused on pooled
trajectory features).
