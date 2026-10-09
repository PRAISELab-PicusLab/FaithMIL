"""Multi-scale trajectory MIL for BCR survival (OWL challenge).

A *trajectory* is the ordered sequence of patches centred on the same tissue
point (x, y) at pyramid levels coarse -> fine (L5 ... L0). A patient is a bag
of trajectories. Pipeline:

    anchors.py       tissue-filtered grid of anchor points per slide (numpy only)
    wsi.py           pyramid geometry, tile-level region reader, tissue mask
    (extract_trajectory_features.py)  frozen encoder -> cached per-level features
    bank.py          cached features -> padded (B, K, L, D) bags
    model.py         level adapters + trajectory aggregator + reliability
                     attention -> local scores + patient risk
    survival.py      Breslow baseline, time-to-recurrence (RMST), risk groups
    explain.py       Shapley values over trajectories, level occlusion
    attribution_map.py / report.py   challenge outputs (TIFF map, PDF report)
"""
