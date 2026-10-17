#!/usr/bin/env python3
"""Frozen features of the tiles of one view with one encoder (see trace_mil/features.py). Needs a GPU for speed.

    python scripts/04_extract_features.py --encoder dino --view children      # the L1 tiles of the tree
    python scripts/04_extract_features.py --encoder dino --view parents       # the L2 tiles of the tree
    python scripts/04_extract_features.py --encoder dino --view ctx2          # centred L2 context of every L1 tile
    python scripts/04_extract_features.py --encoder dino --view ctx3          # centred L3 context of every L1 tile
    python scripts/04_extract_features.py --encoder dino --view grid3         # (also grid0)
    python scripts/04_extract_features.py --encoder hoptimus0 --view children # foundation-model baselines
    python scripts/04_extract_features.py --encoder quiltnet --view children   # image embeddings for the report concepts

`--shard i --num-shards n` splits the cases between n independent processes; finished cases are skipped.
"""
import _repo  # noqa: F401
import argparse

import pandas as pd

from trace_mil import features
from trace_mil.layout import FEATURE_VIEWS, Layout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--encoder", choices=features.ENCODERS, required=True)
    ap.add_argument("--view", choices=list(FEATURE_VIEWS), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="*", default=None)
    a = ap.parse_args()
    lay = Layout(a.root)
    cases = (a.cases or sorted(pd.read_csv(lay.folds_csv).case_id))[a.shard::a.num_shards]
    features.run(lay, a.encoder, a.view, cases, a.workers)


if __name__ == "__main__":
    main()
