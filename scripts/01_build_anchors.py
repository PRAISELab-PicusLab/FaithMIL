#!/usr/bin/env python3
"""Tile anchors from the tissue masks (see trace_mil/anchors.py).

    python scripts/01_build_anchors.py --mode tree                  # L2 parents and their L1 children
    python scripts/01_build_anchors.py --mode grid --level 3        # single-level grid (also --level 0)
"""
import _repo  # noqa: F401
import argparse

import pandas as pd

from trace_mil.anchors import build
from trace_mil.layout import Layout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--mode", choices=["tree", "grid"], required=True)
    ap.add_argument("--level", type=int, default=None, help="grid level (0..5)")
    ap.add_argument("--min-tissue", type=float, default=None, help="override the tissue threshold")
    ap.add_argument("--mask-level", type=int, default=1)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.mode == "grid" and a.level is None:
        ap.error("--level is required for --mode grid")
    lay = Layout(a.root)
    cases = a.cases or sorted(pd.read_csv(lay.folds_csv).case_id)
    s = build(lay, cases, a.mode, a.level, a.min_tissue, a.mask_level, a.workers)
    cols = ["n_parents", "n_children"] if a.mode == "tree" else ["n_anchors"]
    print(s[cols].describe().loc[["min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
